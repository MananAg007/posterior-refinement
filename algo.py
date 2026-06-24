import os
import collections
import copy
import json
import pickle
import hydra.utils

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import trainer_base
import utils
import math
import models
from torch.func import functional_call
from models.dit import modulate_fused
import functools
from entmax import entmax_bisect


def stopgrad(x):
    """Stop gradient for x."""
    return x.detach()

class FLMBase(trainer_base.TrainerBase):
    """Base class for FLM/FMLM.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max
        self.lut_a2g, self.lut_g2a = utils.build_luts(K=self.vocab_size)
        self._is_resuming = (
            config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
        )

    def _validate_configuration(self):
        pass

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def _process_sigma(self, sigma):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _process_model_output(self, model_output, xt, sigma):
        """tanh logit softcap (cap=30) followed by log_softmax over vocab."""
        del xt, sigma
        cap = 30.0
        model_output = cap * torch.tanh(model_output / cap)
        return model_output.log_softmax(dim=-1)

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def _loss(self, x0, valid_tokens,
              current_accumulation_step=None,
              train_mode=False,
              xT=None, given_t=None, not_sampling_t=False):
        """Override to always dispatch to self.loss() for all FLM classes."""
        (input_tokens, output_tokens,
         valid_tokens) = self._process_model_input(x0, valid_tokens)
        loss = self.loss(input_tokens, output_tokens,
                         current_accumulation_step, train_mode,
                         xT=xT, given_t=given_t,
                         not_sampling_t=not_sampling_t)
        assert loss.ndim == 2
        if self.ignore_bos:
            loss[:, 1:] = loss[:, 1:]
            valid_tokens[:, 1:] = valid_tokens[:, 1:]

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens
        return trainer_base.Loss(loss=token_nll,
                                 nlls=nlls,
                                 prior_loss=0.0,
                                 num_tokens=num_tokens)

    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        raise NotImplementedError

    def nll(self, input_tokens, output_tokens,
            current_accumulation_step=None, train_mode=False):
        raise NotImplementedError

    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max
        if accum_step is not None:
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
            perm = torch.randperm(n, device=self.device)
            _eps_t = _eps_t[perm]
        t = (t_max - t_min) * _eps_t + t_min
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            t = t[:batch_dim]
        return t

    def _tau_to_t(self, tau):
        """Convert alpha-space time tau to gamma-space time t.

        When ``algo.skip_time_reparam`` is True (FLM+/FMLM+ today), the
        reparam is the identity. Otherwise applies the LUT
        ``utils.alpha_to_gamma``.
        """
        if getattr(self.config.algo, 'skip_time_reparam', False):
            return tau
        return utils.alpha_to_gamma(tau, self.lut_a2g)

    def _t_to_tau(self, t):
        """Inverse of :meth:`_tau_to_t`."""
        if getattr(self.config.algo, 'skip_time_reparam', False):
            return t
        return utils.gamma_to_alpha(t, self.lut_g2a)

    def corrupt_continuous(self, x0, t):
        """Corrupt data x0 at time t using linear interpolation with Gaussian noise."""
        t = t.unsqueeze(-1).unsqueeze(-1)
        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        x_t = (1 - t) * noise + t * target_data
        return x_t, target_data

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    def on_load_checkpoint(self, checkpoint):
        print("Resuming training from checkpoint...")
        self._is_resuming = True
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = self._filter_checkpoint_state_dict(
                checkpoint['state_dict'])
        if self.config.mode == 'sample_eval':
            if getattr(self.backbone, 'learnable_loss_weighting', None) is not None:
                if not any(k.startswith('backbone.learnable_loss_weighting')
                           for k in checkpoint['state_dict'].keys()):
                    print("Learnable_loss_weighting not found in checkpoint. "
                          "Initializing from scratch for eval mode.")
                    for name, param in self.backbone.learnable_loss_weighting.named_parameters():
                        param_key = f'backbone.learnable_loss_weighting.{name}'
                        checkpoint['state_dict'][param_key] = param.data.clone()
        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = collections.OrderedDict(
            (k, v) for k, v in checkpoint['state_dict'].items()
            if not k.startswith(('teacher', 'diag_teacher')))
        super().on_save_checkpoint(checkpoint)

    def _filter_checkpoint_state_dict(self, state_dict):
        """Filter teacher keys and strip _orig_mod from checkpoint state_dict."""
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(('teacher', 'diag_teacher')):
                continue
            new_key = k.replace('._orig_mod.', '.')
            new_state_dict[new_key] = v
        return new_state_dict

    def forward_no_softmax(self, xt, tau, tau_prime=None, **kwargs):
        tau = self._process_sigma(tau)
        if tau_prime is not None:
            tau_prime = self._process_sigma(tau_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.backbone(xt, tau, tau_prime, **kwargs)
        return model_output

    def _extract_ema_state_dict(self, model, checkpoint):
        """Extract EMA parameters from checkpoint into a state_dict for model."""
        ema_state = checkpoint.get('ema', None)
        if not ema_state:
            print("Warning: No EMA found, using regular state_dict")
            return {k.replace('backbone.', '').replace('._orig_mod.', ''): v
                    for k, v in checkpoint['state_dict'].items()
                    if k.startswith('backbone.')}

        new_sd = collections.OrderedDict()
        shadow_params = ema_state['shadow_params']
        param_names = [n for n, p in model.named_parameters()
                       if p.requires_grad]
        print(f"EMA shadow_params: {len(shadow_params)}, "
              f"Model param_names: {len(param_names)}")
        min_len = min(len(shadow_params), len(param_names))
        for name, val in zip(param_names[:min_len],
                             shadow_params[:min_len]):
            new_sd[name] = val
        for k, v in checkpoint['state_dict'].items():
            clean_k = k.replace('backbone.', '').replace('._orig_mod.', '')
            if (clean_k not in new_sd
                    and clean_k in [n for n, _ in model.named_parameters()]):
                new_sd[clean_k] = v
                print(f"Loaded missing param from state_dict: {clean_k}")
        if len(shadow_params) != len(param_names):
            print(f"Warning: EMA param count mismatch. "
                  f"Loaded {min_len}/{len(param_names)} from EMA, "
                  f"rest from state_dict")
        return new_sd

    def _load_teacher_model(self, path, use_plain_config=True):
        """Load a frozen teacher model from checkpoint. ``use_plain_config``
        temporarily disables double_temb / learnable_loss_weighting to match
        a base model's EMA parameter shapes.
        """
        print(f"Loading teacher model from: {path}")
        if use_plain_config:
            saved = (self.config.algo.double_temb,
                     self.config.algo.learnable_loss_weighting)
            self.config.algo.double_temb = False
            self.config.algo.learnable_loss_weighting = False

        assert self.config.algo.backbone == 'dit', \
            "Only DIT backbone supported for teacher model"
        model = models.dit.DIT(self.config, vocab_size=self.vocab_size)

        if use_plain_config:
            (self.config.algo.double_temb,
             self.config.algo.learnable_loss_weighting) = saved

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = self._extract_ema_state_dict(model, checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _copy_teacher_weights_to_student(self, teacher_dict):
        """Copy teacher weights to student backbone and zero-init sigma_map_prime."""
        with torch.no_grad():
            student_dict = self.backbone.state_dict()
            for name, param in teacher_dict.items():
                print(f"Copying parameter: {name}")
                if name in student_dict:
                    student_dict[name].copy_(param)
            if (hasattr(self.backbone, 'sigma_map_prime')
                    and self.backbone.sigma_map_prime is not None):
                for name, param in self.backbone.sigma_map_prime.named_parameters():
                    if 'mlp.2' in name:
                        param.zero_()
                        print(f"Zero initialized student sigma_map_prime: {name}")

    @staticmethod
    def _zero_init_module(module):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

    @staticmethod
    def _random_init_module(module, std=0.02):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(mean=0.0, std=std)
                if m.bias is not None:
                    m.bias.data.zero_()


class FMLMPlus(FLMBase):
    """Flow Map Language Model (FMLM+): dual-time training over the
    diagonal/off-diagonal/boundary mix, using the
    ``models.dit_fmlm_plus.DIT_FMLM_PLUS`` backbone.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        assert config.algo.backbone in ('dit', 'dit_2d'), (
            'FMLMPlus currently supports backbone in {"dit", "dit_2d"}.')
        if config.algo.backbone == 'dit_2d':
            self.backbone = models.dit_2d.DIT_FLM_PLUS_2D(
                self.config, vocab_size=self.vocab_size)
        else:
            self.backbone = models.dit_fmlm_plus.DIT_FMLM_PLUS(
                self.config, vocab_size=self.vocab_size)
        if self.ema is not None:
            self.ema = models.ema.ExponentialMovingAverage(
                self._get_parameters(),
                decay=self.config.training.ema)
        self.teacher_model = None
        self.teacher_mask_index = self.vocab_size
        self.diag_teacher = None

        self._validate_configuration()

    def setup(self, stage: str):
        del stage
        cfg_p_distill = float(
            getattr(self.config.algo, 'p_distill', 0.0) or 0.0)
        init_path = getattr(
            self.config.algo, 'init_teacher_ckpt_path', None)
        init_path_set = (
            init_path is not None
            and str(init_path) != ''
            and str(init_path) != 'None')

        if cfg_p_distill > 0.0 and self.teacher_model is None:
            ckpt_path = getattr(
                self.config.algo, 'distill_teacher_ckpt_path', None)
            if (ckpt_path is None
                    or str(ckpt_path) == ''
                    or str(ckpt_path) == 'None'):
                raise ValueError(
                    "algo.p_distill > 0 requires algo.distill_teacher_ckpt_path "
                    "to point to a frozen teacher checkpoint.")
            self.teacher_model = self._load_mdlm_teacher(ckpt_path)

        if init_path_set and not getattr(self, '_teacher_init_done', False):
            self._init_backbone_from_teacher(str(init_path))
            self._teacher_init_done = True

        if self.diag_teacher is None:
            ckpt = getattr(self.config.algo, 'diag_teacher_ckpt_path', None)
            if ckpt is not None and str(ckpt) not in ('', 'None'):
                self.diag_teacher = self._load_flm_plus_teacher(str(ckpt))

    def _load_mdlm_teacher(self, path):
        """Build a DIT with vocab_size = vocab + 1 (matching MDLM's appended
        mask class at index ``self.vocab_size``) and load EMA weights from
        the checkpoint. Returns the frozen, eval-mode model."""
        print(f"[FMLMPlus] Loading MDLM teacher from: {path}")
        teacher_vocab_size = self.vocab_size + 1
        saved = (self.config.algo.double_temb,
                 getattr(self.config.algo,
                         'learnable_loss_weighting', False))
        self.config.algo.double_temb = False
        if hasattr(self.config.algo, 'learnable_loss_weighting'):
            self.config.algo.learnable_loss_weighting = False
        try:
            teacher = models.dit.DIT(
                self.config, vocab_size=teacher_vocab_size)
        finally:
            self.config.algo.double_temb = saved[0]
            if hasattr(self.config.algo, 'learnable_loss_weighting'):
                self.config.algo.learnable_loss_weighting = saved[1]

        checkpoint = torch.load(
            path, map_location='cpu', weights_only=False)
        state_dict = self._extract_ema_state_dict(teacher, checkpoint)
        missing, unexpected = teacher.load_state_dict(
            state_dict, strict=False)
        if missing:
            head = list(missing)[:5]
            tail = '...' if len(missing) > 5 else ''
            print(f"[FMLMPlus teacher] missing keys: {head}{tail}")
        if unexpected:
            head = list(unexpected)[:5]
            tail = '...' if len(unexpected) > 5 else ''
            print(f"[FMLMPlus teacher] unexpected keys: {head}{tail}")
        teacher = teacher.to(self.device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher

    def _init_backbone_from_teacher(self, path):
        """Warm-start ``self.backbone`` from a frozen teacher checkpoint,
        dispatching on ``algo.init_teacher_kind`` (``mdlm`` slices the V+1
        vocab axis down to V; ``fmlm``/``flm_plus`` copy params verbatim).
        """
        kind = str(getattr(
            self.config.algo, 'init_teacher_kind', 'mdlm')).lower()
        if kind not in ('mdlm', 'fmlm', 'flm_plus'):
            raise ValueError(
                f"algo.init_teacher_kind must be 'mdlm', 'fmlm' or "
                f"'flm_plus', got {kind!r}.")
        print(f"[FMLMPlus] Initializing student backbone from "
              f"{kind.upper()} teacher: {path}")

        if kind == 'mdlm':
            teacher_vocab_size = self.vocab_size + 1
            saved = (self.config.algo.double_temb,
                     getattr(self.config.algo,
                             'learnable_loss_weighting', False))
            self.config.algo.double_temb = False
            if hasattr(self.config.algo, 'learnable_loss_weighting'):
                self.config.algo.learnable_loss_weighting = False
            try:
                shape_compat = models.dit.DIT(
                    self.config, vocab_size=teacher_vocab_size)
            finally:
                self.config.algo.double_temb = saved[0]
                if hasattr(self.config.algo, 'learnable_loss_weighting'):
                    self.config.algo.learnable_loss_weighting = saved[1]
        elif kind == 'fmlm':
            teacher_vocab_size = self.vocab_size
            shape_compat = models.dit.DIT(
                self.config, vocab_size=teacher_vocab_size)
        else:
            teacher_vocab_size = self.vocab_size
            shape_compat = models.dit_fmlm_plus.DIT_FMLM_PLUS(
                self.config, vocab_size=teacher_vocab_size)

        checkpoint = torch.load(
            path, map_location='cpu', weights_only=False)
        teacher_sd = self._extract_ema_state_dict(shape_compat, checkpoint)

        student_sd = self.backbone.state_dict()
        sliced = collections.OrderedDict()
        skipped = []
        for name, val in teacher_sd.items():
            if name not in student_sd:
                continue
            target = student_sd[name]
            if val.shape == target.shape:
                sliced[name] = val
                continue
            if kind == 'mdlm' and (
                    name == 'vocab_embed.embedding'
                    and val.shape[0] == teacher_vocab_size
                    and target.shape[0] == self.vocab_size):
                sliced[name] = val[:self.vocab_size, :].contiguous()
            elif kind == 'mdlm' and (
                    name == 'output_layer.linear.weight'
                    and val.shape[0] == teacher_vocab_size
                    and target.shape[0] == self.vocab_size):
                sliced[name] = val[:self.vocab_size, :].contiguous()
            elif kind == 'mdlm' and (
                    name == 'output_layer.linear.bias'
                    and val.shape[0] == teacher_vocab_size
                    and target.shape[0] == self.vocab_size):
                sliced[name] = val[:self.vocab_size].contiguous()
            else:
                skipped.append(
                    (name, tuple(val.shape), tuple(target.shape)))

        for name, t_shape, s_shape in skipped:
            print(f"[FMLMPlus init] Skipping shape-mismatched param "
                  f"{name}: teacher {t_shape} vs student {s_shape}")

        missing, unexpected = self.backbone.load_state_dict(
            sliced, strict=False)
        if missing:
            head = list(missing)[:5]
            tail = '...' if len(missing) > 5 else ''
            print(f"[FMLMPlus init] missing keys "
                  f"(left at student init): {head}{tail}")
        if unexpected:
            head = list(unexpected)[:5]
            tail = '...' if len(unexpected) > 5 else ''
            print(f"[FMLMPlus init] unexpected keys: {head}{tail}")

        if self.ema is not None:
            self.ema = models.ema.ExponentialMovingAverage(
                list(self._get_parameters()),
                decay=self.config.training.ema)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, 'teacher_model', None) is not None:
            self.teacher_model.eval()
        if getattr(self, 'diag_teacher', None) is not None:
            self.diag_teacher.eval()
        return self

    @torch.no_grad()
    def _teacher_log_q(self, x0, distill_active):
        """Run the frozen MDLM teacher with ``mask_index`` at active positions
        and ``x0`` everywhere else. Returns (B, L, V) log-probabilities
        sliced to the student's vocab (mask class carries probability 0)."""
        teacher_xt = torch.where(
            distill_active, self.teacher_mask_index, x0)
        sigma = torch.zeros(
            teacher_xt.shape[0],
            device=self.device, dtype=torch.float32)
        with torch.amp.autocast(
                device_type=self.device.type, dtype=torch.float32):
            logits = self.teacher_model(teacher_xt, sigma)
        logits = logits.clone()
        logits[..., self.teacher_mask_index] = self.neg_infinity
        log_q = logits.log_softmax(dim=-1)
        return log_q[..., :self.vocab_size]

    @staticmethod
    def _attention_to_prompt_mask(attention_mask):
        if attention_mask is None:
            return None
        return attention_mask == 0

    def training_step(self, batch, batch_idx):
        current_accumulation_step = (
            batch_idx % self.trainer.accumulate_grad_batches)
        prompt_mask = self._attention_to_prompt_mask(
            batch.get('attention_mask', None))
        losses = self._loss(
            batch['input_ids'],
            batch['attention_mask'],
            current_accumulation_step,
            train_mode=True,
            xT=None if 'xT' not in batch else batch['xT'],
            given_t=batch['given_t'] if 'given_t' in batch else None,
            not_sampling_t=self.config.training.not_sampling_t,
            prompt_mask=prompt_mask,
        )
        self.metrics.update_train(losses.nlls, losses.prior_loss,
                                  losses.num_tokens)
        self.log(name='trainer/loss', value=losses.loss.item(),
                 on_step=True, on_epoch=False, sync_dist=True)
        return losses.loss

    def validation_step(self, batch, batch_idx):
        del batch, batch_idx
        return None

    def _load_flm_plus_teacher(self, path):
        """Load a frozen FLM+ teacher (DIT_FMLM_PLUS backbone) from
        checkpoint; same architecture as the student, so all params load
        verbatim.
        """
        print(f"[FMLMPlus] Loading FLM+ diagonal teacher from: {path}")
        teacher = models.dit_fmlm_plus.DIT_FMLM_PLUS(
            self.config, vocab_size=self.vocab_size)

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = self._extract_ema_state_dict(teacher, checkpoint)
        missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[FMLMPlus diag_teacher] missing keys: {list(missing)[:5]}"
                  + ('...' if len(missing) > 5 else ''))
        if unexpected:
            print(f"[FMLMPlus diag_teacher] unexpected keys: {list(unexpected)[:5]}"
                  + ('...' if len(unexpected) > 5 else ''))
        teacher = teacher.to(self.device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher

    @torch.no_grad()
    def _diag_teacher_forward(self, x_s, s):
        """Run the frozen FLM+ diagonal teacher on corrupted input x_s at
        time s; returns (n_diag, L, V) log-probabilities.
        """
        sigma = self._process_sigma(s)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.diag_teacher(x_s, sigma, sigma)
        return self._process_model_output(model_output, x_s, sigma)

    def _validate_configuration(self):
        assert self.config.algo.double_temb == True, \
            "FMLMPlus requires double time-emb conditioning to be True"

        assert float(self.config.algo.t_min) == 0.0, \
            f"FMLMPlus requires algo.t_min=0.0, got {self.config.algo.t_min}"
        assert float(self.config.algo.t_max) == 1.0, \
            f"FMLMPlus requires algo.t_max=1.0, got {self.config.algo.t_max}"

        assert type(self.config.algo.diagonal_fraction) == float, \
            "diagonal_fraction must be a float"
        assert 0 <= self.config.algo.diagonal_fraction <= 1, \
            "diagonal_fraction must be between 0 and 1"

        assert type(self.config.algo.add_boundary) == bool, \
            "add_boundary must be a bool"

        assert self.config.algo.normalize_loss_by_active_tokens, "You must normalize loss to account for masking of some losses"

        assert self.config.algo.learnable_loss_weighting == False, "You must disable learnable loss weighting"


    def _process_sigma(self, sigma):
        """Pass per-token sigma (B, L) through unchanged to the backbone."""
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        return sigma

    def forward(self, xt, sigma, sigma_prime=None, use_jvp_attn=False,
                **kwargs):
        """Override base TrainerBase.forward to route per-token sigma directly
        to the backbone (no scalar reduction).
        """
        sigma = self._process_sigma(sigma)
        if sigma_prime is not None:
            sigma_prime = self._process_sigma(sigma_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.backbone(
                xt, sigma, sigma_prime,
                use_jvp_attn=use_jvp_attn, **kwargs)
        return self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)

    def corrupt_continuous(self, x1_ids, t):
        """Linear interpolation between Gaussian noise and one-hot x1 at
        per-token time t.
        """
        t = t.unsqueeze(-1)
        target_data = F.one_hot(x1_ids, self.vocab_size).to(self.dtype)
        noise = torch.randn_like(target_data, dtype=self.dtype)
        x_t = (1 - t) * noise + t * target_data
        return x_t, target_data

    def _loss(self, x0, valid_tokens,
              current_accumulation_step=None, train_mode=False,
              xT=None, given_t=None, not_sampling_t=False,
              prompt_mask=None):
        (input_tokens, output_tokens,
         valid_tokens) = self._process_model_input(x0, valid_tokens)
        loss = self.loss(input_tokens, output_tokens,
                         current_accumulation_step, train_mode,
                         xT=xT, given_t=given_t,
                         not_sampling_t=not_sampling_t,
                         prompt_mask=prompt_mask)
        assert loss.ndim == 2
        if self.ignore_bos:
            loss[:, 1:] = loss[:, 1:]
            valid_tokens[:, 1:] = valid_tokens[:, 1:]

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        active = valid_tokens.bool() & (loss > 0)
        denom = active.sum().clamp(min=1)
        token_nll = nlls / denom
        return trainer_base.Loss(loss=token_nll, nlls=nlls,
                                 prior_loss=0.0, num_tokens=num_tokens)

    def _sample_st_primary(self, B, L, prompt_mask=None, accum_step=None):
        """Sample per-token dual-time ``(s, t)`` for training (``s != t`` in
        general). ``algo.sample_clean`` sets the per-token clean-context
        fraction. Prompt tokens are forced to ``t = t_max``.
        """
        time_sampling = getattr(
            self.config.algo, 'time_sampling', 'uniform_diff')

        span = self.t_max - self.t_min
        s_norm = self._sample_t_interval(
            B, accum_step, t_min=self.t_min, t_max=self.t_max)
        d_norm = self._sample_t_interval(
            B, accum_step, t_min=self.t_min, t_max=self.t_max)
        s_norm = (s_norm - self.t_min) / max(span, 1e-8)
        d_norm = (d_norm - self.t_min) / max(span, 1e-8)
        if time_sampling == "uniform_diff":
            s = self.t_min + span * (s_norm * (1.0 - d_norm))
            t = self.t_min + span * (s_norm * (1.0 - d_norm) + d_norm)
        elif time_sampling == "uniform":
            s = self.t_min + span * s_norm
            t = self.t_min + span * d_norm
            s, t = torch.minimum(s, t), torch.maximum(s, t)
        else:
            raise ValueError(f"Time sampling {time_sampling} is not supported")

        # Per-token clean context fraction (Bernoulli f).
        sample_clean = getattr(self.config.algo, 'sample_clean', 'uniform')
        if prompt_mask is not None:
            selectable = ~prompt_mask
        else:
            selectable = torch.ones(B, L, device=self.device,
                                    dtype=torch.bool)
        if sample_clean == 'none':
            is_context = torch.zeros(B, L, device=self.device,
                                     dtype=torch.bool)
        elif sample_clean == 'uniform':
            f = self._sample_t_interval(
                B, accum_step, t_min=0.0, t_max=1.0)
            bern = torch.rand(B, L, device=self.device)
            is_context = (bern < f.unsqueeze(1)) & selectable
        else:
            raise ValueError(
                f"Unsupported sample_clean '{sample_clean}'. "
                "Expected 'uniform' or 'none'.")

        s = s.unsqueeze(1).expand(B, L).clone()
        t = t.unsqueeze(1).expand(B, L).clone()
        s[is_context] = self.t_max
        t[is_context] = self.t_max

        if prompt_mask is not None:
            s[prompt_mask] = self.t_max
            t[prompt_mask] = self.t_max
        return s, t

    def _sample_st_for_batch(self, B, L, prompt_mask=None, accum_step=None):
        return self._sample_st_primary(
            B, L, prompt_mask=prompt_mask, accum_step=accum_step)
    
    def _get_split_indices(self, n, accum_step, ratios):
        """Split batch indices into category sets"""

        batch_dim = n
        if accum_step is not None:
            n_total = self.config.loader.global_batch_size
            num_nodes = self.trainer.num_nodes
            num_devices = self.trainer.num_devices
            accum_batches = self.trainer.accumulate_grad_batches
            node_rank = self.trainer.node_rank
            local_rank = self.trainer.local_rank
            current_accum = accum_step
        else:
            n_total = n
            num_nodes = num_devices = accum_batches = 1
            node_rank = local_rank = current_accum = 0

        total_chunks = num_nodes * num_devices * accum_batches
        chunk_idx = (node_rank * num_devices * accum_batches
                     + local_rank * accum_batches
                     + current_accum)

        num_categories = len(ratios) + 1
        global_counts = [0] * num_categories
        net_ratio, prev_num = 0, 0
        for i, ratio in enumerate(ratios):
            net_ratio += ratio
            num_a = int(n_total * net_ratio)
            global_counts[i] = num_a - prev_num
            prev_num = num_a
        global_counts[-1] = n_total - prev_num

        local_counts = []
        for cnt in global_counts:
            base, rem = divmod(cnt, total_chunks)
            local_counts.append(base + (1 if chunk_idx < rem else 0))

        local_size = sum(local_counts)
        local_assignments = torch.empty(local_size, device=self.device, dtype=torch.long)
        offset = 0
        for i, cnt in enumerate(local_counts):
            local_assignments[offset:offset + cnt] = i
            offset += cnt

        seed = self.global_step * total_chunks + chunk_idx
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        perm = torch.randperm(local_size, device=self.device, generator=g)
        local_assignments = local_assignments[perm][:batch_dim]

        return [(local_assignments == i).nonzero(as_tuple=True)[0]
                for i in range(num_categories)]

    # ── loss ──────────────────────────────────────────────────

    def _clip_renorm_probs(self, p, eps=1e-8):
        """Repair IMF compound prediction: clamp negatives, then renormalize
        so each token's distribution sums to 1.
        """
        p = p.clamp_min(eps)
        return p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

    def loss(self, x1_ids, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False,
             prompt_mask=None):
        del given_t, not_sampling_t, output_tokens, xT, train_mode
        B, L = x1_ids.shape
        s, t = self._sample_st_for_batch(
            B, L, prompt_mask=prompt_mask,
            accum_step=current_accumulation_step)
        
        time_sampling = getattr(
            self.config.algo, 'time_sampling', 'uniform_diff')

        p_distill = float(
            getattr(self.config.algo, 'p_distill', 0.0) or 0.0)
        do_distill = (
            p_distill > 0.0
            and getattr(self, 'teacher_model', None) is not None)
        if not do_distill:
            p_distill = 0.0

        boundary_ratio = 0.0
        if self.config.algo.add_boundary:
            boundary_ratio = ((1 - p_distill)
                              * (1.0 - self.config.algo.diagonal_fraction)
                              * (1.0 / self.config.algo.boundary_prob))

        idx_distill, idx_diag, idx_offdiag_bndry, idx_offdiag = self._get_split_indices(
            B, current_accumulation_step,
            ratios=(p_distill, (1 - p_distill) * self.config.algo.diagonal_fraction,
                    boundary_ratio))

        active_mask = (s < self.t_max - 1e-7)
        distill_active = torch.zeros_like(active_mask)
        if idx_distill.numel() > 0:
            distill_active[idx_distill] = active_mask[idx_distill]
        s[distill_active] = 0.0
        t[distill_active] = 0.0
        if idx_diag.numel() > 0:
            if time_sampling == "uniform_diff":
                diag_delta = t[idx_diag] - s[idx_diag]
                s[idx_diag] = torch.where(active_mask[idx_diag], diag_delta, s[idx_diag])
                t[idx_diag] = torch.where(active_mask[idx_diag], diag_delta, t[idx_diag])
            elif time_sampling == "uniform":
                u = s[idx_diag]/(t[idx_diag] + 1e-8)
                s[idx_diag] = torch.where(active_mask[idx_diag], u, s[idx_diag])
                t[idx_diag] = torch.where(active_mask[idx_diag], u, t[idx_diag])
        boundary_active = torch.zeros_like(active_mask)
        if idx_offdiag_bndry.numel() > 0:
            boundary_active[idx_offdiag_bndry] = active_mask[idx_offdiag_bndry]
        s[boundary_active] = 0.0
        t[boundary_active] = 1.0

        s = torch.clamp(s, 0.0, 1.0)
        t = torch.clamp(t, 0.0, 1.0)

        idx_offdiag = torch.cat([idx_offdiag, idx_offdiag_bndry])
        has_distill = idx_distill.numel() > 0
        has_diag = idx_diag.numel() > 0
        has_offdiag = idx_offdiag.numel() > 0
        
        c_s = self._tau_to_t(s)
        c_t = self._tau_to_t(t)

        if self.config.algo.distillation_method == "PSD":
            x_s, target_data = self.corrupt_continuous(x1_ids, c_s)
            log_D_st = self.forward(x_s, s, t, use_jvp_attn=False)
            if has_distill:
                log_D_0_teacher = self._teacher_log_q(
                    x1_ids[idx_distill], distill_active[idx_distill])
                D_0_teacher = log_D_0_teacher.exp()
                distill_loss = -(D_0_teacher * log_D_st[idx_distill]).sum(dim=-1)
                distill_loss = distill_loss * distill_active[idx_distill]
            else:
                distill_loss = x_s.new_empty((0, L))

            if has_diag:
                if self.diag_teacher is not None:
                    diag_target = stopgrad(
                        self._diag_teacher_forward(
                            x_s[idx_diag], s[idx_diag]).exp())
                else:
                    diag_target = target_data[idx_diag]
                diag_loss = -(diag_target * log_D_st[idx_diag]).sum(dim=-1)
                diag_loss = diag_loss * active_mask[idx_diag]
            else:
                diag_loss = x_s.new_empty((0, L))

            if has_offdiag:
                with torch.no_grad():
                    x_s_od = x_s[idx_offdiag]
                    s_od = s[idx_offdiag]
                    t_od = t[idx_offdiag]
                    u_od = (s_od + t_od)/2.0
                    c_s_od = c_s[idx_offdiag]
                    c_t_od = c_t[idx_offdiag]
                    c_u_od = self._tau_to_t(u_od)
                    c_s_od_exp = c_s_od.unsqueeze(-1)
                    c_u_od_exp = c_u_od.unsqueeze(-1)
                    c_t_od_exp = c_t_od.unsqueeze(-1)
                    D_su_offdiag = self.forward(x_s_od, s_od, u_od).exp()
                    X_su = ((1 - c_u_od_exp + 1e-8) / (1 - c_s_od_exp + 1e-8)) * x_s_od \
                           + ((c_u_od_exp - c_s_od_exp) / (1 - c_s_od_exp + 1e-8)) * D_su_offdiag
                    D_ut_offdiag = self.forward(X_su, u_od, t_od).exp()
                    lambda_sut = ((1 - c_t_od_exp) * (c_u_od_exp - c_s_od_exp)
                                  / ((1 - c_u_od_exp) * (c_t_od_exp - c_s_od_exp) + 1e-8))
                    offdiag_target = stopgrad(
                        lambda_sut * D_su_offdiag + (1 - lambda_sut) * D_ut_offdiag)

                offdiag_loss = -(offdiag_target * log_D_st[idx_offdiag]).sum(dim=-1)
                offdiag_loss = offdiag_loss * active_mask[idx_offdiag]
            else:
                offdiag_loss = x_s.new_empty((0, L))

        else:
            raise NotImplementedError

        loss = torch.zeros(B, L, device=self.device)
        
        if has_distill:
            loss[idx_distill] = distill_loss
            distill_loss_to_log = distill_loss.mean()
        else:
            distill_loss_to_log = loss.new_tensor(0.0)

        if has_diag:
            loss[idx_diag] = diag_loss
            diag_loss_to_log = diag_loss.mean()
        else:
            diag_loss_to_log = loss.new_tensor(0.0)
        
        if has_offdiag:
            loss[idx_offdiag] = offdiag_loss
            offdiag_loss_to_log = offdiag_loss.mean()
        else:
            offdiag_loss_to_log = loss.new_tensor(0.0)
        
        self.log('distill_loss', distill_loss_to_log, prog_bar=True, sync_dist=True)
        self.log('distill_count', idx_distill.numel(), prog_bar=True, sync_dist=True)
        self.log('diag_loss', diag_loss_to_log, prog_bar=True, sync_dist=True)
        self.log('diag_count', idx_diag.numel(), prog_bar=True, sync_dist=True)
        self.log('offdiag_loss', offdiag_loss_to_log, prog_bar=True, sync_dist=True)
        self.log('offdiag_count', idx_offdiag.numel(), prog_bar=True, sync_dist=True)
        self.log('loss', loss.mean(), prog_bar=True, sync_dist=True)

        return loss

    # ── sampling helpers ──────────────────────────────────────

    @torch.no_grad()
    def sample_block_diffusion(self, B, x0=None, prompt_mask=None,
                               eps=1e-5):
        V = self.vocab_size
        L = self.num_tokens
        device = self.device

        block_length = int(self.config.sampling.block_length)
        if block_length < 1:
            raise ValueError(
                f'block_length must be >= 1, got {block_length}')
        steps_per_block = int(self.config.sampling.steps_per_block)
        if steps_per_block < 1:
            raise ValueError(
                f'steps_per_block must be >= 1, got {steps_per_block}')
        num_blocks = math.ceil(L / block_length)

        z = torch.randn((B, L, V), device=device, dtype=self.dtype)

        frozen = None
        if prompt_mask is not None and x0 is not None:
            x0_onehot = F.one_hot(x0, V).float().to(self.dtype)
            z[prompt_mask] = x0_onehot[prompt_mask]
            frozen = prompt_mask.unsqueeze(-1)

        z_init = z.clone()

        for b in range(num_blocks):
            block_start = b * block_length
            block_end = min((b + 1) * block_length, L)

            for local_step in range(steps_per_block):
                t_val_in = local_step / steps_per_block
                t_val_out = (local_step + 1) / steps_per_block

                t_in = torch.ones(B, L, device=device, dtype=self.dtype)
                t_out = torch.ones(B, L, device=device, dtype=self.dtype)
                t_in[:, block_start:] = t_val_in
                t_out[:, block_start:] = t_val_out

                if prompt_mask is not None:
                    t_in[prompt_mask] = 1.0
                    t_out[prompt_mask] = 1.0

                active = t_out > t_in
                if not active.any():
                    continue

                c_in = self._tau_to_t(t_in)
                c_out = self._tau_to_t(t_out)

                log_D_st_pred = self.forward(z, t_in, t_out)
                D_st_pred = log_D_st_pred.exp()
                if t_val_out >= 1.0 - 1e-7:
                    z_new = D_st_pred
                else:
                    weight_z = (1.0 - c_out.unsqueeze(-1)) / (1.0 - c_in.unsqueeze(-1) + 1e-8)
                    weight_D = ((c_out.unsqueeze(-1) - c_in.unsqueeze(-1))
                                / (1.0 - c_in.unsqueeze(-1) + 1e-8))
                    z_new = weight_z * z + weight_D * D_st_pred


                final_now = active & (t_out >= 1.0 - 1e-7)
                ode_now = active & ~final_now

                token_ids = z_new.argmax(dim=-1)
                x1_final = torch.zeros_like(z_new).scatter_(
                    -1, token_ids.unsqueeze(-1), 1.0)

                z_updated = torch.where(
                    final_now.unsqueeze(-1), x1_final,
                    torch.where(ode_now.unsqueeze(-1), z_new, z))
                z = z_updated if frozen is None else torch.where(frozen, z, z_updated)

            if block_end < L:
                restore = z_init[:, block_end:, :]
                if prompt_mask is not None:
                    keep = prompt_mask[:, block_end:].unsqueeze(-1)
                    z[:, block_end:, :] = torch.where(
                        keep, z[:, block_end:, :], restore)
                else:
                    z[:, block_end:, :] = restore

        return z.argmax(dim=-1)

    # ── public sampler ────────────────────────────────────────

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5,
                         x0=None, prompt_mask=None,
                         prompt_tokens=None, prompt_lens=None, **kwargs):
        """Generate samples, conditioned on either ``x0 + prompt_mask`` or
        ``prompt_tokens + prompt_lens``, or unconditional. Dispatches on
        ``sampling.method`` (block-diffusion | refinement).
        """
        del num_steps, kwargs

        if prompt_tokens is not None:
            if isinstance(prompt_tokens, np.ndarray):
                prompt_tokens = torch.from_numpy(prompt_tokens)
            x0 = prompt_tokens.to(self.device, dtype=torch.long)
            B = x0.shape[0]
            L = x0.shape[1]
            if prompt_lens is None:
                raise ValueError(
                    'prompt_lens is required when prompt_tokens is provided.')
            if isinstance(prompt_lens, np.ndarray):
                prompt_lens = torch.from_numpy(prompt_lens)
            prompt_lens = prompt_lens.to(self.device, dtype=torch.long)
            positions = torch.arange(L, device=self.device).unsqueeze(0)
            prompt_mask = positions < prompt_lens.unsqueeze(1)
        elif x0 is not None and prompt_mask is not None:
            B = x0.shape[0]
            x0 = x0.to(self.device)
            prompt_mask = prompt_mask.to(self.device)
        else:
            B = num_samples
            prompt_mask = None

        method = str(self.config.sampling.method)
        assert method in ('block-diffusion', 'refinement'), (
            f"sampling.method must be 'block-diffusion' or 'refinement', "
            f"got {method!r}.")
        if method == 'block-diffusion':
            result = self.sample_block_diffusion(B, x0, prompt_mask, eps)
        else:
            result = self.sample_refinement(B, x0, prompt_mask, eps)

        if x0 is not None and prompt_mask is not None:
            result = torch.where(prompt_mask, x0, result)
        return result

    @torch.no_grad()
    def sample_refinement(self, B, x0=None, prompt_mask=None, eps=1e-5):
        """Posterior Refinement sampler: each round predicts the clean
        sequence, commits cells with confidence >= p (plus any top_k /
        uniform_budget forced commits), and renoises the rest. Returns the
        final (B, L) prediction with clue positions pinned.
        """
        p = float(self.config.sampling.get('refinement_threshold', 0.9))
        if p < 0:
            p = float('inf')
        fresh_noise = bool(
            self.config.sampling.get('refinement_fresh_noise', False))
        top_k_cfg = self.config.sampling.get('refinement_top_k', 0)
        top_k = ('uniform_budget' if str(top_k_cfg) == 'uniform_budget'
                 else int(top_k_cfg))
        rounds = int(self.config.sampling.refinement_rounds)
        refinement_flow_steps = int(
            self.config.sampling.get('refinement_flow_steps', 1))

        device = self.device
        dtype = self.dtype
        V = self.vocab_size
        L = self.num_tokens

        if prompt_mask is None:
            y = torch.zeros((B, L), device=device, dtype=torch.long)
            clue_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
        else:
            y = (x0.long() if x0 is not None
                 else torch.zeros_like(prompt_mask, dtype=torch.long))
            clue_mask = prompt_mask.to(device)

        y_one_hot = F.one_hot(y, num_classes=V).to(dtype)
        noise = torch.randn(B, L, V, device=device, dtype=dtype)
        x_state = torch.where(clue_mask.unsqueeze(-1), y_one_hot, noise)

        print_pmax = bool(
            self.config.sampling.get('refinement_print_pmax', False))

        # Clue cells start clean (s=1), the rest noisy (s=0); t=1 everywhere
        # so the network emits the one-step posterior delta_{s,1}.
        s = torch.where(
            clue_mask,
            torch.ones((B, L), device=device, dtype=dtype),
            torch.zeros((B, L), device=device, dtype=dtype))
        t = torch.ones((B, L), device=device, dtype=dtype)

        commit_stats = []
        full_pred = y.clone()
        for round_idx in range(rounds):
            was_unfrozen_solution = (s < 0.5) & (~clue_mask)

            N = int(refinement_flow_steps)
            if N > 1:
                ones_bl = torch.ones((B, L), device=device, dtype=dtype)
                clean_pos = ~was_unfrozen_solution
                z_round = x_state
                for k in range(N):
                    t_in = torch.where(
                        clean_pos, ones_bl,
                        torch.full((B, L), float(k) / N,
                                   device=device, dtype=dtype))
                    t_out = torch.where(
                        clean_pos, ones_bl,
                        torch.full((B, L), float(k + 1) / N,
                                   device=device, dtype=dtype))
                    c_in = self._tau_to_t(t_in)
                    c_out = self._tau_to_t(t_out)
                    log_D_st = self.forward(z_round, t_in, t_out)
                    D_st = log_D_st.float().exp()
                    if float(k + 1) / N >= 1.0 - 1e-7:
                        break
                    denom = (1.0 - c_in).unsqueeze(-1) + 1e-8
                    weight_z = (1.0 - c_out).unsqueeze(-1) / denom
                    weight_D = (c_out - c_in).unsqueeze(-1) / denom
                    z_new = weight_z * z_round + weight_D * D_st.to(dtype)
                    z_round = torch.where(
                        clean_pos.unsqueeze(-1), z_round, z_new)
            else:
                log_D_st = self.forward(x_state, s, t)
                D_st = log_D_st.float().exp()
            pred = D_st.argmax(dim=-1)
            selected_probs = D_st.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            locked = ~was_unfrozen_solution
            pred = torch.where(locked, x_state.argmax(dim=-1), pred)
            full_pred = torch.where(clue_mask, y, pred)

            already_frozen = (~was_unfrozen_solution) & (~clue_mask)
            threshold_pass = (selected_probs >= p) & was_unfrozen_solution
            new_commits = threshold_pass
            if top_k == 'uniform_budget':
                tokens_left = was_unfrozen_solution.sum(dim=1).float()
                steps_left = rounds - round_idx
                k_row = torch.ceil(
                    tokens_left / steps_left).long()
                masked_probs = selected_probs.float().masked_fill(
                    ~was_unfrozen_solution, float('-inf'))
                order = masked_probs.argsort(dim=1, descending=True)
                ranks = torch.empty_like(order)
                ar = torch.arange(
                    L, device=device).unsqueeze(0).expand(B, L)
                ranks.scatter_(1, order, ar)
                topk_mask = (
                    ranks < k_row.unsqueeze(1)) & was_unfrozen_solution
                new_commits = new_commits | topk_mask
            elif top_k > 0:
                masked_probs = selected_probs.float().masked_fill(
                    ~was_unfrozen_solution, float('-inf'))
                k = min(int(top_k), int(L))
                _, topk_idx = masked_probs.topk(k, dim=1)
                topk_mask = torch.zeros_like(new_commits)
                topk_mask.scatter_(1, topk_idx, True)
                topk_mask = topk_mask & was_unfrozen_solution
                new_commits = new_commits | topk_mask
            # Previously-frozen cells stay frozen — never re-checked.
            accept_mask = new_commits | already_frozen
            newly_committed = new_commits
            cumulative_frozen = accept_mask

            n_refinable = float((~clue_mask).sum().item()) / max(B, 1)
            cum_per_sample = cumulative_frozen.sum(dim=1).float()
            new_per_sample = newly_committed.sum(dim=1).float()
            commit_stats.append({
                'round': int(round_idx),
                'newly_committed': float(new_per_sample.mean().item()),
                'newly_committed_std': float(
                    new_per_sample.std(unbiased=False).item()),
                'cumulative_frozen': float(cum_per_sample.mean().item()),
                'cumulative_frozen_std': float(
                    cum_per_sample.std(unbiased=False).item()),
                'refinable': n_refinable,
                'seq_len': int(L),
            })

            if print_pmax:
                pmax0 = D_st[0].max(dim=-1).values
                frozen0 = cumulative_frozen[0]
                clue0 = clue_mask[0]
                cells = []
                for j in range(L):
                    if clue0[j]:
                        mark = 'c'
                    elif frozen0[j]:
                        mark = '*'
                    else:
                        mark = '.'
                    cells.append(f"{j:2d}{mark}{pmax0[j].item():.3f}")
                n_committed0 = int(cumulative_frozen[0].sum().item())
                print(
                    f"[pmax round={round_idx:3d}/{rounds} "
                    f"committed={n_committed0}/{L}]")
                print("  " + "  ".join(cells))

            if round_idx == rounds - 1:
                break

            renoise_mask = (~accept_mask) & (~clue_mask)
            pred_one_hot = F.one_hot(pred, num_classes=V).to(dtype)
            x_state = torch.where(
                accept_mask.unsqueeze(-1), pred_one_hot, x_state)
            if fresh_noise:
                renoise_noise = torch.randn_like(x_state)
                x_state = torch.where(
                    renoise_mask.unsqueeze(-1), renoise_noise, x_state)
            # else: keep the initial noise at uncommitted positions.

            s = torch.where(accept_mask, torch.ones_like(s), s)
            s = torch.where(renoise_mask, torch.zeros_like(s), s)
        self._last_refine_commit_stats = commit_stats
        return full_pred


class MDLM(trainer_base.TrainerBase):
    """Masked Diffusion Language Model (absorbing-state discrete diffusion),
    used as the TinyGSM MDLM baseline / distillation teacher. Implements
    absorbing-state corruption (``q_xt``), SUBS parameterization, ELBO
    weighting, and a prompt-conditioned ancestral sampler.
    """

    def __init__(self, config, tokenizer):
        vocab_size = len(tokenizer)
        if (not hasattr(tokenizer, 'mask_token')
                or tokenizer.mask_token is None):
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.diffusion_type = 'absorbing'
        self.post_process_mode = getattr(
            config.algo, 'post_process_mode', 'log_probs')
        assert self.parameterization == 'subs', (
            'MDLM expects algo.parameterization=subs, got '
            f'{self.parameterization!r}')
        assert self.post_process_mode in ('log_probs', 'logits'), \
            self.post_process_mode

    # -- corruption / parameterization ------------------------------------
    def prior_sample(self, *batch_dims):
        return self.mask_index * torch.ones(
            *batch_dims, dtype=torch.int64, device=self.device)

    def _sigma_from_alphat(self, alpha_t):
        return -torch.log(alpha_t)

    def _process_sigma(self, sigma):
        if sigma is None:
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.reshape(sigma.shape[0], -1).mean(-1)
        sigma = sigma.squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        return sigma

    def q_xt(self, x, alpha_t, valid_tokens=None):
        """Absorbing forward process: each token is replaced by ``mask_index``
        with probability ``1 - alpha_t``. Positions with ``valid_tokens == 0``
        (e.g. the prompt) are kept clean so the model conditions on them."""
        move_indices = torch.rand(*x.shape, device=x.device) < (1 - alpha_t)
        xt = torch.where(move_indices, self.mask_index, x)
        if valid_tokens is not None:
            xt = torch.where(valid_tokens.bool(), xt, x)
        return xt

    def _subs_parameterization(self, model_output, xt):
        # SUBS: forbid predicting the mask token, renormalize to log-probs, and
        # carry already-revealed tokens over deterministically.
        model_output[:, :, self.mask_index] += self.neg_infinity
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True)
        unmasked = (xt != self.mask_index)
        model_output[unmasked] = self.neg_infinity
        model_output[unmasked, xt[unmasked]] = 0
        return model_output

    def _process_model_output(self, model_output, xt, sigma):
        del sigma
        if self.post_process_mode == 'logits':
            model_output[..., self.mask_index] = self.neg_infinity
            unmasked = (xt != self.mask_index)[..., None]
            deterministic = model_output.new_full(
                model_output.shape, self.neg_infinity)
            deterministic.scatter_(-1, xt.unsqueeze(-1), 0.0)
            return torch.where(unmasked, deterministic, model_output)
        return self._subs_parameterization(model_output, xt)

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    # -- training / eval loss ---------------------------------------------
    def _sample_t(self, n, accum_step=None):
        del accum_step
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
        return (1 - self.sampling_eps) * _eps_t + self.sampling_eps

    def nll_per_token(self, log_x_theta, x0, alpha_t, dalpha_t):
        log_p_theta = torch.gather(
            log_x_theta, -1, x0[:, :, None]).squeeze(-1)
        if getattr(self, 'loss_type', 'elbo') == 'low_var':
            coeff = -1.0
        else:
            coeff = dalpha_t / (1 - alpha_t)
        return coeff * log_p_theta

    def nll(self, x0, output_tokens, valid_tokens=None,
            current_accumulation_step=None, train_mode=False):
        del output_tokens, train_mode
        if self.post_process_mode == 'logits':
            raise ValueError(
                'post_process_mode=logits is sampling-only; use log_probs '
                'for training/eval.')
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        sigma = self._sigma_from_alphat(alpha_t)
        xt = self.q_xt(x0, alpha_t, valid_tokens=valid_tokens)
        log_x_theta = self.forward(xt, sigma)
        return self.nll_per_token(log_x_theta, x0, alpha_t, dalpha_t)

    def _loss(self, x0, valid_tokens, current_accumulation_step=None,
              train_mode=False, **kwargs):
        del kwargs
        (input_tokens, output_tokens,
         valid_tokens) = self._process_model_input(x0, valid_tokens)
        loss = self.nll(input_tokens, output_tokens, valid_tokens,
                        current_accumulation_step, train_mode)
        assert loss.ndim == 2
        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens.clamp(min=1)
        return trainer_base.Loss(loss=token_nll, nlls=nlls,
                                 prior_loss=0.0, num_tokens=num_tokens)

    # -- sampling ---------------------------------------------------------
    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5,
                         x0=None, prompt_mask=None,
                         prompt_tokens=None, prompt_lens=None, **kwargs):
        """Prompt-conditioned ancestral sampling. Accepts either
            * prompt_tokens + prompt_lens (TinyGSM / GSM8K eval style), or
            * x0 + prompt_mask, or
            * neither (unconditional),
        and returns a ``(B, L)`` LongTensor of token ids. ``num_steps`` is the
        number of ancestral denoising steps."""
        kwargs.pop('keep_k', None)
        kwargs.pop('rounds', None)
        temperature = kwargs.pop('temperature', None)
        if temperature is None:
            temperature = self.config.sampling.get('temperature', 1.0)
        temperature = float(temperature)
        if kwargs:
            print('[MDLM.generate_samples] ignoring kwargs: '
                  f'{list(kwargs.keys())}')

        if num_steps is None:
            steps_cfg = self.config.sampling.steps
            try:
                num_steps = int(steps_cfg)
            except (TypeError, ValueError):
                num_steps = int(list(steps_cfg)[0])
        num_steps = int(num_steps)

        if prompt_tokens is not None:
            if isinstance(prompt_tokens, np.ndarray):
                prompt_tokens = torch.from_numpy(prompt_tokens)
            x0 = prompt_tokens.to(self.device, dtype=torch.long)
            B, L = x0.shape
            if prompt_lens is None:
                raise ValueError(
                    'prompt_lens is required when prompt_tokens is provided.')
            if isinstance(prompt_lens, np.ndarray):
                prompt_lens = torch.from_numpy(prompt_lens)
            prompt_lens = prompt_lens.to(self.device, dtype=torch.long)
            positions = torch.arange(L, device=self.device).unsqueeze(0)
            prompt_mask = positions < prompt_lens.unsqueeze(1)
        elif x0 is not None and prompt_mask is not None:
            x0 = x0.to(self.device, dtype=torch.long)
            prompt_mask = prompt_mask.to(self.device).bool()
            B, L = x0.shape
        else:
            B, L = int(num_samples), self.num_tokens
            x0, prompt_mask = None, None

        sampler = getattr(self, 'sampler', 'ancestral') or 'ancestral'
        if sampler == 'confidence':
            return self._confidence_sample(
                B, L, x0, prompt_mask, num_steps, eps, temperature)
        return self._ancestral_sample(
            B, L, x0, prompt_mask, num_steps, eps, temperature)

    @torch.no_grad()
    def _ancestral_sample(self, B, L, x0, prompt_mask, num_steps, eps,
                          temperature):
        device = self.device
        xt = self.prior_sample(B, L)
        if prompt_mask is not None:
            xt = torch.where(prompt_mask, x0, xt)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        ones = torch.ones(B, 1, device=device)
        for i in range(num_steps):
            is_last = (i == num_steps - 1)
            _, alpha_t = self.noise(timesteps[i] * ones)
            if alpha_t.ndim == 1:
                alpha_t = alpha_t.unsqueeze(-1)
            if is_last:
                alpha_s = torch.ones_like(alpha_t)
            else:
                _, alpha_s = self.noise(timesteps[i + 1] * ones)
                if alpha_s.ndim == 1:
                    alpha_s = alpha_s.unsqueeze(-1)
            sigma_t = self._sigma_from_alphat(alpha_t)
            log_probs = self.forward(xt, sigma_t)
            x0_sample = trainer_base.sample_categorical(
                log_probs.exp(), temperature=temperature)
            if is_last:
                should_denoise = torch.ones_like(xt, dtype=torch.bool)
            else:
                denoise_prob = (alpha_s - alpha_t) / (1 - alpha_t)
                should_denoise = torch.rand(B, L, device=device) < denoise_prob
            is_masked = (xt == self.mask_index)
            xt = torch.where(is_masked & should_denoise, x0_sample, xt)
            if prompt_mask is not None:
                xt = torch.where(prompt_mask, x0, xt)
        return xt

    @torch.no_grad()
    def _confidence_sample(self, B, L, x0, prompt_mask, num_steps, eps,
                           temperature):
        """MaskGIT-style confidence-based parallel decoding for MDLM: from the
        all-mask prior, each step commits the most confident still-masked
        positions (``ceil(remaining_masks / remaining_steps)`` per step) with
        no remasking. ``temperature=0`` recovers argmax.
        """
        x = self.prior_sample(B, L)
        if prompt_mask is not None:
            x = torch.where(prompt_mask, x0, x)
        for step in range(num_steps):
            is_mask = (x == self.mask_index)
            num_masked = is_mask.sum(dim=-1)
            if int(num_masked.max()) == 0:
                break
            remaining_steps = num_steps - step
            k = torch.ceil(
                num_masked.float() / remaining_steps).long().clamp(min=1)

            frac_masked = is_mask.float().mean(dim=-1, keepdim=True)
            t = frac_masked.clamp(min=eps, max=1.0 - eps)
            _, alpha_t = self.noise(t)
            sigma = self._sigma_from_alphat(alpha_t)

            log_p = self.forward(x, sigma)
            conf, argmax = log_p.max(dim=-1)
            if temperature == 0.0:
                pred = argmax
            else:
                pred = trainer_base.sample_categorical(
                    log_p.exp(), temperature)
            conf = conf.masked_fill(~is_mask, float('-inf'))

            ranks = torch.argsort(
                torch.argsort(conf, dim=-1, descending=True), dim=-1)
            commit = (ranks < k.unsqueeze(1)) & is_mask
            x = torch.where(commit, pred, x)

        if prompt_mask is not None:
            x = torch.where(prompt_mask, x0, x)
        return x

    # -- checkpoint loading ----------------------------------------------
    def load_state_dict(self, state_dict, strict=True):
        if any('._orig_mod.' in k for k in state_dict):
            state_dict = {k.replace('._orig_mod.', '.'): v
                          for k, v in state_dict.items()}
        own = set(self.state_dict().keys())
        dropped = [k for k in state_dict if k not in own]
        if dropped:
            print(f'[MDLM] dropping {len(dropped)} checkpoint key(s) absent '
                  f'from the model (e.g. {dropped[:3]})')
            state_dict = {k: v for k, v in state_dict.items() if k in own}
        return super().load_state_dict(state_dict, strict=False)

