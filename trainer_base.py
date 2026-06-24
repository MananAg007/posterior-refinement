import itertools
import json
import math
import os
import random
import sys
import inspect

from dataclasses import dataclass


from tqdm import tqdm
import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb
from torch.cuda.amp import autocast
import torch.distributed as dist
import dataloader
import metrics
import models
import utils


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    prior_loss: torch.FloatTensor
    num_tokens: torch.FloatTensor


class LogLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-3  # To be consistent with SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/noise_lib.py#L56

    def forward(self, t):
        t = (1 - self.eps) * t
        alpha_t = 1 - t
        dalpha_t = - (1 - self.eps) + t * 0
        assert alpha_t.shape == dalpha_t.shape
        return dalpha_t, alpha_t


def sample_categorical(categorical_probs, temperature=1.0):
    if temperature != 1.0:
        categorical_probs = categorical_probs.pow(1.0 / temperature)
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(
        * x.shape,
        * ((1,) * (len(reference.shape) - len(x.shape))))


def _maybe_scalar_int(v):
    # Sweep configs pass block_length / steps_per_block as lists; the
    # block-diffusion eval flattens them, but other schedules don't, so
    # coerce defensively for log records.
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            seq = list(v)
        except TypeError:
            return 0
        if len(seq) == 1:
            return int(seq[0])
        return 0


class TrainerBase(L.LightningModule):
    def __init__(
            self,
            config,
            tokenizer: transformers.PreTrainedTokenizer,
            vocab_size=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if hasattr(self.config.algo, 'ignore_bos'):
            self.ignore_bos = config.algo.ignore_bos
        else:
            self.ignore_bos = False
        if hasattr(self.config.algo, 'loss_type'):
            self.loss_type = config.algo.loss_type
        self.tokenizer = tokenizer
        if vocab_size is None:
            self.vocab_size = len(self.tokenizer)
        else:
            self.vocab_size = vocab_size
        self.sampler = self.config.sampling.predictor
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.parameterization = self.config.algo.parameterization
        if self.config.algo.backbone == 'dit':
            self.backbone = models.dit.DIT(
                self.config, vocab_size=self.vocab_size)
        elif self.config.algo.backbone == 'dit_2d':
            self.backbone = models.dit_2d.DIT2D(
                self.config, vocab_size=self.vocab_size)
        elif self.config.algo.backbone == 'dimamba':
            self.backbone = models.dimamba.DiMamba(
                self.config,
                vocab_size=self.vocab_size,
                pad_token_id=self.tokenizer.pad_token_id)
        elif self.config.algo.backbone == 'hf_dit':
            self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
                config.eval.checkpoint_path, trust_remote_code=True)
        elif self.config.algo.backbone == 'sphere-dit':
            from models import sphere_dit
            self.backbone = sphere_dit.SphereDiT(
                self.config, vocab_size=self.vocab_size)
        elif self.config.algo.backbone == 'candi-dit':
            # CANDI's Continuous Diffusion Transformer. ``vocab_size`` here
            # already includes the appended mask token (set by CANDI.__init__).
            self.backbone = models.candi_dit.ContDIT(
                self.config, vocab_size=self.vocab_size)

        self._pending_ema_state = None
        self.T = self.config.algo.T
        self.num_tokens = self.config.model.length
        self.softplus = torch.nn.Softplus()
        self.p_nucleus = self.config.sampling.p_nucleus
        # Noise Schedule
        self.noise = LogLinear()

        self.is_sflm_sudoku = getattr(self.config.data,
                                      'tokenizer_name_or_path', '') == 'sudoku'
        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=self.config.eval.gen_ppl_eval_model_name_or_path,
            eval_ppl_batch_size=self.config.eval.perplexity_batch_size,
            is_sudoku=False)
        # Lazily populated on first sflm_sudoku validation.
        self._sflm_sudoku_eval_cache = None
        # Lazily populated on first validation that needs the real GSM8K
        # test prompts (see gsm8k_test_data.load_gsm8k_test_eval).
        self._gsm8k_test_eval_cache = None

        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
            self._get_parameters(),
            decay=self.config.training.ema)
        else:
            self.ema = None


        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.algo.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self.target_tokens = None


    def _validate_configuration(self):
        assert self.config.algo.backbone in {'dit', 'hf_dit'}
        if self.config.algo.parameterization == 'ar':
            assert not self.config.algo.time_conditioning
            assert self.config.prior.type == 'none'

        if self.parameterization in {'score', 'mean'}:
            assert self.time_conditioning
        if self.T > 0:
            assert self.parameterization != 'score'

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.metrics.to(*args, **kwargs)
        return self

    def q_xt(self, x, alpha_t):
        raise NotImplementedError

    def _get_parameters(self):
        return itertools.chain(self.backbone.parameters(),
                               self.noise.parameters())

    def _eval_mode(self):
        if self.ema and not self.config.eval.disable_ema:
            print('Copying EMA parameters to model')
            self.ema.store(self._get_parameters())
            self.ema.copy_to(self._get_parameters())
        else:
            print('No EMA parameters')
        self.backbone.eval()
        self.noise.eval()

    def _train_mode(self):
        if self.ema:
            self.ema.restore(self._get_parameters())
        self.backbone.train()
        self.noise.train()

    def load_state_dict(self, state_dict, strict=True):
        if any('_orig_mod' in k for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('._orig_mod.', '.')
                new_state_dict[new_key] = v
            state_dict = new_state_dict
        
        if hasattr(self, 'teacher_model') and self.teacher_model is not None:
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('teacher_model.'):
                    filtered_state_dict[k] = v
            state_dict = filtered_state_dict

        ret = super().load_state_dict(state_dict, strict=strict)
        
        if self.ema:
            ema_sd = getattr(self, "_pending_ema_state", None)
            ema_loaded = False

            if ema_sd is not None:
                try:
                    self.ema.load_state_dict(ema_sd)
                    current_params = list(self._get_parameters())

                    if len(self.ema.shadow_params) == len(current_params):
                        shapes_match = all(
                            s.shape == p.shape
                            for s, p in zip(self.ema.shadow_params, current_params)
                        )
                        if shapes_match:
                            ema_loaded = True
                        else:
                            print("[WARNING] EMA shape mismatch - will reinitialize from loaded weights")
                    else:
                        print("[WARNING] EMA count mismatch - will reinitialize from loaded weights")

                except Exception as e:
                    print(f"[WARNING] Failed to load EMA after weights load: {e}")

            if not ema_loaded:
                print("Initializing EMA from loaded model weights")
                import models.ema
                self.ema = models.ema.ExponentialMovingAverage(
                    list(self._get_parameters()),
                    decay=self.config.training.ema
                )

            self._pending_ema_state = None

        return ret

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self._pending_ema_state = checkpoint.get('ema', None)
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint['loops'][
            'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress'][
            'current']['completed']

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed']
        # is 1 iteration behind, so we're using the optimizer's progress.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['total'][
            'completed'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total'][
            'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['current'][
            'completed'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['current'][
            'completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps,
        # not the number of local steps, so we don't multiply with
        # self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.state_dict'][
            '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total']['completed']
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler,
                   'state_dict'):
            sampler_state_dict = self.trainer.\
                train_dataloader.sampler.state_dict()
            checkpoint['sampler'][
                'random_state'] = sampler_state_dict.get(
                'random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed)
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, 'shuffle'):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if (distributed
                and self.fast_forward_epochs is not None
                    and self.fast_forward_batches is not None):
                dl_sampler.load_state_dict({'epoch': self.fast_forward_epochs, 'counter': (self.fast_forward_batches * self.config.loader.batch_size)})
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=True))
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._get_parameters())

    def _process_sigma(self, sigma):
        raise NotImplementedError

    def _process_model_output(self, model_output, xt, sigma):
        raise NotImplementedError

    def forward(self, xt, sigma, sigma_prime=None, use_jvp_attn=False):

        sigma = self._process_sigma(sigma)
        if sigma_prime is not None:
            sigma_prime = self._process_sigma(sigma_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.backbone(xt, sigma, sigma_prime, use_jvp_attn=use_jvp_attn)
        
        return self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)

    def on_train_epoch_start(self):
        self.metrics.reset()
        assert self.metrics.train_nlls.nll.mean_value == 0
        assert self.metrics.train_nlls.nll.weight == 0

    def training_step(self, batch, batch_idx):
        current_accumulation_step = (
            batch_idx % self.trainer.accumulate_grad_batches)

        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            current_accumulation_step,
                            train_mode=True,
                            xT=None if 'xT' not in batch else batch['xT'],
                            given_t=batch['given_t'] if 'given_t' in batch else None,
                            not_sampling_t=self.config.training.not_sampling_t
                            )
        self.metrics.update_train(losses.nlls, losses.prior_loss,
                                  losses.num_tokens)
        self.log(name='trainer/loss',
                 value=losses.loss.item(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return losses.loss

    def on_train_epoch_end(self):
        # NOTE:
        # Originally, this method re-logged validation NLL metrics at the end
        # of every *training* epoch by iterating over `self.metrics.valid_nlls`
        # and calling `.compute()` again.
        #
        # That extra logging turned out to be a non-trivial bottleneck and also
        # caused `val/*` metrics to appear much more frequently in WandB than
        # actual validation runs (which already log in `on_validation_epoch_end`).
        #
        # We therefore keep this hook but make it a no-op to avoid the
        # unnecessary per-train-epoch metric computation/logging. All
        # validation-related metrics are still logged from
        # `on_validation_epoch_end`, which is called whenever validation runs.
        return

    def on_validation_epoch_start(self):
        self.metrics.reset()
        self._eval_mode()
        assert self.metrics.valid_nlls.nll.mean_value == 0
        assert self.metrics.valid_nlls.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        del batch_idx
        if self.metrics.is_sudoku:
            return
        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            xT=None if 'xT' not in batch else batch['xT']
                            )
        self.metrics.update_valid(losses.nlls, losses.prior_loss,
                                  losses.num_tokens)
        return losses.loss

    def _run_tinygsm_pass(self, dataset, gold_answers, N, *,
                          log_tag, print_label,
                          gen_kwargs_extra=None, out_path=None,
                          row_extra=None):
        """One generation+scoring pass over the first ``N`` rows of
        ``dataset``. Resets and updates ``self.metrics.gsm8k_acc``; writes
        one JSONL row per sample to ``out_path`` (when not None); logs
        ``val/gsm8k_acc{log_tag}`` at epoch end; prints a rank-0 banner
        labelled ``print_label`` plus the first sample's preview. Returns
        the first sample's preview dict (or ``None`` if ``N == 0``).

        Args:
          log_tag: suffix appended to the wandb log key
            (e.g. ``'_T81_k1'`` or ``'_SPB1_BL01'``).
          print_label: short label used in the rank-0 print banner
            (e.g. ``'T81 k=1'`` or ``'SPB1_BL01'``).
          gen_kwargs_extra: kwargs passed through to
            :meth:`generate_samples` (e.g. ``num_steps``, ``keep_k``).
          row_extra: dict of extra fields embedded in each JSONL row
            (and in the preview dict). Used to stamp sweep coordinates
            (``keep_k``, ``block_length``, ``steps_per_block``) into the
            output.

        ``gold`` / ``pred`` are redacted when they're ints exceeding
        ~4000 digits (Python>=3.11 :func:`json.dumps` raises
        ``ValueError`` past that limit — the model can emit code like
        ``10**99999`` whose evaluated result blows past it).
        """
        import json
        import utils_gsm8k
        if N == 0:
            return None
        bsz = int(self.config.loader.eval_batch_size)
        timeout_s = float(self.config.eval.get('gsm8k_exec_timeout', 1.0) or 1.0)
        gen_kwargs_extra = dict(gen_kwargs_extra or {})
        row_extra = dict(row_extra or {})

        def _safe(v):
            if (isinstance(v, int) and not isinstance(v, bool)
                    and v.bit_length() > 13000):
                return f'<huge_int bits={v.bit_length()}>'
            return v

        self.metrics.gsm8k_acc.reset()
        is_rank_zero = (self.trainer.global_rank == 0)
        out_f = open(out_path, 'w') if (out_path is not None and is_rank_zero) else None
        first_preview = None
        try:
            for start in range(0, N, bsz):
                # Clamp to N so the final ragged batch doesn't index past the
                # dataset (N is rarely a multiple of eval_batch_size).
                batch_indices = list(range(start, min(start + bsz, N)))
                cur_bsz = len(batch_indices)
                prompt_tokens = np.stack(
                    [np.asarray(dataset.tokens[i], dtype=np.int64)
                     for i in batch_indices], axis=0)
                prompt_lens = np.array(
                    [int(dataset.prompt_len[i]) for i in batch_indices],
                    dtype=np.int64)

                gen_kwargs = dict(
                    num_samples=cur_bsz,
                    prompt_tokens=prompt_tokens,
                    prompt_lens=prompt_lens,
                    **gen_kwargs_extra,
                )
                samples = self.generate_samples(**gen_kwargs)
                texts = self.tokenizer.batch_decode(samples)

                for j, text in enumerate(texts):
                    gold = gold_answers[start + j]
                    is_correct, pred = utils_gsm8k.score_sample(
                        text, gold, timeout_s=timeout_s, return_answer=True)
                    self.metrics.gsm8k_acc.update(float(bool(is_correct)))
                    plen = int(prompt_lens[j])
                    if out_f is not None:
                        prompt_text = self.tokenizer.decode(samples[j, :plen])
                        completion = self.tokenizer.decode(samples[j, plen:])
                        row = {'idx': start + j, **row_extra,
                               'prompt': prompt_text, 'completion': completion,
                               'gold': _safe(gold), 'pred': _safe(pred),
                               'correct': bool(is_correct)}
                        out_f.write(json.dumps(row, ensure_ascii=False) + '\n')
                    if first_preview is None:
                        completion_p = self.tokenizer.decode(samples[j, plen:])
                        prompt_text_p = self.tokenizer.decode(samples[j, :plen])
                        first_preview = {
                            'prompt': prompt_text_p,
                            'completion': completion_p,
                            'gold': _safe(gold), 'pred': _safe(pred),
                            'correct': bool(is_correct),
                            **row_extra,
                        }
        finally:
            if out_f is not None:
                out_f.close()

        acc = self.metrics.gsm8k_acc.compute()
        self.log(f'val/gsm8k_acc{log_tag}', acc,
                 on_epoch=True, on_step=False, sync_dist=True)
        if is_rank_zero:
            print(f"[eval step={self.global_step} {print_label}] "
                  f"gsm8k_acc={float(acc):.4f} (N={N})")
            if out_path is not None:
                print(f"[eval] wrote {N} generations to {out_path}")
            if first_preview is not None:
                mark = '✓' if first_preview['correct'] else '✗'
                print(f"--- Sample gsm8k ({print_label}) [{mark} "
                      f"gold={first_preview['gold']} "
                      f"pred={first_preview['pred']}] ---")
                print('[PROMPT]')
                print(first_preview['prompt'])
                print('[COMPLETION]')
                print(first_preview['completion'])
                print('---')
        return first_preview

    def _load_gsm8k_test_eval_data(self, n):
        """Load (and cache) the first ``n`` records of the real GSM8K
        test set for in-training accuracy eval. The cache stores the
        widest ``n`` ever requested so subsequent smaller requests are
        served from memory by slicing.
        """
        import gsm8k_test_data
        if self._gsm8k_test_eval_cache is not None:
            d, ga, cached_n = self._gsm8k_test_eval_cache
            if cached_n >= n:
                sub = gsm8k_test_data.GSM8KTestDataset(
                    d.tokens[:n], d.prompt_len[:n])
                return sub, ga[:n], n
        json_path = self.config.eval.get('gsm8k_test_json', None)
        if json_path is None:
            json_path = os.path.join(
                os.path.dirname(__file__),
                'data', 'gsm8k_test.json')
        dataset, gold, kept_n = gsm8k_test_data.load_gsm8k_test_eval(
            json_path=json_path,
            tokenizer=self.tokenizer,
            seq_len=self.num_tokens,
            n=n,
            # Match the prompt->code boundary the model was trained with.
            # The TinyGSM cache uses config.data.separator; e.g. the s-flm MDLM
            # checkpoint was trained with the *literal* two-char string "\n"
            # (YAML single-quoted '\n'), not a real newline. Falling back to a
            # real newline preserves the previous behaviour for FMLM+.
            separator=self.config.data.get('separator', '\n'),
        )
        self._gsm8k_test_eval_cache = (dataset, gold, kept_n)
        if self.trainer.global_rank == 0:
            print(f'[gsm8k_test] loaded {kept_n} eval prompts '
                  f'(requested n={n}) from {json_path}')
        return dataset, gold, kept_n

    def _evaluate_gsm8k_test(self, n):
        """Run real-GSM8K-test eval on the first ``n`` problems and log
        ``val/gsm8k_test_acc_T{steps}`` via ``_run_tinygsm_pass``.
        """
        dataset, gold_answers, kept_n = self._load_gsm8k_test_eval_data(n)
        if kept_n == 0:
            if self.trainer.global_rank == 0:
                print('[gsm8k_test] kept_n=0 — nothing to evaluate.')
            return
        num_steps = int(self.config.sampling.steps)
        save_dir = self._gsm8k_save_dir()
        is_rank_zero = (self.trainer.global_rank == 0)
        step_tag = (f'_step{self.global_step}'
                    if self.global_step else '')
        out_path = (
            os.path.join(save_dir, f'gsm8k_test_T{num_steps}{step_tag}.jsonl')
            if is_rank_zero else None)
        self.metrics.gsm8k_acc.reset()
        self._run_tinygsm_pass(
            dataset, gold_answers, kept_n,
            log_tag=f'_test_T{num_steps}',
            print_label=f'gsm8k_test T{num_steps}',
            gen_kwargs_extra={'num_steps': num_steps},
            out_path=out_path,
        )

    def _gsm8k_save_dir(self):
        """Resolve the rank-0 directory where per-pass JSONL files are
        written. Non-rank-0 ranks get ``None``.
        """
        if self.trainer.global_rank != 0:
            return None
        save_dir = self.config.eval.get('gsm8k_samples_dir', None)
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'gsm8k_samples')
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _load_sflm_sudoku_eval_data(self):
        """Load the cached sflm_sudoku validation split once. Returns a
        numpy (N, 180) int64 array (full input_ids; first 91 are the puzzle
        prompt, last 89 are the ground-truth solution).
        """
        if self._sflm_sudoku_eval_cache is not None:
            return self._sflm_sudoku_eval_cache
        import datasets as hf_datasets
        difficulty = self.config.data.difficulty
        num_train = self.config.data.num_train
        num_valid = self.config.data.num_valid
        data_seed = self.config.data.data_seed
        save_dir = (f'{self.config.data.cache_dir}/sflm_sudoku'
                    f'_{difficulty}_train{num_train}'
                    f'_valid{num_valid}_seed{data_seed}')
        val_path = os.path.join(save_dir, 'validation')
        if not os.path.exists(val_path):
            print(f"[sflm_sudoku eval] {val_path} not found, skipping.")
            self._sflm_sudoku_eval_cache = (None, 0)
            return self._sflm_sudoku_eval_cache
        ds = hf_datasets.load_from_disk(val_path)
        # Optional cap: eval.sflm_sudoku_num_eval (default: full val set).
        cap = self.config.eval.get('sflm_sudoku_num_eval', None)
        if cap is not None:
            ds = ds.select(range(min(int(cap), len(ds))))
        test_data = np.asarray(ds['input_ids'], dtype=np.int64)
        bsz = int(self.config.loader.eval_batch_size)
        N = (len(test_data) // bsz) * bsz
        if N == 0:
            print(f"[sflm_sudoku eval] {len(test_data)} examples < "
                  f"eval_batch_size={bsz}, skipping.")
            self._sflm_sudoku_eval_cache = (None, 0)
            return self._sflm_sudoku_eval_cache
        test_data = test_data[:N]
        self._sflm_sudoku_eval_cache = (test_data, N)
        return self._sflm_sudoku_eval_cache

    def _run_sflm_sudoku_pass(self, test_data, N, num_steps,
                              extra_kwargs=None, log_tag=''):
        """One eval pass over the sflm_sudoku validation set.
        Each row is ``[BOS] puzzle(89) [BOS] solution(89)`` = 180 tokens.
        Conditions on the first 91 tokens and asks the model to fill the
        last 89; correct iff all 89 generated tokens match ground truth.
        Logs ``val/sflm_sudoku_acc_T{num_steps}{tag}``.
        """
        PROMPT_LEN = 91
        SOL_LEN = 89

        bsz = int(self.config.loader.eval_batch_size)
        prompt_lens_np = np.full((bsz,), PROMPT_LEN, dtype=np.int64)
        total_correct = 0
        is_rank_zero = (self.trainer.global_rank == 0)
        logged_sample = False

        out_path = None
        records = [] if is_rank_zero else None
        if is_rank_zero:
            save_dir = self.config.eval.get('sflm_sudoku_samples_dir', None)
            if save_dir is None:
                save_dir = os.path.join(os.getcwd(), 'sflm_sudoku_samples')
            os.makedirs(save_dir, exist_ok=True)
            tag_clean = log_tag.lstrip('_') if log_tag else 'pass'
            step_prefix = f'step{self.global_step}_' if self.global_step else ''
            out_path = os.path.join(
                save_dir, f'{step_prefix}T{num_steps}_{tag_clean}.jsonl')

        for start in range(0, N, bsz):
            end = start + bsz
            batch_np = test_data[start:end]
            gen_kwargs = dict(
                num_samples=bsz,
                num_steps=num_steps,
                prompt_tokens=batch_np,
                prompt_lens=prompt_lens_np,
            )
            if extra_kwargs:
                gen_kwargs.update(extra_kwargs)
            samples = self.generate_samples(**gen_kwargs)
            samples_cpu = samples.detach().cpu()
            generated = samples_cpu[:, PROMPT_LEN:PROMPT_LEN + SOL_LEN]
            gt = torch.from_numpy(batch_np[:, PROMPT_LEN:PROMPT_LEN + SOL_LEN])
            correct = (generated == gt).all(dim=1)
            total_correct += int(correct.sum().item())

            if records is not None:
                for j in range(batch_np.shape[0]):
                    records.append({
                        'idx': start + j,
                        'num_steps': num_steps,
                        'block_length': _maybe_scalar_int(self.config.sampling.get('block_length', 0)),
                        'steps_per_block': _maybe_scalar_int(self.config.sampling.get('steps_per_block', 0)),
                        'prompt': batch_np[j, :PROMPT_LEN].tolist(),
                        'predicted': generated[j].tolist(),
                        'gold': gt[j].tolist(),
                        'correct': bool(correct[j].item()),
                    })

            if not logged_sample and is_rank_zero:
                logged_sample = True
                print(f"\n[sflm_sudoku step={self.global_step} "
                      f"T{num_steps}{log_tag}] sample:")
                print(f"  prompt   : {self.tokenizer.decode(batch_np[0, :PROMPT_LEN])}")
                print(f"  predicted: {self.tokenizer.decode(generated[0].tolist())}")
                print(f"  gold     : {self.tokenizer.decode(gt[0].tolist())}")
                print(f"  correct  : {bool(correct[0].item())}\n")

        accuracy = total_correct / max(N, 1)
        if records is not None and out_path is not None:
            summary = {
                '_summary': True,
                'num_steps': num_steps,
                'block_length': _maybe_scalar_int(self.config.sampling.get('block_length', 0)),
                'steps_per_block': _maybe_scalar_int(self.config.sampling.get('steps_per_block', 0)),
                'accuracy': accuracy,
                'total_correct': total_correct,
                'total': int(N),
                'tag': log_tag.lstrip('_') if log_tag else '',
                'global_step': int(self.global_step),
            }
            with open(out_path, 'w') as out_f:
                out_f.write(json.dumps(summary, ensure_ascii=False) + '\n')
                for rec in records:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        log_key = f'val/sflm_sudoku_acc_T{num_steps}{log_tag}'
        self.log(log_key, accuracy,
                 on_epoch=True, on_step=False, sync_dist=True)
        if is_rank_zero:
            print(f"[sflm_sudoku step={self.global_step} "
                  f"T{num_steps}{log_tag}] acc={accuracy:.4f} (N={N})")
            if out_path is not None:
                print(f"[eval] wrote {N} generations to {out_path}")
        return accuracy

    def _evaluate_sflm_sudoku(self, num_steps):
        """sflm_sudoku eval: prompt = ``[BOS] puzzle(89) [BOS]`` (91 tokens),
        model fills the last 89 with the solution via ``generate_samples``
        (dispatched by ``sampling.method``).
        """
        test_data, N = self._load_sflm_sudoku_eval_data()
        if test_data is None:
            return None
        return self._run_sflm_sudoku_pass(test_data, N, num_steps)

    def on_validation_epoch_end(self):

        if not self.metrics.is_sudoku:
            for k, v in self.metrics.valid_nlls.items():
                self.log(name=k,  value=v.compute(), on_step=False,
                         on_epoch=True, sync_dist=True)

        # Real GSM8K test eval (first N problems). Decoupled from
        # eval.generate_samples / compute_generative_perplexity so it
        # can fire on its own; gated by eval.gsm8k_test_n > 0. The
        # sanity-check rule mirrors the one used by the gen-PPL path:
        # skip during sanity unless eval.compute_perplexity_on_sanity.
        gsm8k_test_n = int(self.config.eval.get('gsm8k_test_n', 0) or 0)
        if (gsm8k_test_n > 0
                and (self.config.eval.compute_perplexity_on_sanity
                     or not self.trainer.sanity_checking)):
            self._evaluate_gsm8k_test(n=gsm8k_test_n)

        if ((self.config.eval.compute_perplexity_on_sanity
             or not self.trainer.sanity_checking)
                and self.config.eval.generate_samples):

            num_steps = int(self.config.sampling.steps)
            if hasattr(self.metrics, 'gen_ppl'):
                self.metrics.gen_ppl.reset()
            if hasattr(self.metrics, 'sample_entropy'):
                self.metrics.sample_entropy.reset()
            if hasattr(self.metrics, 'sudoku_validity'):
                self.metrics.sudoku_validity.reset()

            if self.is_sflm_sudoku:
                self._evaluate_sflm_sudoku(num_steps)
            else:
                current_text_samples = []
                for _ in range(self.config.sampling.num_sample_batches):
                    samples = self.generate_samples(
                        num_samples=self.config.loader.eval_batch_size,
                        num_steps=num_steps
                    )

                    self.metrics.record_entropy(samples)

                    if hasattr(self.metrics, 'sudoku_validity'):
                        self.metrics.record_sudoku_validity(samples)

                    decoded_batch = self.tokenizer.batch_decode(samples)

                    if len(current_text_samples) < self.config.sampling.num_sample_log:
                        current_text_samples.extend(decoded_batch)

                    if self.config.eval.compute_generative_perplexity and not self.metrics.is_sudoku:
                        self.metrics.record_generative_perplexity(
                            decoded_batch, self.num_tokens, self.device)

                if self.config.eval.compute_generative_perplexity and not self.metrics.is_sudoku:
                    gen_ppl_val = self.metrics.gen_ppl.compute()
                    sample_entropy_val = self.metrics.sample_entropy.compute()
                    self.log(f'val/gen_ppl_T{num_steps}', gen_ppl_val,
                             on_epoch=True, on_step=False, sync_dist=True)
                    self.log(f'val/sample_entropy_T{num_steps}', sample_entropy_val,
                             on_epoch=True, on_step=False, sync_dist=True)
                    if self.trainer.global_rank == 0:
                        print(f"[eval step={self.global_step} T{num_steps}] "
                              f"gen_ppl={float(gen_ppl_val):.3f} "
                              f"sample_entropy={float(sample_entropy_val):.3f}")
                        if current_text_samples:
                            print(f"--- Sample (T{num_steps}) ---")
                            print(current_text_samples[0])
                            print("---")

                if (self.trainer.global_rank == 0
                        and hasattr(self.trainer.logger, 'log_table')):
                    log_samples = current_text_samples[:self.config.sampling.num_sample_log]
                    self.trainer.logger.log_table(
                        key=f'samples_T{num_steps}@global_step{self.global_step}',
                        columns=['Generated Samples'],
                        data=[[s] for s in log_samples]
                    )

                if hasattr(self.metrics, 'sudoku_validity'):
                    self.log(f'val/sudoku_validity_T{num_steps}',
                            self.metrics.sudoku_validity.compute(),
                            on_epoch=True, on_step=False,
                            sync_dist=True)
                    if self.trainer.global_rank == 0:
                        print(f"[eval step={self.global_step} T{num_steps}] "
                              f"validity={self.metrics.sudoku_validity.compute():.2f}%")
                        if current_text_samples:
                            print(f"--- Sample sudoku (T{num_steps}) ---")
                            print(current_text_samples[0])
                            print("---")

        self._train_mode()

    def on_test_epoch_start(self):
        self._eval_mode()
        self.xTx0s = []

    def test_step(self, batch, batch_idx):
        xT = batch
        x0 = self.generate_samples(xT.shape[0], xT=xT.detach().clone())
        pair = torch.stack([xT, x0], dim=0)  # 2 B N
        self.xTx0s.append(pair)
        return 0.

    def on_test_epoch_end(self):
        # gather across all GPUs
        self.xTx0s = torch.cat(self.xTx0s, dim=1)  # 2 B N
        torch.distributed.barrier()

        # if multi gpu
        if torch.distributed.is_initialized():
            data_xTx0s_all = [torch.empty_like(self.xTx0s) for _ in range(
                torch.distributed.get_world_size())] if self.trainer.global_rank == 0 else None
            torch.distributed.gather(self.xTx0s,
                                     data_xTx0s_all,
                                     dst=0)

        if self.trainer.global_rank == 0:
            xTx0s = torch.cat(data_xTx0s_all, dim=1).cpu()[
                :, :self.config.sampling.num_reflow_samples]
            xTs, x0s = xTx0s[0], xTx0s[1]

            save_path = self.config.data.cache_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            xTs = xTs.cpu().numpy()
            x0s = x0s.cpu().numpy()
            xT_path = os.path.join(save_path, 'xT.npy')
            x0_path = os.path.join(save_path, 'x0.npy')
            np.save(xT_path, xTs)
            np.save(x0_path, x0s)
            print('xT shape:', xTs.shape)
            print('x0 shape:', x0s.shape)
            print('xT saved to:', xT_path)
            print('x0 saved to:', x0_path)
        return
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._get_parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1,
                    self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {'scheduler': scheduler,
                          'interval': 'step',
                          'monitor': 'val/loss',
                          'name': 'trainer/lr'}
        return [optimizer], [scheduler_dict]

    def generate_samples(self, num_samples, num_steps, eps, xT, given_t):
        raise NotImplementedError

    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        self._eval_mode()

        num_steps = int(self.config.sampling.steps)
        batch_samples = self.generate_samples(
            num_samples=self.config.loader.eval_batch_size,
            num_steps=num_steps,
            eps=eps)
        if isinstance(batch_samples, torch.Tensor):
            batch_samples = [batch_samples[i] for i in range(batch_samples.shape[0])]
        all_samples = list(batch_samples)
        self._train_mode()
        return all_samples

    def _process_model_input(self, x0, valid_tokens):
        raise NotImplementedError

    def nll(self, input_tokens, output_tokens,
            current_accumulation_step=None, train_mode=False):
        raise NotImplementedError

    def _loss(self, x0, valid_tokens,
              current_accumulation_step=None,
              train_mode=False,
              xT=None, given_t=None, not_sampling_t=False):
        (input_tokens, output_tokens,
         valid_tokens) = self._process_model_input(
            x0, valid_tokens)
        loss = self.nll(input_tokens, output_tokens,
                        current_accumulation_step, train_mode)
            

        assert loss.ndim == 2
        if self.ignore_bos:
            loss[:, 1:] = loss[:, 1:]
            valid_tokens[:, 1:] = valid_tokens[:, 1:]

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens

        return Loss(loss=token_nll,
                    nlls=nlls,
                    prior_loss=0.0,
                    num_tokens=num_tokens)
