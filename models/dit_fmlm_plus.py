"""DIT backbone variant with per-token time conditioning for Masked-FLM.

Mirrors FLM_PLUS's DIT behavior: when sigma is (B, L), a separate time embedding is
produced per token and broadcast through adaLN, supporting
masked-diffusion / block-diffusion training and sampling schedules.

Kept separate from models.dit so the original DDiTBlock / DIT classes are
untouched.
"""

import einops
import flash_attn
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit import (
    DIT,
    DDiTBlock,
    apply_rotary_pos_emb,
    modulate_fused,
    torch_compile_deco,
)


class DDiTBlockFLM_PLUS(DDiTBlock):
    """DDiTBlock accepting either global (B, cond_dim) or per-token
    (B, L, cond_dim) conditioning tensor `c`."""

    def forward(self, x, rotary_cos_sin=None, c=None, seqlens=None,
                exclude_last_token=False, use_jvp_attn=False):
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        if self.adaLN:
            _mod = self.adaLN_modulation(c)
            if c.ndim == 2:  # global (B, d) — broadcast to all tokens
                _mod = _mod[:, None]
            (shift_msa, scale_msa, gate_msa, shift_mlp,
             scale_mlp, gate_mlp) = _mod.chunk(6, dim=2)
            x = modulate_fused(x, shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv, 'b s (three h d) -> b s three h d',
            three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype),
                use_flash=not use_jvp_attn)

        if use_jvp_attn:
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = self.custom_sdpa(q, k, v, softcap=self.softcap)
            x = x.transpose(1, 2)
        else:
            x = flash_attn.flash_attn_qkvpacked_func(
                qkv, 0.0, causal=False, softcap=self.softcap)

        x = einops.rearrange(x, 'b s h d -> b s (h d)')

        if self.adaLN:
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, gate_msa, x_skip, self.dropout)
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(
                    self.norm2(x), shift_mlp, scale_mlp)),
                None, gate_mlp, x, self.dropout)
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, scale, x_skip, self.dropout)
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class DIT_FMLM_PLUS(DIT):
    """DIT with per-token sigma support.

    Overrides `__init__` to swap in `DDiTBlockFLM_PLUS` blocks and `forward` to
    route per-token (B, L) sigma through the TimestepEmbedder.
    """

    def __init__(self, config, vocab_size: int):
        super().__init__(config, vocab_size)
        if not self.causal:
            dim = config.model.hidden_size
            cond_dim = config.model.cond_dim
            blocks = []
            for _ in range(config.model.n_blocks):
                blocks.append(DDiTBlockFLM_PLUS(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    adaLN=self.adaLN,
                    dropout=config.model.dropout))
            self.blocks = nn.ModuleList(blocks)

        # Optional per-position clean/noisy segment embedding (BERT/MAE-style),
        # added to the token features right after vocab_embed. Index 0 = noisy
        # (to-predict) position, index 1 = clean context position. Only built
        # when the conditional flow-map objective requests it; otherwise None
        # so other algos pay no parameter cost.
        self.cond_seg_embed = None
        if getattr(config.algo, 'cf_cond_mask_embed', False):
            self.cond_seg_embed = nn.Embedding(2, config.model.hidden_size)
            nn.init.normal_(self.cond_seg_embed.weight, std=0.02)

    @torch_compile_deco
    def forward(self, x, sigma, sigma_prime=None, use_jvp_attn=False,
                seg_ids=None):
        # Class conditioning is handled MDLM-style via a clean class "prompt"
        # token prepended to the sequence (and a correspondingly enlarged
        # vocab), NOT through AdaLN. There is therefore no `y` / label-embedder
        # path here: the time embedding is the only adaLN conditioning signal.
        x = self.vocab_embed(x)
        if self.cond_seg_embed is not None and seg_ids is not None:
            # Per-position clean(1)/noisy(0) tag added into the residual stream.
            x = x + self.cond_seg_embed(seg_ids)

        if self.causal:
            t_cond = None
        else:
            if sigma.ndim == 2:  # per-token (B, L)
                B_s, L_s = sigma.shape
                t_emb = self.sigma_map(sigma.reshape(B_s * L_s)).reshape(B_s, L_s, -1)
            else:  # global (B,)
                t_emb = self.sigma_map(sigma)
            if sigma_prime is not None:
                if sigma_prime.ndim == 2:
                    B_sp, L_sp = sigma_prime.shape
                    emb_fn = (self.sigma_map_prime
                              if self.sigma_map_prime is not None
                              else self.sigma_map)
                    t_prime_emb = emb_fn(
                        sigma_prime.reshape(B_sp * L_sp)
                    ).reshape(B_sp, L_sp, -1)
                else:
                    emb_fn = (self.sigma_map_prime
                              if self.sigma_map_prime is not None
                              else self.sigma_map)
                    t_prime_emb = emb_fn(sigma_prime)
                t_emb = t_emb + t_prime_emb

            t_cond = F.silu(t_emb)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond,
                                   seqlens=None,
                                   exclude_last_token=self.is_di4c,
                                   use_jvp_attn=use_jvp_attn)
            x = self.output_layer(x, c=t_cond)

        return x
