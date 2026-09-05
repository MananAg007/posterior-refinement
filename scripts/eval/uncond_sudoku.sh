#!/bin/bash
export HYDRA_FULL_ERROR=1

CKPT="/path/to/uncond_sudoku.ckpt"

python -u -m main \
  hydra.run.dir=outputs/uncond_sudoku/fmlm_plus \
  mode=sample_eval \
  seed=1 \
  loader.eval_batch_size=256 \
  data=sudoku-uncond \
  model=mini \
  model.length=81 \
  algo=fmlm_plus \
  sampling.num_sample_batches=8 \
  trainer.precision=bf16 \
  eval.checkpoint_path=${CKPT} \
  eval.disable_ema=False \
  eval.generate_samples=True \
  eval.compute_generative_perplexity=False \
  sampling.steps=1 \
  sampling.method=refinement \
  sampling.refinement_threshold=0.999 \
  sampling.refinement_fresh_noise=True \
  sampling.refinement_top_k=uniform_budget \
  sampling.refinement_flow_steps=1 \
  sampling.refinement_rounds=3 \
  +wandb.offline=true
