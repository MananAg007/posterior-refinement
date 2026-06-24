#!/bin/bash
export HYDRA_FULL_ERROR=1

DATA_DIR="/path/to/sudoku"        # dataset cache directory
CKPT="/path/to/sudoku_medium.ckpt" # checkpoint to evaluate (match DIFFICULTY)
DIFFICULTY="medium"               # one of: easy, medium, hard

python -u -m main \
  hydra.run.dir=outputs/sflm_sudoku_hard/fmlm_plus_sflm_sudoku_hard_eval \
  mode=ppl_eval \
  seed=1 \
  loader.global_batch_size=200 \
  loader.batch_size=200 \
  loader.eval_batch_size=200 \
  loader.num_workers=8 \
  data=sudoku \
  data.cache_dir=${DATA_DIR} \
  data.difficulty=${DIFFICULTY} \
  model=mini \
  model.length=180 \
  algo=fmlm_plus \
  sampling.num_sample_batches=1 \
  trainer.precision=bf16 \
  eval.checkpoint_path=${CKPT} \
  eval.disable_ema=False \
  eval.generate_samples=True \
  eval.compute_generative_perplexity=False \
  +eval.sflm_sudoku_num_eval=2000 \
  sampling.steps=4 \
  sampling.method=refinement \
  sampling.refinement_threshold=0.999 \
  sampling.refinement_fresh_noise=True \
  sampling.refinement_top_k=uniform_budget \
  sampling.refinement_flow_steps=1 \
  sampling.refinement_rounds=4 \
  +wandb.offline=true
