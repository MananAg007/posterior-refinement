#!/bin/bash
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="/path/to/tinygsm"  # dataset cache directory
CKPT="/path/to/tinygsm.ckpt" # checkpoint to evaluate

python -u -m main \
  hydra.run.dir=outputs/tinygsm/fmlm_plus_tinygsm_gsm8k_eval \
  mode=ppl_eval \
  seed=1 \
  loader.global_batch_size=32 \
  loader.batch_size=32 \
  loader.eval_batch_size=16 \
  loader.num_workers=16 \
  trainer.limit_val_batches=1 \
  data=tinygsm \
  data.cache_dir=${DATA_DIR} \
  model=small \
  model.length=512 \
  algo=fmlm_plus \
  sampling.num_sample_batches=1 \
  trainer.precision=bf16 \
  eval.checkpoint_path=${CKPT} \
  eval.disable_ema=False \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  +eval.gsm8k_test_n=16 \
  +eval.gsm8k_exec_timeout=5.0 \
  sampling.steps=128 \
  sampling.method=refinement \
  sampling.refinement_threshold=0.999 \
  sampling.refinement_fresh_noise=True \
  sampling.refinement_top_k=uniform_budget \
  sampling.refinement_flow_steps=1 \
  sampling.refinement_rounds=32 \
  +wandb.offline=true
