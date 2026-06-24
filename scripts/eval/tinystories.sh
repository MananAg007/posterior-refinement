#!/bin/bash
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="/path/to/tinystories"  # dataset cache directory
CKPT="/path/to/tinystories.ckpt" # checkpoint to evaluate

python -u -m main \
  hydra.run.dir=outputs/tinystories/fmlm_plus \
  mode=sample_eval \
  seed=1 \
  loader.eval_batch_size=128 \
  data=tinystories \
  data.cache_dir=${DATA_DIR} \
  model=mini \
  model.length=128 \
  algo=fmlm_plus \
  sampling.num_sample_batches=8 \
  trainer.precision=bf16 \
  eval.checkpoint_path=${CKPT} \
  eval.disable_ema=False \
  eval.generate_samples=True \
  eval.compute_generative_perplexity=True \
  eval.gen_ppl_eval_model_name_or_path=gpt2-large \
  sampling.method=refinement \
  sampling.refinement_threshold=0.999 \
  sampling.refinement_fresh_noise=True \
  sampling.refinement_top_k=uniform_budget \
  sampling.refinement_flow_steps=8 \
  sampling.refinement_rounds=4 \
  +wandb.offline=true
