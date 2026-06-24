#!/bin/bash

DATA_DIR="/path/to/tinystories"             # dataset cache directory
SAVE_DIR="/path/to/checkpoints/tinystories" # checkpoint output directory

python -u -m main \
  mode=train \
  loader.global_batch_size=512 \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  data=tinystories \
  data.cache_dir=${DATA_DIR} \
  model=mini \
  model.length=128 \
  algo=fmlm_plus \
  algo.sample_clean=uniform \
  algo.double_temb=True \
  algo.skip_time_reparam=True \
  algo.diagonal_fraction=0.5 \
  algo.boundary_prob=32 \
  algo.add_boundary=True \
  algo.distillation_method=PSD \
  algo.p_distill=0.0 \
  algo.distill_teacher_ckpt_path=null \
  algo.init_teacher_ckpt_path=null \
  algo.init_teacher_kind=flm_plus \
  algo.diag_teacher_ckpt_path=null \
  strategy.find_unused_parameters=false \
  training.antithetic_sampling=True \
  sampling.num_sample_batches=1 \
  sampling.method=block-diffusion \
  sampling.block_length=128 \
  sampling.steps_per_block=128 \
  trainer.max_steps=100000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=5000 \
  trainer.limit_val_batches=1 \
  trainer.num_sanity_val_steps=1 \
  +trainer.check_val_every_n_epoch=null \
  optim.lr=3e-4 \
  eval.compute_generative_perplexity=True \
  eval.generate_samples=True \
  eval.gen_ppl_eval_model_name_or_path=gpt2-large \
  wandb.project=tinystories \
  wandb.name=fmlm_plus \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
  checkpointing.save_dir=${SAVE_DIR}
