#!/bin/bash

DATA_DIR="/path/to/tinygsm"             # dataset cache directory
SAVE_DIR="/path/to/checkpoints/tinygsm" # checkpoint output directory

python -u -m main \
  mode=train \
  loader.global_batch_size=512 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  data=tinygsm \
  data.cache_dir=${DATA_DIR} \
  model=small \
  model.length=512 \
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
  sampling.block_length=512 \
  sampling.steps_per_block=512 \
  trainer.max_steps=250000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=5000 \
  trainer.limit_val_batches=1 \
  trainer.num_sanity_val_steps=1 \
  +trainer.check_val_every_n_epoch=null \
  optim.lr=3e-4 \
  eval.compute_generative_perplexity=False \
  eval.generate_samples=False \
  +eval.gsm8k_test_n=100 \
  +eval.gsm8k_exec_timeout=5.0 \
  wandb.project=tinygsm \
  wandb.name=fmlm_plus_base \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
  checkpointing.save_dir=${SAVE_DIR}
