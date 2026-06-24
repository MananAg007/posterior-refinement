import json
import os
import itertools
import functools
import inspect
import argparse
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import torch.multiprocessing as _torch_mp
try:
    _torch_mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
torch.load = functools.partial(torch.load, weights_only=False)
from torch.distributed import init_process_group, destroy_process_group
import wandb
import algo
import dataloader
import utils

import numpy as np
from datetime import datetime

import uuid

# Allow torch.load(weights_only=True) to safely unpickle Hydra configs stored in checkpoints
torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig, omegaconf.base.ContainerMetadata, omegaconf.base.Metadata])

omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
    if 'hf' in config.algo.backbone:
        return diffusion_model(
            config, tokenizer=tokenizer).to('cuda')

    return diffusion_model.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        weights_only=False)


@L.pytorch.utilities.rank_zero_only
def _print_config(
        config: omegaconf.DictConfig,
        resolve: bool = True,
        save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            '{}/config_tree.txt'.format(
                config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [
            ('train', train_ds), ('valid', valid_ds)]:
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('Batch input_ids.shape', batch['input_ids'].shape)
        first = batch['input_ids'][0, :k]
        last = batch['input_ids'][0, -k:]
        print(f'First {k} tokens:', tokenizer.decode(first))
        print('ids:', first)
        print(f'Last {k} tokens:', tokenizer.decode(last))
        print('ids:', last)


def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    is_sudoku = getattr(model.metrics, 'is_sudoku', False)
    if not is_sudoku:
        model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides

    num_steps = int(config.sampling.steps)

    samples_path = config.eval.generated_samples_path
    base, ext = os.path.splitext(samples_path)

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Enter eval mode once (EMA swap happens here)
    model._eval_mode()

    if not is_sudoku:
        model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if is_sudoku:
        model.metrics.sudoku_validity.reset()
    all_samples = []

    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / num_steps)
            text_samples = intermediate_samples[-1]
            all_samples.extend(list(text_samples))
        else:
            batch_samples = model.generate_samples(
                num_samples=config.loader.eval_batch_size,
                num_steps=num_steps)
            if isinstance(batch_samples, torch.Tensor):
                tensor_samples = batch_samples
                batch_samples = [batch_samples[i]
                                 for i in range(batch_samples.shape[0])]
            else:
                tensor_samples = None
            model.metrics.record_entropy(batch_samples)
            if is_sudoku:
                if tensor_samples is None:
                    tensor_samples = torch.stack(
                        [s if isinstance(s, torch.Tensor) else torch.tensor(s)
                         for s in batch_samples])
                model.metrics.record_sudoku_validity(tensor_samples)
            text_samples = model.tokenizer.batch_decode(batch_samples)
            if (config.eval.compute_generative_perplexity
                    and not is_sudoku):
                model.metrics.record_generative_perplexity(
                    text_samples, config.model.length, model.device)
            all_samples.extend(list(text_samples))

    generative_ppl = 0.
    entropy = 0.
    validity = 0.
    if not config.sampling.semi_ar:
        entropy = model.metrics.sample_entropy.compute().item()
        if is_sudoku:
            validity = model.metrics.sudoku_validity.compute().item()
            print(f'Steps {num_steps} — Sudoku validity: {validity:.2f}%, '
                  f'Sample entropy: {entropy}')
        else:
            if config.eval.compute_generative_perplexity:
                generative_ppl = model.metrics.gen_ppl.compute().item()
            print(f'Steps {num_steps} — Generative perplexity: '
                  f'{generative_ppl}, Sample entropy: {entropy}')

    # Restore training mode (EMA restore happens here)
    model._train_mode()

    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    out_path = f'{base}_steps_{num_steps}{ext}'
    with fsspec.open(out_path, 'w') as f:
        payload = {'num_steps': num_steps,
                   'entropy': entropy,
                   'generated_seqs': all_samples}
        if is_sudoku:
            payload['sudoku_validity'] = validity
        else:
            payload['generative_ppl'] = generative_ppl
        json.dump(payload, f, indent=4)
    print(f'Samples saved at: {out_path}')


def _eval_ppl(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Perplexity Eval.')

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed)
    trainer.validate(model, valid_ds)


def _train(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wid = config.wandb.get('id')
        if not wid or len(str(wid)) > 16:
            wid = str(uuid.uuid4().hex[:8])
        config.wandb.id = wid
        if config.wandb.get('name'):
            config.wandb.name = f"{config.wandb.name}_{wid}"
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)

    if (config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(
            config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    if config.training.finetune_path != '':
        assert utils.fsspec_exists(config.training.finetune_path)
        model = diffusion_model.load_from_checkpoint(
            config.training.finetune_path,
            tokenizer=tokenizer,
            config=config,
            weights_only=False)
    else:
        model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)

    fit_kwargs = {}
    if 'weights_only' in inspect.signature(trainer.fit).parameters:
        fit_kwargs['weights_only'] = False
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path, **fit_kwargs)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    if config.algo.name == 'fmlm_plus':
        diffusion_model = algo.FMLMPlus
    elif config.algo.name == 'mdlm':
        diffusion_model = algo.MDLM
    else:
        raise ValueError(
            f'Invalid algorithm name: {config.algo.name}')
    kwargs = {'diffusion_model': diffusion_model,
              'config': config,
              'tokenizer': tokenizer,
              'logger': logger}
    if config.mode == 'sample_eval':
        _generate_samples(**kwargs)
    elif config.mode == 'ppl_eval':
        _eval_ppl(**kwargs)
    else:
        _train(**kwargs)


if __name__ == '__main__':
    # allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
