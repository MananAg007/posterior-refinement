import functools
import itertools
import json
import math
import os
import numpy as np

import re
import shutil
import typing
import urllib
import zipfile
from typing import Optional

import datasets
import fsspec
import numpy as np
import requests
import tokenizers
import torch
import transformers

from einops import repeat
import utils

LOGGER = utils.get_logger(__name__)


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x


def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


def scientific_papers_detokenizer(x):
    x = wt_detokenizer(x)
    x = lm1b_detokenizer(x)
    return x


class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, config, model):
        self.config = config
        self.num_samples = config.sampling.num_reflow_samples
        self.batch_size = config.loader.eval_batch_size
        self.create_data(model)

    def create_data(self, model):
        num_devices = torch.cuda.device_count()
        samples_per_gpu = self.num_samples // (self.batch_size * num_devices)
        if self.num_samples % (self.batch_size * num_devices) != 0:
            total_samples = (samples_per_gpu + 1) * \
                self.batch_size * num_devices
        else:
            total_samples = samples_per_gpu * self.batch_size * num_devices
        with torch.no_grad():
            self.data = model.prior_sample(
                total_samples, self.config.model.length).cpu().numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ReflowDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self.load_data()

    def load_data(self):
        cache_dir = self.config.data.cache_dir
        self.x0 = np.load(os.path.join(
            cache_dir, 'x0.npy'), allow_pickle=True)
        self.xT = np.load(os.path.join(
            cache_dir, 'xT.npy'), allow_pickle=True)
        self.given_t = np.load(os.path.join(
            cache_dir, 'ts.npy'), allow_pickle=True) if os.path.exists(os.path.join(cache_dir, 'ts.npy')) else None

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.x0[idx],
            'attention_mask': np.ones_like(self.x0[idx]),
            'xT': self.xT[idx],
            'given_t': self.given_t[idx] if self.given_t is not None else 0.0,
        }


class SyntheticTokenizer(
        transformers.PreTrainedTokenizer):

    def __init__(
            self,
            vocab_size,
            bos_token="[BOS]",
            eos_token="[EOS]",
            sep_token=None,
            cls_token=None,
            pad_token=None,
            mask_token=None,
            unk_token=None,
            **kwargs):

        self.tokens = []

        for i in range(vocab_size - 2):
            # appending space for readability
            self.tokens.append(str(i) + " ")

        self._vocab_str_to_int = {
            '[BOS]': vocab_size - 2,
            '[EOS]': vocab_size - 1,
            ** {ch: i for i, ch in enumerate(self.tokens)}}

        self._vocab_int_to_str = {
            v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(
            token, self._vocab_str_to_int['[UNK]'])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def _generate_synthetic_data(dataset_size,
                             seq_len, vocab_size):
    dataset = np.zeros((dataset_size, seq_len), dtype=int)
    # tokens representing sequence boundary
    dataset[:, 0] = vocab_size - 2  # bos
    dataset[:, -1] = vocab_size - 1  # eos

    for i in range(dataset_size):
        # sample from 0, 1, ..., vocab_size - 3
        temp = np.random.randint(vocab_size - 2)
        for j in reversed(range(1, seq_len - 1)):
            dataset[i, j] = temp
            if temp != 0:
                temp = temp // 4
            else:
                temp = np.random.randint(vocab_size - 2)

    return dataset


def generate_synthetic_dataset(train_dataset_size,
                               validation_dataset_size,
                               seq_len, vocab_size):
    np.random.seed(42)
    train_data = torch.from_numpy(
        _generate_synthetic_data(train_dataset_size,
                                 seq_len, vocab_size))
    train_dataset = datasets.Dataset.from_dict({
        'input_ids': train_data,
        'attention_mask': torch.ones_like(train_data),
    })
    train_dataset.set_format(type='torch')

    np.random.seed(41)
    validation_data = torch.from_numpy(
        _generate_synthetic_data(validation_dataset_size,
                                 seq_len, vocab_size))
    validation_dataset = datasets.Dataset.from_dict({
        'input_ids': validation_data,
        'attention_mask': torch.ones_like(validation_data),
    })
    validation_dataset.set_format(type='torch')

    return {
        'train': train_dataset,
        'validation': validation_dataset,
    }


def generate_alpha8_dataset(
    train_dataset_size: int,
    validation_dataset_size: int,
    seq_len: int = 8,
    num_classes: int = 26,
):
    """Generate ultra-simple sequences where each sample is the same letter repeated.

    - input_ids shape: [N, 8]
    - values: integers in [0, num_classes-1]
    - attention_mask: ones with same shape
    """
    assert seq_len == 8, "alpha8 dataset expects seq_len=8"
    assert 1 <= num_classes <= 26, "num_classes should be within 1..26"

    def _make(N, seed):
        rng = np.random.default_rng(seed)
        letters = rng.integers(low=0, high=num_classes,
                               size=(N, 1), dtype=np.int64)
        arr = np.repeat(letters, seq_len, axis=1)
        x = torch.from_numpy(arr.astype(np.int64))
        ds = datasets.Dataset.from_dict({
            'input_ids': x,
            'attention_mask': torch.ones_like(x),
        })
        ds.set_format(type='torch')
        return ds

    train_ds = _make(train_dataset_size, seed=1234)
    valid_ds = _make(validation_dataset_size, seed=1235)
    return {
        'train': train_ds,
        'validation': valid_ds,
    }

class SyntheticAlign(torch.utils.data.Dataset):
    def __init__(self, config, N=100_000):
        self.config = config
        self.N = N
        self.L = self.config.model.length
        self.build_data()
    
    def build_data(self):
        self.x0 = np.random.randint(0, self.config.data.vocab_size, (self.N, 1))
        self.x0 = repeat(self.x0, 'b 1 -> b l', l=self.L)

    def __len__(self):
        return len(self.x0)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.x0[idx],
            'attention_mask': np.ones_like(self.x0[idx]),
        }
    
class SyntheticTokenizer(
    transformers.PreTrainedTokenizer):
    
    def __init__(
        self,
        vocab_size,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token=None,
        cls_token=None,
        pad_token=None,
        mask_token=None,
        unk_token=None,
        **kwargs):
        
        self.tokens = []
        
        for i in range (vocab_size - 2):
            # appending space for readability
            self.tokens.append(str(i) + ", ")
        
        self._vocab_str_to_int = {
            '[BOS]': vocab_size - 2,
            '[EOS]': vocab_size - 1,
            ** {ch: i for i, ch in enumerate(self.tokens)}}
        
        self._vocab_int_to_str = {
            v: k for k, v in self._vocab_str_to_int.items()}
        
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(
            token, self._vocab_str_to_int['[UNK]'])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def _generate_synthetic_data(dataset_size, 
                                                         seq_len, vocab_size):
    dataset = np.zeros((dataset_size, seq_len), dtype=int)
    # tokens representing sequence boundary
    dataset[:, 0] = vocab_size - 2  # bos
    dataset[:, -1] = vocab_size - 1  # eos

    for i in range(dataset_size):
        # sample from 0, 1, ..., vocab_size - 3
        temp = np.random.randint(vocab_size - 2)
        for j in reversed(range(1, seq_len - 1)):
            dataset[i, j] = temp
            if temp != 0:
                temp = temp // 4
            else:
                temp = np.random.randint(vocab_size - 2)

    return dataset


def generate_synthetic_dataset(train_dataset_size, 
                                                             validation_dataset_size, 
                                                             seq_len, vocab_size):
    np.random.seed(42)
    train_data = torch.from_numpy(
        _generate_synthetic_data(train_dataset_size, 
                                                         seq_len, vocab_size))
    train_dataset = datasets.Dataset.from_dict({
        'input_ids': train_data, 
        'attention_mask': torch.ones_like(train_data),
    })
    train_dataset.set_format(type='torch')

    np.random.seed(41)
    validation_data = torch.from_numpy(
        _generate_synthetic_data(validation_dataset_size, 
                                                         seq_len, vocab_size))
    validation_dataset = datasets.Dataset.from_dict({
        'input_ids': validation_data, 
        'attention_mask': torch.ones_like(validation_data),
    })
    validation_dataset.set_format(type='torch')

    return {
        'train': train_dataset,
        'validation': validation_dataset,
    }
    
class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(
            self,
            bos_token='[BOS]',
            eos_token='[EOS]',
            sep_token='[SEP]',
            cls_token='[CLS]',
            pad_token='[PAD]',
            mask_token='[MASK]',
            unk_token='[UNK]',
            **kwargs):
        self.characters = list('abcdefghijklmnopqrstuvwxyz ')
        self._vocab_str_to_int = {
            '[CLS]': 0,
            '[SEP]': 1,
            '[BOS]': 2,
            '[EOS]': 3,
            '[MASK]': 4,
            '[PAD]': 5,
            '[RESERVED]': 6,
            '[UNK]': 7,
            ** {ch: i + 8 for i, ch in enumerate(self.characters)}}
        self._vocab_int_to_str = {
            v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(
            token, self._vocab_str_to_int['[UNK]'])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


class Alpha8Tokenizer(transformers.PreTrainedTokenizer):
    """Minimal tokenizer for 26 lowercase letters.

    IDs:
      - 'a'..'z' -> 0..25
      - [BOS]=26, [EOS]=27, [PAD]=28, [UNK]=29
    """

    def __init__(
            self,
            bos_token='[BOS]',
            eos_token='[EOS]',
            pad_token='[PAD]',
            unk_token='[UNK]',
            sep_token=None,
            cls_token=None,
            mask_token=None,
            **kwargs):
        self.characters = list('abcdefghijklmnopqrstuvwxyz')
        self._vocab_str_to_int = {
            '[BOS]': 26,
            '[EOS]': 27,
            '[PAD]': 28,
            '[UNK]': 29,
            **{ch: i for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k,
                                  v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int['[UNK]'])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def get_pseudo_dataloader(config, tokenizer, model):
    dataset = PseudoDataset(config, model)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.loader.eval_batch_size,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=False,
        persistent_workers=True
    )
    return dataloader


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset


def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
    """Adapted from:
      https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

      Args:
        cache_dir: str, path to cache directory.
        max_seq_length: int, maximum length of sequences.
            (default: 256, as in D3PM codebase.)
        drop_last: bool, whether to drop the last incomplete
            batch. (default: True, as in D3PM codebase.)
        crop_train: bool, whether to subsample contiguous
            subsequences from training example. serves to
            make sure transformer models with absolute position
            embeddings do not have incorrect position-wise
            marginals. (default: False, but necessary to match D3PM AR)

      Returns:
        dataset: dataset.DatasetDict, with keys 'train',
            'valid', 'test'.
    """
    url = 'http://mattmahoney.net/dc/text8.zip'
    if not crop_train:
        cache_dir = f'{cache_dir}/text8'
    else:
        cache_dir = f'{cache_dir}/text8-crop-train'
    split_names = ['train', 'validation', 'test']
    if not all([
        utils.fsspec_exists(os.path.join(cache_dir, split))
        for split in split_names
    ]):
        # Check if raw data exists
        raw_cache_dir = os.path.join(cache_dir, 'raw_data')
        if not all([
            utils.fsspec_exists(
                os.path.join(raw_cache_dir, f'text8.{split}.txt'))
            for split in split_names
        ]):
            if not utils.fsspec_exists(
                    os.path.join(raw_cache_dir, 'text8.zip')):
                utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
                LOGGER.info('Downloading text8 from URL {}.'.format(url))
                with (urllib.request.urlopen(url) as in_stream,
                      open(os.path.join(raw_cache_dir, 'text8.zip'),
                           'wb') as out_file):
                    shutil.copyfileobj(in_stream, out_file)

            with fsspec.open(
                    os.path.join(raw_cache_dir, 'text8.zip'),
                    'rb') as f:
                rawdata = zipfile.ZipFile(f).read(
                    'text8').decode('utf-8')

            # Splits taken from D3PM codebase
            splits = {
                'train': rawdata[:90000000],
                'validation': rawdata[90000000: 95000000],
                'test': rawdata[95000000:],
            }

            for split, data in splits.items():
                _path = os.path.join(raw_cache_dir,
                                     f'text8.{split}.txt')
                with fsspec.open(_path, 'w') as f:
                    f.write(data)
        else:
            splits = {}
            for split in split_names:
                _path = os.path.join(raw_cache_dir,
                                     f'text8.{split}.txt')
                with fsspec.open(_path, 'r') as f:
                    splits[split] = f.read()

        # Chunk and save as datasets.DatasetDict
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        dataset_dict = {}
        for k, v in splits.items():
            if k == 'train' and crop_train == True:
                chunk_size = 2 * max_seq_length
            else:
                chunk_size = max_seq_length
            text = list(chunks(v, chunk_size))
            if drop_last and len(text[-1]) < chunk_size:
                text = text[:-1]
            dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        dataset = datasets.load_from_disk(cache_dir)

    return dataset


def _group_texts(examples, block_size, bos, eos):
    # Concatenate all texts.
    concatenated_examples = list(itertools.chain(* examples['input_ids']))
    total_length = len(concatenated_examples)
    # TODO(yair): look into not dropping the remainder but rather padding it.
    # We drop the small remainder, and if the total_length < block_size - 2
    # we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of
    # this drop, you can customize this part to your needs.
    new_block_size = block_size - 2  # [BOS] and [EOS] to be added
    total_length = (total_length // new_block_size) * new_block_size
    # Split by chunks of max_len.
    result = {}
    _values = []
    _attn_masks = []
    for i in range(0, total_length, new_block_size):
        _values.append(
            [bos]
            + concatenated_examples[i: i + new_block_size]
            + [eos])
        _attn_masks.append(torch.ones(block_size))
    result['input_ids'] = _values
    result['attention_mask'] = _attn_masks
    return result


class SflmSudokuTokenizer:
    """Port of s-flm `SudokuTokenizer` (see ~/FLM/s-flm/dataloader.py).

    Vocabulary (12 tokens):
      0     - empty cell (in partial grids)
      1-9   - cell digits
      10    - row separator (also doubles as EOS in s-flm)
      11    - BOS

    Sequence layout:
      total_seq_len = 180  ([BOS] puzzle(89) [BOS] solution(89))
      prompt_len    = 91   ([BOS] + puzzle(89) + [BOS])
      seq_len       = 89   per grid (81 digits + 8 row separators)
    """

    def __init__(self):
        self.empty_token_id = 0
        self.row_separator_id = 10
        self.bos_token_id = 11
        self.vocab_size = 12
        self.eos_token_id = 10
        self.pad_token_id = None
        self.mask_token = None
        self.mask_token_id = None

        self.grid_size = 9
        self.num_cells = self.grid_size * self.grid_size
        self.num_separators = self.grid_size - 1
        self.seq_len = self.num_cells + self.num_separators
        self.prompt_len = 1 + self.seq_len + 1
        self.total_seq_len = 2 * self.seq_len + 2

        self._id_to_str = {
            0: '_', 1: '1', 2: '2', 3: '3', 4: '4',
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: '|', 11: '<BOS>',
        }

    def __len__(self):
        return self.vocab_size

    def decode(self, token_ids):
        return ' '.join(self._id_to_str.get(int(t), '?')
                        for t in token_ids)

    def batch_decode(self, sequences, **kwargs):
        return [self.decode(seq) for seq in sequences]


def _sflm_get_sudoku_dataset(config, tokenizer):
    """Generate or load the s-flm sudoku dataset (HF DatasetDict).

    Each example is `[BOS] partial_grid(89) [BOS] solution(89)` (180 tokens),
    with attention_mask 0 on the prompt (first 91) and 1 on the solution
    (last 89). The cache directory is keyed on (difficulty, num_train,
    num_valid, seed) so changing those produces a fresh cache.
    """
    import sudoku_generator

    assert isinstance(tokenizer, SflmSudokuTokenizer), (
        f'sflm_sudoku dataset requires SflmSudokuTokenizer, '
        f'got {type(tokenizer).__name__}')
    difficulty = config.data.difficulty
    num_train = config.data.num_train
    num_valid = config.data.num_valid
    data_seed = config.data.data_seed
    num_workers = config.data.sudoku_num_workers

    save_dir = (f'{config.data.cache_dir}/sflm_sudoku'
                f'_{difficulty}_train{num_train}'
                f'_valid{num_valid}_seed{data_seed}')
    split_names = ['train', 'validation']
    if not all([
        utils.fsspec_exists(os.path.join(save_dir, split))
        for split in split_names
    ]):
        LOGGER.info(
            f'Generating sflm_sudoku dataset: difficulty={difficulty}, '
            f'num_train={num_train}, num_valid={num_valid}, '
            f'seed={data_seed}.')
        raw = sudoku_generator.generate_sudoku_dataset(
            num_train=num_train,
            num_valid=num_valid,
            difficulty=difficulty,
            seed=data_seed,
            tokenizer=tokenizer,
            num_workers=num_workers)
        dataset = datasets.DatasetDict({
            'train': datasets.Dataset.from_dict({
                'input_ids': raw['train']['input_ids'],
                'attention_mask': raw['train']['attention_mask'],
            }),
            'validation': datasets.Dataset.from_dict({
                'input_ids': raw['validation']['input_ids'],
                'attention_mask': raw['validation']['attention_mask'],
            }),
        })
        dataset.save_to_disk(save_dir)
        LOGGER.info(f'sflm_sudoku dataset saved to {save_dir}.')
    else:
        LOGGER.info(f'Loading sflm_sudoku dataset from {save_dir}.')
        dataset = datasets.load_from_disk(save_dir)
    return dataset


def _sflm_get_tiny_gsm_dataset(config, tokenizer):
    """Port of s-flm `get_tiny_gsm_dataset` (HF DatasetDict).

    Downloads `TinyGSM/TinyGSM`, tokenizes each example as
    `[BOS] question sep answer [EOS]`, then either wraps (concat+chunk to
    `block_size`) or pads to `block_size`. attention_mask layout matches
    s-flm: 1 on supervised positions (answer + EOS, plus prompt if
    `train_on_prompt`, plus pad if `train_on_pad`), 0 elsewhere.

    The on-disk cache key is identical to s-flm's so caches built by one
    repo can be reused by the other.
    """
    mask_tag = 'full' if config.data.train_on_prompt else 'answer_only'
    filter_tag = '_filtered' if config.data.filter_too_long else ''
    wrap_tag = '_wrapped' if config.data.wrap else ''
    train_on_pad = '_train_on_pad' if config.data.train_on_pad else ''
    tokenizer_tag = config.data.tokenizer_name_or_path.replace('/', '__')
    save_dir = (f'{config.data.cache_dir}/tiny_gsm_bs{config.model.length}'
                f'_{mask_tag}{filter_tag}{wrap_tag}{train_on_pad}_{tokenizer_tag}')
    split_names = ['train', 'validation']
    if not all([
        utils.fsspec_exists(os.path.join(save_dir, split))
        for split in split_names
    ]):
        LOGGER.info('Preparing tinygsm dataset.')
        ds = datasets.load_dataset(
            'TinyGSM/TinyGSM', split='train',
            cache_dir=config.data.cache_dir)

        EOS = tokenizer.eos_token_id
        BOS = tokenizer.bos_token_id
        PAD = tokenizer.pad_token_id
        sep_ids = tokenizer(config.data.separator,
                            add_special_tokens=False).input_ids
        block_size = config.model.length
        train_on_prompt = config.data.train_on_prompt

        def tokenize_qa(example):
            q_ids = tokenizer(
                example['question'].strip(),
                add_special_tokens=False).input_ids
            a_ids = tokenizer(
                example['code'].strip(),
                add_special_tokens=False).input_ids
            ids = [BOS] + q_ids + sep_ids + a_ids + [EOS]
            prompt_len = 1 + len(q_ids) + len(sep_ids)
            return {'input_ids': ids, 'prompt_len': prompt_len}

        tokenized = ds.map(
            tokenize_qa,
            num_proc=config.loader.num_workers,
            remove_columns=ds.column_names,
            desc='Tokenizing tinygsm')

        if config.data.filter_too_long:
            assert not config.data.wrap
            before = len(tokenized)
            tokenized = tokenized.filter(
                lambda x: len(x['input_ids']) <= block_size,
                num_proc=config.loader.num_workers,
                desc='Filtering too-long examples')
            LOGGER.info(
                f'Filtered tinygsm: {before} -> {len(tokenized)} '
                f'({before - len(tokenized)} removed)')

        if config.data.wrap:
            tokenized = tokenized.remove_columns('prompt_len')

            def wrap_batch(examples):
                all_ids = list(itertools.chain.from_iterable(
                    examples['input_ids']))
                total = (len(all_ids) // block_size) * block_size
                chunks = [all_ids[i:i + block_size]
                          for i in range(0, total, block_size)]
                masks = [[1] * block_size] * len(chunks)
                return {'input_ids': chunks, 'attention_mask': masks}

            tokenized = tokenized.map(
                wrap_batch,
                batched=True,
                batch_size=1000,
                num_proc=config.loader.num_workers,
                remove_columns=tokenized.column_names,
                desc='Wrapping tinygsm')
        else:
            def pad_and_mask(example):
                ids = example['input_ids']
                n = len(ids)
                prompt_len = example['prompt_len']
                if n >= block_size:
                    ids = ids[:block_size - 1] + [EOS]
                else:
                    ids = ids + [PAD] * (block_size - n)
                mask_start = 0 if train_on_prompt \
                    else min(prompt_len, block_size)
                if config.data.train_on_pad:
                    mask_end = block_size
                else:
                    mask_end = min(n, block_size)
                mask = [0] * mask_start \
                    + [1] * (mask_end - mask_start) \
                    + [0] * (block_size - mask_end)
                return {'input_ids': ids, 'attention_mask': mask}
            tokenized = tokenized.map(
                pad_and_mask,
                num_proc=config.loader.num_workers,
                desc='Padding tinygsm')

        tmp = tokenized.train_test_split(
            test_size=config.data.val_ratio,
            seed=config.data.val_seed)
        dataset = datasets.DatasetDict({
            'train': tmp['train'],
            'validation': tmp['test']})
        dataset.save_to_disk(save_dir)
    else:
        LOGGER.info(f'Loading tinygsm dataset from {save_dir}.')
        dataset = datasets.load_from_disk(save_dir)
    return dataset


def get_dataset(dataset_name,
                tokenizer,
                wrap,
                mode,
                cache_dir,
                insert_eos=True,
                block_size=1024,
                num_proc=len(os.sched_getaffinity(0)),
                streaming=False,
                revision: Optional[str] = None,
                config=None):
    eos_tag = ''
    if not insert_eos:
        eos_tag = '_eosFalse'
    if wrap:
        filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat'
    else:
        filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped{eos_tag}.dat'
    _path = os.path.join(cache_dir, filename)

    if utils.fsspec_exists(_path):
        LOGGER.info(f'Loading data from: {_path}')
        return datasets.load_from_disk(_path).with_format('torch')
    LOGGER.info(f'Generating new data at: {_path}')
    LOGGER.info(f'{streaming=}')

    crop_train = dataset_name == 'text8-crop'
    if mode == 'train' and crop_train:
        # double block size for sub-sampling
        block_size *= 2

    if dataset_name == 'wikitext103':
        dataset = datasets.load_dataset(
            'wikitext',
            name='wikitext-103-raw-v1',
            cache_dir=cache_dir,
            revision=revision)
    elif dataset_name == 'wikitext2':
        dataset = datasets.load_dataset(
            'wikitext',
            name='wikitext-2-raw-v1',
            cache_dir=cache_dir,
            revision=revision)
    elif dataset_name == 'ptb':
        dataset = datasets.load_dataset(
            'ptb_text_only',
            cache_dir=cache_dir,
            revision=revision)
    elif dataset_name == 'lambada':
        dataset = get_lambada_test_dataset()
    elif dataset_name == 'synthetic-align':
        dataset = SyntheticAlign(config)
        return dataset
    elif dataset_name == 'text8':
        assert wrap
        assert revision is None
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size)
    elif dataset_name == 'text8-1k':
        assert wrap
        assert revision is None
        full_dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size)
        train_subset = full_dataset['train'].select(
            range(min(1000, len(full_dataset['train']))))
        dataset = datasets.DatasetDict({
            'train': train_subset,
            'validation': train_subset,
            'test': full_dataset['test']
        })
    elif dataset_name == 'text8-crop':
        assert revision is None
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size, crop_train=True)
    elif dataset_name == 'openwebtext-train':
        dataset = datasets.load_dataset(
            'openwebtext',
            split='train[:-100000]',
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            num_proc=num_proc,
            trust_remote_code=True)
    elif dataset_name == 'openwebtext-valid':
        dataset = datasets.load_dataset(
            'openwebtext',
            split='train[-100000:]',
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            num_proc=num_proc,
            trust_remote_code=True)
    elif dataset_name == 'scientific_papers_arxiv':
        dataset = datasets.load_dataset(
            'scientific_papers', 'arxiv',
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision)
    elif dataset_name == 'scientific_papers_pubmed':
        dataset = datasets.load_dataset(
            'scientific_papers', 'pubmed',
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision)
    elif dataset_name == 'ag_news':
        dataset = datasets.load_dataset(
            'ag_news',
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision)
    elif dataset_name == 'synthetic-alpha8':
        # Extremely simple debugging dataset: sequences like aaaaaaaa, gggggggg (IDs 0..25)
        dataset = generate_alpha8_dataset(
            train_dataset_size=10000,
            validation_dataset_size=128,  # Small validation set for fast debugging
            seq_len=8,
            num_classes=26,
        )
    elif dataset_name == 'synthetic':
        assert streaming
        assert wrap  # i.e., no pad tokens
        dataset = generate_synthetic_dataset(
            train_dataset_size=100000,
            validation_dataset_size=1024,
            seq_len=32,
            vocab_size=256,
        )
    elif dataset_name == 'reflow-dataset':
        dataset = ReflowDataset(config)
        return dataset
    elif dataset_name == 'tinygsm':
        dataset = _sflm_get_tiny_gsm_dataset(config, tokenizer)
        split = 'train' if mode == 'train' else 'validation'
        return dataset[split].with_format('torch')
    elif dataset_name == 'sudoku':
        dataset = _sflm_get_sudoku_dataset(config, tokenizer)
        split = 'train' if mode == 'train' else 'validation'
        return dataset[split].with_format('torch')
    else:
        dataset = datasets.load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,
            revision=revision)

    if dataset_name in ['lambada', 'openwebtext-train',
                        'openwebtext-valid', 'Skylion007/openwebtext-1k',
                        'openwebtext-20']:
        data = dataset
    else:
        data = dataset[mode]
        if dataset_name in ['synthetic', 'synthetic-alpha8']:
            return data

    if dataset_name.startswith('wikitext'):
        detokenizer = wt_detokenizer
    elif dataset_name == 'ptb':
        detokenizer = ptb_detokenizer
    elif dataset_name == 'lm1b':
        detokenizer = lm1b_detokenizer
    elif dataset_name == 'lambada':
        detokenizer = lambada_detokenizer
    elif dataset_name.startswith('scientific_papers'):
        detokenizer = scientific_papers_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text
        return detok

    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    BOS = tokenizer.encode(tokenizer.bos_token)[0]

    def preprocess_and_tokenize(example):
        if dataset_name == 'ptb':
            text = example['sentence']
        elif 'scientific_papers' in dataset_name:
            text = example['article']
        else:
            text = example['text']

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'

        if wrap:
            tokens = tokenizer(text,
                               add_special_tokens=False,
                               return_attention_mask=False,
                               return_token_type_ids=False)
            if insert_eos:
                tokens = {'input_ids':
                          [t + [EOS] for t in tokens['input_ids']]}
            # Still missing BOS, but will be added in group_texts
        else:
            tokens = tokenizer(text,
                               max_length=block_size,
                               padding='max_length',
                               truncation=True,
                               add_special_tokens=True,
                               return_attention_mask=True,
                               return_token_type_ids=True)
        return tokens

    if streaming:
        tokenized_dataset = data.map(
            preprocess_and_tokenize,
            batched=True)
    else:
        tokenized_dataset = data.map(
            preprocess_and_tokenize,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc='Tokenizing')
    if dataset_name == 'ptb':
        tokenized_dataset = tokenized_dataset.remove_columns(
            'sentence')
    elif 'scientific_papers' in dataset_name:
        tokenized_dataset = tokenized_dataset.remove_columns([
            'article', 'abstract', 'section_names'])
    elif dataset_name == 'ag_news':
        tokenized_dataset = tokenized_dataset.remove_columns(
            ['text', 'label'])
    else:
        tokenized_dataset = tokenized_dataset.remove_columns(
            'text')

    if not wrap:
        if not streaming:
            tokenized_dataset.save_to_disk(_path)
        return tokenized_dataset.with_format('torch')

    group_texts = functools.partial(
        _group_texts, block_size=block_size, bos=BOS, eos=EOS)
    if streaming:
        chunked_dataset = tokenized_dataset.map(
            group_texts,
            batched=True)
    else:
        chunked_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc='Grouping')
        chunked_dataset.save_to_disk(_path)
    chunked_dataset = chunked_dataset.with_format('torch')
    return chunked_dataset


class VocabSizeTokenizerWrapper:
    """Tokenizer wrapper that reports ``vocab_size == len(tokenizer)`` (i.e.
    counting tokens added after construction, such as ``[PAD]``).

    Ported from the s-flm repo. This repo's ``get_tokenizer`` does not wrap
    tokenizers, but s-flm-trained checkpoints (e.g. the MDLM TinyGSM ckpt)
    pickled a reference to this class in their saved hyperparameters, so it must
    exist here for ``torch.load`` to unpickle them. ``algo.MDLM`` mirrors the
    wrapper's effect directly by sizing its vocab from ``len(tokenizer)``.
    """

    def __init__(self, tokenizer):
        object.__setattr__(self, '_tokenizer', tokenizer)

    def _wrapped(self):
        return object.__getattribute__(self, '_tokenizer')

    @property
    def vocab_size(self):
        return len(self._wrapped())

    def __len__(self):
        return len(self._wrapped())

    def __call__(self, *args, **kwargs):
        return self._wrapped()(*args, **kwargs)

    def __getattr__(self, name):
        if name == '_tokenizer':
            raise AttributeError(name)
        return getattr(self._wrapped(), name)

    def __setattr__(self, name, value):
        if name == '_tokenizer':
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped(), name, value)

    def __repr__(self):
        return f'Wrapped<{self._wrapped()}>'


def get_tokenizer(config):
    if config.data.tokenizer_name_or_path == 'text8':
        tokenizer = Text8Tokenizer()
    elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
        tokenizer = transformers.BertTokenizer.\
            from_pretrained('bert-base-uncased')
    elif config.data.tokenizer_name_or_path == 'synthetic-align':
        tokenizer = SyntheticTokenizer(vocab_size=config.data.vocab_size+2)
    elif config.data.tokenizer_name_or_path in ['alpha8', 'synthetic-alpha8']:
        tokenizer = Alpha8Tokenizer()
    elif config.data.tokenizer_name_or_path == 'sudoku':
        return SflmSudokuTokenizer()
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.data.tokenizer_name_or_path)

    if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
            or isinstance(tokenizer, transformers.GPT2Tokenizer)):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id))

    # For wrapped batches:
    #  [BOS] sent1 [EOS] sent2-fragment [EOS]
    #  [BOS] sent2-fragment [EOS] sent3 [EOS]
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                'Tokenizer must have a bos_token or '
                f'cls_token: {tokenizer}')
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
                'Tokenizer must have a eos_token '
                f'or sep_token: {tokenizer}')
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
    num_gpus = torch.cuda.device_count()
    assert (config.loader.global_batch_size
            == (config.loader.batch_size
                * config.trainer.num_nodes
                * num_gpus
                * config.trainer.accumulate_grad_batches))
    if config.loader.global_batch_size % (
            num_gpus * config.trainer.accumulate_grad_batches) != 0:
        raise ValueError(
            f'Train Batch Size {config.training.batch_size}'
            f'not divisible by {num_gpus} gpus with accumulation '
            f'{config.trainer.accumulate_grad_batches}.')
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
            f'Eval Batch Size for {config.loader.eval_batch_size} '
            f'not divisible by {num_gpus}.')
    if skip_train:
        train_set = None
    else:
        train_set = get_dataset(
            config.data.train,
            tokenizer,
            mode='train',
            wrap=config.data.wrap,
            insert_eos=config.data.insert_train_eos,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            streaming=config.data.streaming,
            num_proc=config.loader.num_workers,
            revision=config.data.get("train_revision", None),
            config=config)

    if config.data.valid in ['text8', 'lm1b', 'ag_news']:
        validation_split = 'test'
    else:
        validation_split = 'validation'
    if skip_valid:
        valid_set = None
    else:
        valid_set = get_dataset(
            config.data.valid,
            tokenizer,
            wrap=config.data.wrap,
            mode=validation_split,
            cache_dir=config.data.cache_dir,
            insert_eos=config.data.insert_valid_eos,
            block_size=config.model.length,
            streaming=config.data.streaming,
            num_proc=config.loader.num_workers,
            revision=config.data.get("valid_revision", None),
            config=config)

    if skip_train:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=not config.data.streaming,
            persistent_workers=True)
        train_loader.tokenizer = tokenizer
    if skip_valid:
        valid_loader = None
    else:
        if valid_seed is None:
            shuffle_valid = False
            generator = None
        else:
            shuffle_valid = True
            generator = torch.Generator().manual_seed(valid_seed)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=shuffle_valid,
            generator=generator)
        # Will be used in generative perplexity calculation
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called beforehand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop('shuffle', None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'random_state': self.generator.get_state(),
                'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get('random_state'))
        self.counter = state_dict['counter']
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'epoch': self.epoch, 'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.counter = state_dict['counter']
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(
                    padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
