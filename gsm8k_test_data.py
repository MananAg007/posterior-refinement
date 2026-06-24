"""Real GSM8K test-set loader for in-training validation.

The records under ``data/gsm8k_test.json`` are
``{prompt, response_ground_truth}`` pairs taken from the official
GSM8K test split (copied verbatim from /home/mananaga/FLM/s-flm/data,
which is in turn the same data file the PUMA / S-FLM code points at).

Each example is tokenized as ``[BOS] + question + sep`` and
right-padded to ``seq_len``, producing the
``dataset.tokens[i]`` / ``dataset.prompt_len[i]`` protocol that
``TrainerBase._run_tinygsm_pass`` already consumes for TinyGSM eval.
That lets us reuse the existing generate-and-score plumbing — we
just swap the dataset and the gold-answer source.

Gold answers are parsed from the trailing ``#### N`` marker in
``response_ground_truth`` and stringified so they flow through
``utils_gsm8k.score_sample(text, gold)`` unchanged. The model — when
trained on TinyGSM-style programs — still emits Python code in
response to a natural-language GSM8K prompt, so the existing
code-execution scorer is the right comparison.
"""
import json
import re
from typing import List, Optional

import numpy as np


_GSM8K_GOLD_RE = re.compile(r'####\s*(-?[\d.,]+)')


def parse_gold_numeric(response_ground_truth: str) -> Optional[str]:
    m = _GSM8K_GOLD_RE.search(response_ground_truth)
    if m is None:
        return None
    return m.group(1).replace(',', '').strip() or None


class GSM8KTestDataset:
    """Tiny in-memory wrapper matching
    ``dataset.tokens[i]`` / ``dataset.prompt_len[i]``.
    """

    def __init__(self, tokens: List[np.ndarray],
                 prompt_len: List[int]):
        self.tokens = tokens
        self.prompt_len = prompt_len

    def __len__(self):
        return len(self.tokens)


def load_gsm8k_test_eval(json_path: str, tokenizer, seq_len: int,
                         n: Optional[int] = None,
                         separator: str = '\n',
                         max_prompt_frac: float = 0.5):
    """Build a ``GSM8KTestDataset`` + matching ``gold_answers`` list.

    Args:
      json_path: path to a JSON list of
        ``{prompt, response_ground_truth}`` records (GSM8K test set).
      tokenizer: the active HF tokenizer (same one used for training,
        so the model sees consistent token ids).
      seq_len: model sequence length; prompts are right-padded to
        this size with ``tokenizer.pad_token_id`` (falling back to
        0 when undefined).
      n: optional cap on records read from the JSON. ``None`` reads
        all records, then prompt-length filtering may trim further.
      separator: appended after the question (matches s-flm's
        ``get_gsm8k_test_dataset`` which uses ``\\n``).
      max_prompt_frac: skip any example whose tokenized prompt
        exceeds ``max_prompt_frac * seq_len`` so the sampler has
        room to generate. Defaults to half.

    Returns:
      (dataset, gold_answers, kept_n) — ``kept_n`` is the number of
      examples actually retained after filtering and may be less
      than ``n``.
    """
    with open(json_path) as f:
        records = json.load(f)
    if n is not None:
        records = records[:n]

    bos = tokenizer.bos_token_id
    sep_ids = tokenizer(separator,
                        add_special_tokens=False).input_ids
    pad = (tokenizer.pad_token_id
           if tokenizer.pad_token_id is not None else 0)
    max_prompt_len = int(seq_len * max_prompt_frac)

    tokens_arr: List[np.ndarray] = []
    prompt_lens: List[int] = []
    gold: List[Optional[str]] = []

    for rec in records:
        q_ids = tokenizer(rec['prompt'].strip(),
                          add_special_tokens=False).input_ids
        prompt_ids = (([bos] if bos is not None else [])
                      + list(q_ids) + list(sep_ids))
        if len(prompt_ids) > max_prompt_len:
            continue
        padded = np.full(seq_len, pad, dtype=np.int64)
        padded[:len(prompt_ids)] = prompt_ids
        tokens_arr.append(padded)
        prompt_lens.append(len(prompt_ids))
        gold.append(parse_gold_numeric(rec['response_ground_truth']))

    return GSM8KTestDataset(tokens_arr, prompt_lens), gold, len(tokens_arr)
