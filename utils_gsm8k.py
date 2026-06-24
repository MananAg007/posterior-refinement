"""GSM8K code-execution scoring utilities.

Ported verbatim (with only style tweaks) from
FLM_PLUS/flowmaps_from_scratch/flow_maps/eval/gsm8k_exec.py so we can score
AR-generated TinyGSM samples by executing the produced Python code and
comparing the numeric return value of `simple_math_problem()` against the
gold answer parsed from the GSM8K test set.

Public API:
    score_sample(generated_text, gold_answer, timeout_s) -> bool
    execute_for_answer(code_text, timeout_s) -> Any | None
"""

import contextlib
import math
import re
import signal
import warnings
from decimal import Decimal, InvalidOperation
from typing import Any, Optional


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


@contextlib.contextmanager
def _time_limit(timeout_s: float):
    has_alarm = hasattr(signal, 'SIGALRM') and hasattr(signal, 'setitimer')
    old_handler = None
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def _safe_exec(code: str) -> dict:
    import math as _math

    safe_builtins = {
        'abs': abs, 'min': min, 'max': max, 'sum': sum,
        'len': len, 'range': range, 'enumerate': enumerate,
        'int': int, 'float': float, 'str': str, 'bool': bool,
        'round': round, 'print': print,
        'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
        'zip': zip, 'map': map, 'filter': filter,
        'sorted': sorted, 'reversed': reversed,
    }

    def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'math':
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f'Import blocked: {name}')

    safe_builtins['__import__'] = _limited_import

    ns = {'__builtins__': safe_builtins, 'math': _math}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        exec(code, ns, ns)

    return ns


def _extract_code(text: str) -> str:
    # Jump to the program start FIRST. The decoded sample is the full
    # sequence and begins with the prompt's leading BOS (`<|endoftext|>`);
    # splitting on the stoppers before this would take the content *before*
    # that leading BOS — i.e. the empty string — discarding all the code.
    i = text.find('def ')
    if i != -1:
        text = text[i:]

    # Now trim trailing generation (the eot + PAD tail) after the code.
    for stopper in ['<|endoftext|>', '<|eot_id|>', '</s>']:
        if stopper in text:
            text = text.split(stopper, 1)[0]

    fence = re.search(r'```(?:python)?\s*(.*?)```', text,
                      flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)

    i = text.find('def ')
    if i != -1:
        text = text[i:]

    text = text.strip()

    lines = text.splitlines()
    for k in range(0, min(50, len(lines))):
        candidate = '\n'.join(lines[: len(lines) - k]).strip()
        if not candidate:
            continue
        try:
            compile(candidate, '<sample>', 'exec')
            return candidate
        except SyntaxError:
            continue
    return text


def _to_number(x: Any) -> Optional[Any]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        if abs(x - round(x)) < 1e-6:
            return int(round(x))
        return x
    if isinstance(x, str):
        m = re.search(r'[-+]?\d[\d,]*\.?\d*', x)
        if not m:
            return None
        s = m.group(0).replace(',', '')
        if s.count('.') == 1:
            f = float(s)
            if abs(f - round(f)) < 1e-6:
                return int(round(f))
            return f
        return int(s)
    try:
        import numpy as _np
        if isinstance(x, _np.integer):
            return int(x)
        if isinstance(x, _np.floating):
            xf = float(x)
            if not math.isfinite(xf):
                return None
            if abs(xf - round(xf)) < 1e-6:
                return int(round(xf))
            return xf
    except ImportError:
        pass
    return None


def _numbers_equal(pred: Any, gold: Any) -> bool:
    if pred is None or gold is None:
        return False
    if isinstance(pred, float) or isinstance(gold, float):
        try:
            pred_dec = Decimal(str(pred))
            gold_dec = Decimal(str(gold))
        except (InvalidOperation, ValueError, OverflowError):
            return False
        return abs(pred_dec - gold_dec) <= Decimal('1e-3')
    return int(pred) == int(gold)


def execute_for_answer(code_text: str, timeout_s: float = 1.0) -> Optional[Any]:
    code = _extract_code(code_text)
    try:
        with _time_limit(timeout_s):
            ns = _safe_exec(code)
            fn = ns.get('simple_math_problem', None)
            if fn is None:
                return None
            return fn()
    except Exception:
        return None


def score_sample(
    generated_text: str,
    gold_answer: str,
    timeout_s: float = 1.0,
    return_answer: bool = False,
):
    if gold_answer is None:
        return (False, None) if return_answer else False
    result = execute_for_answer(generated_text, timeout_s=timeout_s)
    pred = _to_number(result)
    gold = _to_number(gold_answer)
    correct = _numbers_equal(pred, gold)
    return (correct, pred) if return_answer else correct
