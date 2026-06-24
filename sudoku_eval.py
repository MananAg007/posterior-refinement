"""Conditional Sudoku evaluation utilities.

Verifies 162-token Sudoku sequences emitted by the cond_sudoku setup:
tokens[:81] = puzzle clues (0 = empty, 1-9 = digit), tokens[81:] = solution
(1-9). A prediction is correct iff (a) the solution part is a valid 9x9
Sudoku and (b) every non-empty clue cell matches the corresponding
solution cell.
"""

import torch


def sudoku_check(pred: torch.Tensor) -> torch.Tensor:
    """Return (B,) bool: solution part forms a valid Sudoku.

    Args:
      pred: (B, 81) integer tensor with values in {1, ..., 9}.
    """
    B = pred.shape[0]
    x = pred.view(B, 9, 9)
    in_range = (x >= 1) & (x <= 9)
    ref = torch.arange(1, 10, device=pred.device, dtype=pred.dtype).view(1, 1, 9)

    def groups_ok(groups: torch.Tensor) -> torch.Tensor:
        sorted_groups, _ = torch.sort(groups, dim=-1)
        return (sorted_groups == ref).all(dim=-1)

    rows_ok = groups_ok(x)
    cols_ok = groups_ok(x.transpose(1, 2))
    blocks = x.view(B, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).contiguous().view(B, 9, 9)
    blocks_ok = groups_ok(blocks)

    return (in_range.all(dim=(1, 2))
            & rows_ok.all(dim=1)
            & cols_ok.all(dim=1)
            & blocks_ok.all(dim=1))


def verify_sudoku(pred: torch.Tensor) -> torch.Tensor:
    """Verify (B, 162) Sudoku sequences. Returns (B,) bool tensor.

    A row is correct iff:
      1. pred[:, 81:] is a valid Sudoku.
      2. For every clue position i in pred[:, :81] with clue != 0,
         pred[:, 81 + i] equals that clue.
    """
    clues = pred[:, :81]
    sol = pred[:, 81:]
    clue_ok = ((clues == 0) | (sol == clues)).all(dim=1)
    sudoku_ok = sudoku_check(sol)
    return clue_ok & sudoku_ok


def format_sudoku_board(tokens) -> str:
    """Pretty-print an 81-token grid as a 9x9 board (0 -> '.')."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    lines = []
    for r in range(9):
        row = tokens[r * 9:(r + 1) * 9]
        row_str = ' '.join(str(v) if v != 0 else '.' for v in row)
        lines.append(row_str)
        if r in (2, 5):
            lines.append('-' * 17)
    return '\n'.join(lines)
