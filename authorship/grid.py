"""BigP3BCI 6x6 grid alphabet recovered from the EDF per-symbol flash channels.

Each BigP3BCI EDF exposes one channel per grid symbol, named ``<label>_<row>_<column>``.
Those channels are the authoritative record of which symbols were illuminated on each
flash, which matters because checkerboard conditions illuminate arbitrary symbol subsets
rather than whole rows or columns.
"""

from __future__ import annotations

FLASH_CHANNELS: tuple[str, ...] = (
    "A_1_1", "B_1_2", "C_1_3", "D_1_4", "E_1_5", "F_1_6",
    "G_2_1", "H_2_2", "I_2_3", "J_2_4", "K_2_5", "L_2_6",
    "M_3_1", "N_3_2", "O_3_3", "P_3_4", "Q_3_5", "R_3_6",
    "S_4_1", "T_4_2", "U_4_3", "V_4_4", "W_4_5", "X_4_6",
    "Y_5_1", "Z_5_2", "Sp_5_3", "1_5_4", "2_5_5", "3_5_6",
    "4_6_1", "5_6_2", "6_6_3", "7_6_4", "8_6_5", "9_6_6",
)

SPACE_LABEL = "Sp"


def parse_symbol_channel(name: str) -> tuple[str, int, int]:
    """Split a flash-channel name into its display symbol, row and column."""
    label, row, column = name.rsplit("_", 2)
    symbol = " " if label == SPACE_LABEL else label
    return symbol, int(row), int(column)


SYMBOLS: tuple[str, ...] = tuple(parse_symbol_channel(name)[0] for name in FLASH_CHANNELS)

_SYMBOL_TO_INDEX = {symbol: index for index, symbol in enumerate(SYMBOLS)}


def target_code_to_symbol(code: int) -> str:
    """Map a BigP3BCI target or selection code to its display symbol.

    The archive codes selections as one-based indices into the grid in the same order
    the per-symbol flash channels appear in the EDF header.
    """
    if not 1 <= code <= len(SYMBOLS):
        raise ValueError(f"target code out of range: {code}")
    return SYMBOLS[code - 1]


def symbol_index(symbol: str) -> int:
    """Return the zero-based alphabet position of a display symbol."""
    try:
        return _SYMBOL_TO_INDEX[symbol]
    except KeyError as error:
        raise ValueError(f"symbol outside the grid alphabet: {symbol!r}") from error
