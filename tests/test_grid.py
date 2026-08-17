import pytest

from authorship.grid import (
    FLASH_CHANNELS,
    SYMBOLS,
    parse_symbol_channel,
    symbol_index,
    target_code_to_symbol,
)


def test_parse_symbol_channel():
    assert parse_symbol_channel("A_1_1") == ("A", 1, 1)
    assert parse_symbol_channel("Sp_5_3") == (" ", 5, 3)
    assert parse_symbol_channel("9_6_6") == ("9", 6, 6)


def test_alphabet_is_36_unique():
    assert len(FLASH_CHANNELS) == 36
    assert len(SYMBOLS) == 36
    assert len(set(SYMBOLS)) == 36


def test_grid_positions_cover_six_by_six():
    positions = {parse_symbol_channel(name)[1:] for name in FLASH_CHANNELS}
    assert positions == {(row, column) for row in range(1, 7) for column in range(1, 7)}


def test_target_code_maps_to_symbol():
    assert target_code_to_symbol(1) == "A"
    assert target_code_to_symbol(27) == " "
    assert target_code_to_symbol(36) == "9"


def test_target_code_rejects_out_of_range():
    with pytest.raises(ValueError):
        target_code_to_symbol(0)
    with pytest.raises(ValueError):
        target_code_to_symbol(37)


def test_symbol_index_round_trips():
    for code in range(1, 37):
        assert symbol_index(target_code_to_symbol(code)) == code - 1
