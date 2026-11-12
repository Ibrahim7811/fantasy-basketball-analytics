import pytest
from nba_data import get_player_id

def test_get_player_id():
    player_id = get_player_id("Nikola Jokic")

    assert player_id == 203999
    assert isinstance(player_id, int)

def test_invalid_player_name():
    with pytest.raises(ValueError, match="Player not found"):
        get_player_id("Nonexistent Basketball Player")

def test_player_name_case_insensitive():
    player_id = get_player_id("nikola jokic")

    assert player_id == 203999
