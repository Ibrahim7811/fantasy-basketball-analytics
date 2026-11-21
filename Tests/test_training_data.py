import sqlite3
from unittest.mock import Mock
import pandas as pd
import pytest
from fantasy_basketball import training_data

WEIGHTS = {"PTS": 1.0, "TOV": -2.0}

def make_training_logs(player_id, year, points, team_id=1):
    count = len(points)

    return pd.DataFrame({
        "GAME_ID": [
            f"002{year % 100:02d}{i + 1:05d}"
            for i in range(count)
        ],
        "PLAYER_ID": [player_id] * count,
        "TEAM_ID": [team_id] * count,
        "GAME_DATE": pd.to_datetime([
            f"{year}-11-{1 + 2 * i:02d}"
            for i in range(count)
        ]),
        "PTS": points,
        "MIN": [float(player_id + 10 * (year - 2023))] * count,
        "TOV": [1] * count,
    })

@pytest.fixture
def connection():
    return Mock(spec=sqlite3.Connection)

@pytest.fixture
def histories(monkeypatch):
    stored = {
        (10, "2024-25"): make_training_logs(
            10, 2024, [12, 22, 32]
        ),
        (10, "2025-26"): make_training_logs(
            10, 2025, [42, 52, 62]
        ),
        (20, "2025-26"): make_training_logs(
            20, 2025, [72, 82, 92], team_id=2
        ),
        (30, "2025-26"): make_training_logs(
            30, 2025, [27]
        ),
    }

    def fake_load(_connection, player_id, season):
        return stored.get((player_id, season), pd.DataFrame()).copy(deep=True)
    monkeypatch.setattr(training_data, "load_player_game_logs", fake_load)
    return stored

def test_build_training_dataset_values_and_history_separation(connection, histories):
    result = training_data.build_training_dataset(
        connection,
        [20, 10],
        ["2025-26", "2024-25"],
        WEIGHTS,
    )
    expected = pd.DataFrame({
        "GAME_ID": [
            "0022400002", "0022400003",
            "0022500002", "0022500002",
            "0022500003", "0022500003",
        ],
        "PLAYER_ID": [10, 10, 10, 20, 10, 20],
        "TEAM_ID": [1, 1, 1, 2, 1, 2],
        "GAME_DATE": pd.to_datetime([
            "2024-11-03", "2024-11-05",
            "2025-11-03", "2025-11-03",
            "2025-11-05", "2025-11-05",
        ]),
        "SEASON": ["2024-25"] * 2 + ["2025-26"] * 4,
        "FP_AVG_5": [10.0, 15.0, 40.0, 70.0, 45.0, 75.0],
        "FP_AVG_10": [10.0, 15.0, 40.0, 70.0, 45.0, 75.0],
        "REST_DAYS": [1.0] * 6,
        "MIN_AVG_5": [20.0, 20.0, 30.0, 40.0, 30.0, 40.0],
        "MIN_AVG_10": [20.0, 20.0, 30.0, 40.0, 30.0, 40.0],
        "FANTASY_POINTS": [20.0, 30.0, 50.0, 80.0, 60.0, 90.0],
    })
    pd.testing.assert_frame_equal(result, expected)

def test_duplicate_and_missing_requests_do_not_change_dataset(connection, histories):
    expected = training_data.build_training_dataset(connection, [10, 20], ["2024-25", "2025-26"], WEIGHTS)
    result = training_data.build_training_dataset(
        connection,
        [20, 10, 20, 999],
        ["2025-26", "2024-25", "2025-26", "1999-00"],
        WEIGHTS,
    )
    pd.testing.assert_frame_equal(result, expected)

@pytest.mark.parametrize(
    "player_ids, seasons",
    [
        ([], ["2025-26"]),
        ([10], []),
        ([999], ["2025-26"]),
        ([10], ["1999-00"]),
    ],
)
def test_build_training_dataset_rejects_missing_logs(connection, histories, player_ids, seasons):
    with pytest.raises(ValueError, match="No game logs"):
        training_data.build_training_dataset(connection, player_ids, seasons, WEIGHTS)

def test_build_training_dataset_rejects_no_usable_rows(connection, histories):
    # Player 30 has one game, so all prior-history features are missing.
    with pytest.raises(ValueError, match="No usable training rows"):
        training_data.build_training_dataset(connection, [30], ["2025-26"], WEIGHTS)

def test_current_game_score_does_not_change_its_features(connection, histories):
    before = training_data.build_training_dataset(connection, [10, 20], ["2024-25", "2025-26"], WEIGHTS)

    # Change the final game's score for player 10.
    histories[(10, "2025-26")].loc[2, "PTS"] = 999

    after = training_data.build_training_dataset(connection, [10, 20], ["2024-25", "2025-26"], WEIGHTS)
    features = training_data.FEATURE_COLUMNS
    pd.testing.assert_frame_equal(before[features], after[features])
    changed_target = after.loc[
        (after["PLAYER_ID"] == 10)
        & (after["GAME_ID"] == "0022500003"),
        "FANTASY_POINTS",
    ].item()
    assert changed_target == pytest.approx(997.0)

@pytest.fixture
def split_dataset():
    df = pd.DataFrame(
        [
            ("g05", 20, "2025-11-05", 80.0),
            ("g02", 20, "2025-11-02", 20.0),
            ("g01", 20, "2025-11-01", 10.0),
            ("g04", 10, "2025-11-04", 60.0),
            ("g01", 10, "2025-11-01", 15.0),
            ("g03", 10, "2025-11-03", 40.0),
            ("g05", 10, "2025-11-05", 70.0),
        ],
        columns=[
            "GAME_ID",
            "PLAYER_ID",
            "GAME_DATE",
            "FANTASY_POINTS",
        ],
        index=range(10, 17),
    )
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

@pytest.mark.parametrize(
    "test_fraction, train_positions, test_positions",
    [
        (0.2, [4, 2, 1, 5, 3], [6, 0]),
        (0.4, [4, 2, 1, 5], [3, 6, 0]),
        (0.01, [4, 2, 1, 5, 3], [6, 0]),
        (0.99, [4, 2], [1, 5, 3, 6, 0]),
    ],
)
def test_split_training_dataset_chronological(split_dataset, test_fraction, train_positions, test_positions):
    train_df, test_df = training_data.split_training_dataset(split_dataset, test_fraction=test_fraction)
    expected_train = split_dataset.iloc[train_positions].reset_index(drop=True)
    expected_test = split_dataset.iloc[test_positions].reset_index(drop=True)

    pd.testing.assert_frame_equal(train_df, expected_train)
    pd.testing.assert_frame_equal(test_df, expected_test)

    assert train_df["GAME_DATE"].max() < test_df["GAME_DATE"].min()
    assert set(train_df["GAME_DATE"]).isdisjoint(test_df["GAME_DATE"])
    assert len(train_df) + len(test_df) == len(split_dataset)

def test_split_training_dataset_defaults_to_twenty_percent(split_dataset):
    train_df, test_df = training_data.split_training_dataset(split_dataset)
    assert train_df["GAME_DATE"].nunique() == 4
    assert test_df["GAME_DATE"].nunique() == 1

@pytest.mark.parametrize("test_fraction", [-0.1, 0, 1, 1.1, float("nan")])
def test_split_training_dataset_rejects_invalid_fraction(split_dataset, test_fraction):
    with pytest.raises(ValueError, match="test_fraction"):
        training_data.split_training_dataset(split_dataset, test_fraction=test_fraction)

def test_split_training_dataset_rejects_empty_input(split_dataset):
    empty_df = split_dataset.iloc[:0].copy()

    with pytest.raises(ValueError, match="cannot be empty"):
        training_data.split_training_dataset(empty_df)

def test_split_training_dataset_rejects_one_unique_date(split_dataset):
    one_date_df = split_dataset.loc[split_dataset["GAME_DATE"] == pd.Timestamp("2025-11-01")].copy()
    with pytest.raises(ValueError, match="At least two unique game dates"):
        training_data.split_training_dataset(one_date_df)

def test_split_training_dataset_preserves_input_and_returns_copies(split_dataset):
    original = split_dataset.copy(deep=True)
    train_df, test_df = training_data.split_training_dataset(split_dataset)
    pd.testing.assert_frame_equal(split_dataset, original)
    train_df.loc[0, "FANTASY_POINTS"] = -999.0
    test_df.loc[0, "FANTASY_POINTS"] = -888.0
    pd.testing.assert_frame_equal(split_dataset, original)

@pytest.mark.parametrize(
    "validation_fraction, train_positions, validation_positions",
    [
        (0.2, [4, 2, 1, 5], [3]),
        (0.4, [4, 2, 1], [5, 3]),
    ],
)
def test_three_way_split_values_and_order(split_dataset,validation_fraction,train_positions,validation_positions):
    train_df, validation_df, test_df = (training_data.split_train_validation_test(split_dataset,validation_fraction=validation_fraction,test_fraction=0.2,))
    expected_train = split_dataset.iloc[train_positions].reset_index(drop=True)
    expected_validation = split_dataset.iloc[validation_positions].reset_index(drop=True)
    expected_test = split_dataset.iloc[[6, 0]].reset_index(drop=True)
    pd.testing.assert_frame_equal(train_df, expected_train)
    pd.testing.assert_frame_equal(validation_df, expected_validation)
    pd.testing.assert_frame_equal(test_df, expected_test)
    assert train_df["GAME_DATE"].max() < validation_df["GAME_DATE"].min()
    assert validation_df["GAME_DATE"].max() < test_df["GAME_DATE"].min()
    assert (len(train_df) + len(validation_df) + len(test_df) == len(split_dataset))

def test_three_way_split_defaults(split_dataset):
    train_df, validation_df, test_df = (training_data.split_train_validation_test(split_dataset))
    assert train_df["GAME_DATE"].nunique() == 3
    assert validation_df["GAME_DATE"].nunique() == 1
    assert test_df["GAME_DATE"].nunique() == 1

@pytest.mark.parametrize(
    "argument",
    ["validation_fraction", "test_fraction"],
)
@pytest.mark.parametrize("value", [-0.1, 0, 1, 1.1, float("nan")])
def test_three_way_split_rejects_invalid_fractions(split_dataset, argument, value):
    with pytest.raises(ValueError, match=argument):
        training_data.split_train_validation_test(split_dataset, **{argument: value})

@pytest.mark.parametrize(
    "validation_fraction, test_fraction",
    [(0.5, 0.5), (0.6, 0.5)],
)
def test_three_way_split_rejects_invalid_sum(split_dataset, validation_fraction, test_fraction):
    with pytest.raises(ValueError, match="sum to less than 1"):
        training_data.split_train_validation_test(split_dataset,validation_fraction=validation_fraction,test_fraction=test_fraction)

def test_three_way_split_rejects_empty_input(split_dataset):
    with pytest.raises(ValueError, match="cannot be empty"):
        training_data.split_train_validation_test(split_dataset.iloc[:0].copy())

@pytest.mark.parametrize("date_count", [1, 2])
def test_three_way_split_rejects_insufficient_dates(split_dataset, date_count):
    dates = pd.date_range("2025-11-01", periods=date_count)
    small_df = split_dataset.loc[split_dataset["GAME_DATE"].isin(dates)].copy()
    with pytest.raises(ValueError, match="At least three unique game dates"):
        training_data.split_train_validation_test(small_df)

def test_three_way_split_preserves_input_and_returns_copies(split_dataset):
    original = split_dataset.copy(deep=True)
    splits = training_data.split_train_validation_test(split_dataset)
    pd.testing.assert_frame_equal(split_dataset, original)
    for frame in splits:
        frame.loc[0, "FANTASY_POINTS"] = -999.0
    pd.testing.assert_frame_equal(split_dataset, original)