import pandas as pd
import pytest
from fantasy_basketball.data_processing import clean_game_logs, calculate_fantasy_points, add_rolling_averages, add_rest_days, prepare_game_logs, normalize_schedule

@pytest.fixture
def sample_logs():
    return pd.DataFrame(
        {
            "GAME_DATE": [
                "Nov 05, 2026",
                "Nov 01, 2026",
                "Nov 03, 2026",
            ],
            "PTS": [30, 10, 20],
        },
        index=[8, 3, 6],
    )

def test_converts_dates_to_datetime(sample_logs):
    result = clean_game_logs(sample_logs)

    assert pd.api.types.is_datetime64_any_dtype(result["GAME_DATE"])
    assert result["GAME_DATE"].tolist() == [
        pd.Timestamp("2026-11-01"),
        pd.Timestamp("2026-11-03"),
        pd.Timestamp("2026-11-05"),
    ]

def test_sorts_games_and_preserves_row_data(sample_logs):
    result = clean_game_logs(sample_logs)

    assert result["GAME_DATE"].is_monotonic_increasing
    assert result["PTS"].tolist() == [10, 20, 30]

def test_resets_index_without_adding_column(sample_logs):
    result = clean_game_logs(sample_logs)

    assert result.index.equals(pd.RangeIndex(3))
    assert result.columns.tolist() == sample_logs.columns.tolist()

def test_does_not_modify_original(sample_logs):
    original = sample_logs.copy(deep=True)

    result = clean_game_logs(sample_logs)

    pd.testing.assert_frame_equal(sample_logs, original)
    assert result is not sample_logs

@pytest.fixture
def scoring_logs():
    return pd.DataFrame({
        "PTS": [20, 10],
        "REB": [10, 5],
        "AST": [4, 2],
        "TOV": [3, 2],
    })

def test_calculates_fantasy_points(scoring_logs):
    weights = {"PTS": 1.0, "REB": 1.2, "AST": 1.5}

    result = calculate_fantasy_points(scoring_logs, weights)

    # Game 1: 20 + 12 + 6 = 38
    # Game 2: 10 + 6 + 3 = 19
    assert result["FANTASY_POINTS"].tolist() == pytest.approx([38.0, 19.0])

def test_negative_scoring_weights(scoring_logs):
    weights = {"PTS": 1.0, "TOV": -2.0}

    result = calculate_fantasy_points(scoring_logs, weights)

    # Game 1: 20 - 6 = 14
    # Game 2: 10 - 4 = 6
    assert result["FANTASY_POINTS"].tolist() == pytest.approx([14.0, 6.0])

def test_empty_scoring_weights(scoring_logs):
    result = calculate_fantasy_points(scoring_logs, {})

    assert result["FANTASY_POINTS"].tolist() == [0.0, 0.0]

def test_scoring_preserves_original_dataframe(scoring_logs):
    original = scoring_logs.copy(deep=True)

    result = calculate_fantasy_points(scoring_logs, {"PTS": 1.0})

    pd.testing.assert_frame_equal(scoring_logs, original)
    assert result is not scoring_logs

def test_rolling_averages_use_previous_games():
    df = pd.DataFrame({
        "FANTASY_POINTS": [20, 30, 40, 50, 60],
    })

    result = add_rolling_averages(df, windows=(3,))

    expected = pd.Series(
        [float("nan"), 20.0, 25.0, 30.0, 40.0],
        name="FP_AVG_3",
    )

    pd.testing.assert_series_equal(result["FP_AVG_3"], expected)

def test_default_rolling_windows():
    df = pd.DataFrame({
        "FANTASY_POINTS": list(range(10, 130, 10)),
    })

    result = add_rolling_averages(df)

    assert "FP_AVG_5" in result.columns
    assert "FP_AVG_10" in result.columns

    # Before game 12, the previous five scores are 70–110.
    assert result["FP_AVG_5"].iloc[-1] == pytest.approx(90.0)

    # Before game 12, the previous ten scores are 20–110.
    assert result["FP_AVG_10"].iloc[-1] == pytest.approx(65.0)

def test_rolling_averages_preserve_original():
    df = pd.DataFrame(
        {"FANTASY_POINTS": [20, 30, 40]},
        index=[8, 3, 6],
    )
    original = df.copy(deep=True)

    result = add_rolling_averages(df, windows=(3,))

    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(result[df.columns], original)
    assert result is not df

def test_calculates_rest_days():
    df = pd.DataFrame({
        "GAME_DATE": pd.to_datetime([
            "2026-11-01",
            "2026-11-02",
            "2026-11-05",
            "2026-11-07",
        ]),
    })

    result = add_rest_days(df)

    expected = pd.Series(
        [float("nan"), 0.0, 2.0, 1.0],
        name="REST_DAYS",
    )

    pd.testing.assert_series_equal(result["REST_DAYS"], expected)

def test_rest_days_preserve_original():
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime([
                "2026-11-01",
                "2026-11-03",
            ]),
            "FANTASY_POINTS": [20, 30],
        },
        index=[8, 3],
    )
    original = df.copy(deep=True)

    result = add_rest_days(df)

    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(result[df.columns], original)
    assert result is not df

def test_prepare_game_logs_pipeline():
    df = pd.DataFrame(
        {
            "GAME_DATE": [
                "Nov 07, 2026",
                "Nov 01, 2026",
                "Nov 05, 2026",
                "Nov 02, 2026",
            ],
            "PTS": [50, 20, 40, 30],
            "TOV": [4, 1, 3, 2],
            "MIN": [40.0, 10.0, 30.0, 20.0],
        },
        index=[8, 3, 6, 2],
    )
    original = df.copy(deep=True)

    result = prepare_game_logs(
        df,
        scoring_weights={"PTS": 1.0, "TOV": -1.0},
        windows=(2,),
    )

    nan = float("nan")
    expected = pd.DataFrame({
        "GAME_DATE": pd.to_datetime([
            "2026-11-01",
            "2026-11-02",
            "2026-11-05",
            "2026-11-07",
        ]),
        "PTS": [20, 30, 40, 50],
        "TOV": [1, 2, 3, 4],
        "MIN": [10.0, 20.0, 30.0, 40.0],
        "FANTASY_POINTS": [19.0, 28.0, 37.0, 46.0],
        "FP_AVG_2": [nan, 19.0, 23.5, 32.5],
        "MIN_AVG_2": [nan, 10.0, 15.0, 25.0],
        "REST_DAYS": [nan, 0.0, 2.0, 1.0],
    })

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(df, original)

@pytest.fixture
def raw_schedule():
    df = pd.DataFrame(
        [
            ("0022500003", "11/05/2025 00:00:00", 1, 2),
            ("0012500001", "10/02/2025 00:00:00", 1, 2),
            ("0022500002", "11/01/2025 00:00:00", 2, 1),
            ("0022500001", "11/01/2025 00:00:00", 1, 2),
            ("0022500004", "11/06/2025 00:00:00", 999, 2),
            ("0022500005", "11/07/2025 00:00:00", 1, 0),
        ],
        columns=[
            "gameId",
            "gameDate",
            "homeTeam_teamId",
            "awayTeam_teamId",
        ],
        index=[8, 3, 6, 2, 9, 5],
    )
    df["gameStatus"] = 3
    return df

@pytest.fixture
def schedule_team_lookup():
    return {"HOM": 1, "AWY": 2}

def test_normalize_schedule_filters_sorts_and_renames(raw_schedule, schedule_team_lookup):
    result = normalize_schedule(raw_schedule, schedule_team_lookup)

    expected = pd.DataFrame({
        "GAME_ID": ["0022500001", "0022500002", "0022500003"],
        "GAME_DATE": pd.to_datetime([
            "2025-11-01",
            "2025-11-01",
            "2025-11-05",
        ]),
        "HOME_TEAM_ID": [1, 2, 1],
        "AWAY_TEAM_ID": [2, 1, 2],
    })

    pd.testing.assert_frame_equal(result, expected)

@pytest.mark.parametrize("with_columns", [False, True])
def test_normalize_schedule_handles_empty_input(raw_schedule, schedule_team_lookup, with_columns):
    df = raw_schedule.iloc[:0] if with_columns else pd.DataFrame()

    result = normalize_schedule(df, schedule_team_lookup)

    assert result.empty
    assert result.columns.tolist() == [
        "GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID",
    ]
    assert pd.api.types.is_datetime64_any_dtype(result["GAME_DATE"])

def test_normalize_schedule_handles_no_matching_games(raw_schedule, schedule_team_lookup):
    # This fixture row is a preseason game.
    df = raw_schedule.iloc[[1]]

    result = normalize_schedule(df, schedule_team_lookup)

    assert result.empty
    assert result.columns.tolist() == [
        "GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID",
    ]
    assert pd.api.types.is_datetime64_any_dtype(result["GAME_DATE"])

def test_normalize_schedule_preserves_original(raw_schedule, schedule_team_lookup):
    original = raw_schedule.copy(deep=True)

    result = normalize_schedule(raw_schedule, schedule_team_lookup)

    pd.testing.assert_frame_equal(raw_schedule, original)
    assert result is not raw_schedule