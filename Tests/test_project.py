
import sys
import pytest
from project import parse_arguments, main, parse_lineup_slots, validate_date_range


def test_parse_arguments(monkeypatch):
    # Simulate command-line arguments
    monkeypatch.setattr(sys, "argv", [
        "project.py",
        "--team-a", "team_a.csv",
        "--team-b", "team_b.csv",
        "--start", "2026-11-09",
        "--end", "2026-11-15"
    ])

    args = parse_arguments()

    assert args.team_a == "team_a.csv"
    assert args.team_b == "team_b.csv"
    assert args.start == "2026-11-09"
    assert args.end == "2026-11-15"


def test_missing_required_arguments(monkeypatch):
    # Omit the required --end argument
    monkeypatch.setattr(sys, "argv", [
        "project.py",
        "--team-a", "team_a.csv",
        "--team-b", "team_b.csv",
        "--start", "2026-11-09"
    ])

    with pytest.raises(SystemExit) as exc_info:
        parse_arguments()

    assert exc_info.value.code == 2


def test_main(tmp_path, monkeypatch, capsys):
    # Create temporary team CSV files
    team_a_file = tmp_path / "team_a.csv"
    team_b_file = tmp_path / "team_b.csv"

    team_a_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        "Player A,PG,50.0,2026-11-09,DEN\n",
        encoding="utf-8"
    )

    team_b_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        "Player B,PG,35.0,2026-11-09,BOS\n",
        encoding="utf-8"
    )

    # Simulate the CLI arguments
    monkeypatch.setattr(sys, "argv", [
        "project.py",
        "--team-a", str(team_a_file),
        "--team-b", str(team_b_file),
        "--start", "2026-11-09",
        "--end", "2026-11-09"
    ])

    # Execute the application
    main()

    # Capture terminal output
    output = capsys.readouterr().out

    # Verify the report
    assert "HEAD-TO-HEAD MATCHUP REPORT" in output
    assert "2026-11-09" in output
    assert "Weekly Total - Team A: 50.00" in output
    assert "Weekly Total - Team B: 35.00" in output
    assert "Projected Winner: Team A" in output
    assert "Point Differential: +15.00" in output

@pytest.mark.parametrize(
    "raw_slots, expected",
    [
        ("PG,SG,C1", ["PG", "SG", "C1"]),
        ("pg,sg,util1", ["PG", "SG", "UTIL1"]),
        ("PG, SG, UTIL1", ["PG", "SG", "UTIL1"]),
        ("C1,C2,G,F", ["C1", "C2", "G", "F"])
    ]
)
def test_parse_valid_lineup_slots(raw_slots, expected):
    slots = parse_lineup_slots(raw_slots)

    assert slots == expected

@pytest.mark.parametrize(
    "raw_slots",
    [
        "PG,XYZ,C1",
        "PG,,SG",
        "",
        "   "
    ]
)
def test_parse_invalid_lineup_slots(raw_slots):
    with pytest.raises(ValueError, match="Invalid lineup slots"):
        parse_lineup_slots(raw_slots)

@pytest.mark.parametrize(
    "raw_slots",
    [
        "PG,PG,SG",
        "C1,C2,C1",
        "PG,pg"
    ]
)
def test_parse_duplicate_lineup_slots(raw_slots):
    with pytest.raises(ValueError, match="Duplicate lineup slots"):
        parse_lineup_slots(raw_slots)

@pytest.mark.parametrize(
    "start_date, end_date, expected",
    [
        ("2026-11-09", "2026-11-15", True),
        ("2026-11-09", "2026-11-09", True),
        ("2026-11-15", "2026-11-09", False),
        ("2026-13-09", "2026-11-15", False),
        ("2026-02-30", "2026-03-05", False),
        ("20261109", "2026-11-15", False),
    ]
)
def test_validate_date_range(start_date, end_date, expected):
    assert validate_date_range(start_date, end_date) is expected

@pytest.mark.parametrize(
    "slots, expected_error",
    [
        ("PG,SG,XYZ", "Invalid lineup slots"),
        ("PG,PG,SG", "Duplicate lineup slots")
    ]
)
def test_main_invalid_slots(monkeypatch, capsys, slots, expected_error):
    monkeypatch.setattr(sys, "argv", [
        "project.py",
        "--team-a", "team_a.csv",
        "--team-b", "team_b.csv",
        "--start", "2026-11-09",
        "--end", "2026-11-15",
        "--slots", slots
    ])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2

    output = capsys.readouterr().out
    assert expected_error in output

def test_main_missing_csv(monkeypatch, capsys, tmp_path):
    missing_file = tmp_path / "nonexistent.csv"

    monkeypatch.setattr(sys, "argv", [
        "project.py",
        "--team-a", str(missing_file),
        "--team-b", str(tmp_path / "team_b.csv"),
        "--start", "2026-11-09",
        "--end", "2026-11-15"
    ])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2

    output = capsys.readouterr().out
    assert "Error loading roster" in output

def test_main_invalid_csv(monkeypatch, capsys, tmp_path):
    csv_file = tmp_path / "invalid.csv"

    csv_file.write_text(
        "name,positions,schedule,team\n"
        "Nikola Jokic,C,2026-11-09,DEN\n",
        encoding="utf-8"
    )

    monkeypatch.setattr(sys, "argv", [
        "project.py",
        "--team-a", str(csv_file),
        "--team-b", str(tmp_path / "team_b.csv"),
        "--start", "2026-11-09",
        "--end", "2026-11-15"
    ])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2

    output = capsys.readouterr().out
    assert "Invalid roster data" in output
    assert "projected_points" in output
