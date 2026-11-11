from player_class import Player
import csv
from math import isfinite
def load_players(filename: str) -> list[Player]:
    players = []
    with open(filename, mode="r", newline="", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        #print(f"Detected CSV headers: {csv_reader.fieldnames!r}")
        required_columns = {"name", "positions", "projected_points", "schedule", "team"}
        actual_columns = set(csv_reader.fieldnames or [])
        missing_columns = required_columns - actual_columns
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        valid_positions = {"PG", "SG", "SF", "PF", "C"}
        VALID_NBA_TEAMS = {
            "ATL", "BOS", "BKN", "CHA", "CHI",
            "CLE", "DAL", "DEN", "DET", "GSW",
            "HOU", "IND", "LAC", "LAL", "MEM",
            "MIA", "MIL", "MIN", "NOP", "NYK",
            "OKC", "ORL", "PHI", "PHX", "POR",
            "SAC", "SAS", "TOR", "UTA", "WAS"}
        for row in csv_reader:
            if None in row or None in row.values():
                raise ValueError("Malformed CSV row")
            name = (row['name'] or "").strip()
            positions = [position.strip() for position in row["positions"].split(";")]
            if not name:
                raise ValueError("Missing player name")
            if not positions or not (set(positions).issubset(set(valid_positions))):
                raise ValueError("Invalid player positions")
            team = (row["team"] or "").strip().upper()
            if (team not in VALID_NBA_TEAMS):
                raise ValueError("Invalid NBA team")
            try:
                projected_points = float(row["projected_points"])
            except (ValueError, TypeError):
                raise ValueError("Invalid projected points") from None
            if not isfinite(projected_points):
                raise ValueError("Invalid projected points")
            schedule = [date.strip() for date in row["schedule"].split(";") if date.strip()]
            positions = [position.strip() for position in row["positions"].split(";")]
            players.append(Player(row['name'].strip(),positions,float(row['projected_points'].strip()),schedule,team))
    return players