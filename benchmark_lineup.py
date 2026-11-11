import time
from player_class import Player
from lineup import build_optimal_lineup, lineup_total

if __name__ == "__main__":

    players = [
            Player("Nikola Jokic", ["C"], 58.5, ["2026-11-05"], "DEN"),
            Player("Victor Wembanyama", ["C", "PF"], 55.2, ["2026-11-05"], "SAS"),
            Player("Cade Cunningham", ["PG", "SG"], 48.25, ["2026-11-05"], "DET"),
            Player("Jalen Brunson", ["PG"], 46.8, ["2026-11-05"], "NYK"),
            Player("Anthony Edwards", ["SG", "SF"], 47.5, ["2026-11-05"], "MIN"),
            Player("Paolo Banchero", ["PF", "SF"], 42.5, ["2026-11-05"], "ORL"),
            Player("Jaylen Brown", ["SG", "SF"], 43.2, ["2026-11-05"], "BOS"),
            Player("Karl-Anthony Towns", ["C", "PF"], 44.3, ["2026-11-05"], "NYK"),
            Player("Jarrett Allen", ["C"], 35.8, ["2026-11-05"], "CLE"),
            Player("Derrick White", ["PG", "SG"], 36.4, ["2026-11-05"], "BOS"),
            Player("Mikal Bridges", ["SG", "SF"], 34.5, ["2026-11-05"], "NYK"),
            Player("Josh Hart", ["SG", "SF"], 33.7, ["2026-11-05"], "NYK")]
    slots = [
    "PG", "SG", "G",
    "SF", "PF", "F",
    "C1", "C2",
    "UTIL1", "UTIL2"]

    start = time.perf_counter()

    optimal_lineup = build_optimal_lineup(players, "2026-11-05", slots)

    elapsed_time = time.perf_counter() - start

    print(f"Players: {len(players)}")
    print(f"Starting slots: {len(slots)}")
    print(f"Assigned players: {len(optimal_lineup)}")
    print(f"Optimal projection: {lineup_total(optimal_lineup):.2f}")
    print(f"Execution time: {elapsed_time:.8f} seconds")