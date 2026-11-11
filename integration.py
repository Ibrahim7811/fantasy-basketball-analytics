from data_loader import load_players
from lineup import build_optimal_lineup, lineup_total

def main():
    date = "2026-11-09"
    lineup_slots = [
        "PG", "SG", "G",
        "SF", "PF", "F",
        "C1", "C2",
        "UTIL1", "UTIL2"]
    players = load_players('players.csv')

    optimal_lineup = build_optimal_lineup(players,date,lineup_slots)
    total = lineup_total(optimal_lineup)
    for slot, player in optimal_lineup.items():
        print(f"{slot}: {player._name} ({player._projected_points} pts)")
    print(f"Total projected points: {total:.2f}")

if __name__ == "__main__":
    main()