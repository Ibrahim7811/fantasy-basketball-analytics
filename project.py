import argparse
from datetime import date
from data_loader import load_players
from matchup import analyze_matchup
from report import format_matchup_report

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fantasy Basketball Head-to-Head Matchup Analyzer")
    parser.add_argument("--team-a",type=str,required=True,help="Path to Team A's CSV roster")
    parser.add_argument("--team-b",type=str,required=True,help="Path to Team B's CSV roster")
    parser.add_argument("--start",type=str,required=True,help="Matchup start date (YYYY-MM-DD)")
    parser.add_argument("--end",type=str,required=True,help="Matchup end date (YYYY-MM-DD)")
    parser.add_argument("--slots",type=str,default="PG,SG,G,SF,PF,F,C1,C2,UTIL1,UTIL2",help="Comma separated lineup positions")
    return parser.parse_args()

def parse_lineup_slots(raw_slots: str) -> list[str]:
    slots = raw_slots.split(',')
    for slot in range(len(slots)):
        slots[slot] = slots[slot].strip().upper()
    valid_slots = {
    "PG", "SG", "G",
    "SF", "PF", "F",
    "C1", "C2",
    "UTIL1", "UTIL2"}
    if not set(slots).issubset(set(valid_slots)):
        raise ValueError("Invalid lineup slots")
    if len(set(slots)) != len(slots):
        raise ValueError("Duplicate lineup slots")
    return slots

def validate_date_range(start_date: str, end_date: str) -> bool:
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        if start.isoformat() != start_date or end.isoformat() != end_date:
            raise ValueError("Invalid date format")
        if start > end:
            print("Start date must not be after end date")
            return False
        return True
    except ValueError:
        print("Invalid date format (YYYY-MM-DD)")
        return False

def main():
    args = parse_arguments()
    if not validate_date_range(args.start, args.end):
        raise SystemExit(2)
    try:
        lineup_slots = parse_lineup_slots(args.slots)
    except ValueError as error:
        print(f"Invalid lineup configuration: {error}")
        raise SystemExit(2)
    try:
        team_a_players = load_players(args.team_a)
        team_b_players = load_players(args.team_b)
    except FileNotFoundError as error:
        print(f"Error loading roster: {error}")
        raise SystemExit(2)
    except ValueError as error:
        print(f"Invalid roster data: {error}")
        raise SystemExit(2)
    matchup_results = analyze_matchup(team_a_players,team_b_players,args.start,args.end,lineup_slots)
    report = format_matchup_report(matchup_results)
    print(report)

if __name__ == "__main__":
    main()