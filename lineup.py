from player_class import Player

def build_greedy_lineup(players: list[Player], date: str, lineup_slots: list[str]) -> dict:
    sorted_eligible_players = sorted(players,key=lambda player: player._projected_points,reverse=True)
    selected_players = set()
    lineup = {}
    for slot in lineup_slots:
        for player in sorted_eligible_players:
            if player.plays_on(date) and player not in selected_players and player.eligible_for(slot):
                lineup[slot] = player
                selected_players.add(player)
                break
    return lineup

def lineup_total(lineup: dict) -> float:
    return sum(
        (player._projected_points for player in lineup.values()),
        0.0)

def build_optimal_lineup(players: list[Player],date: str,lineup_slots: list[str]) -> dict:
    nodes_visited = 0
    has_util = any(slot.startswith('UTIL') for slot in lineup_slots)
    current_lineup = {}
    selected_players = set()
    best_lineup = {}
    best_score = float("-inf")
    active_players = [player for player in players if player.plays_on(date)]
    if (not has_util):
        active_players = [player for player in active_players if any(player.eligible_for(slot) for slot in lineup_slots)]
    active_players = sorted(active_players, key = lambda player: player._projected_points, reverse = True)
    eligible_by_slot = {slot: [player for player in active_players if player.eligible_for(slot)] for slot in lineup_slots}

    def backtrack(slot_index):
        nonlocal best_score, best_lineup, nodes_visited
        nodes_visited += 1
        if slot_index == len(lineup_slots):
            if ((current_score := lineup_total(current_lineup)) > best_score):
                best_score = current_score
                best_lineup = current_lineup.copy()
            return
        remaining_slots = len(lineup_slots) - slot_index
        remaining_players = [player for player in active_players if player not in selected_players]
        remaining_top_players = remaining_players[:remaining_slots]
        max_additional_points = sum(max(player._projected_points, 0) for player in remaining_top_players)
        upper_bound = lineup_total(current_lineup) + max_additional_points
        if (best_score >= upper_bound):
            return
        current_slot = lineup_slots[slot_index]
        for player in eligible_by_slot[current_slot]:
            if (player not in selected_players):
                current_lineup[current_slot] = player
                selected_players.add(player)
                backtrack(slot_index + 1)
                selected_players.remove(player)
                del current_lineup[current_slot]
        backtrack(slot_index + 1)
    backtrack(0)
    #print(f"Nodes visited: {nodes_visited}")
    return best_lineup