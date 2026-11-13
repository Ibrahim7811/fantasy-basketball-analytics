class Player:
    def __init__(self, name : str, positions : list[str], projected_points : float, schedule : list[str], team : str):
        self._name = name
        self._positions = positions
        self._projected_points = projected_points
        self._schedule = schedule
        self._team = team

    def plays_on(self, date : str) -> bool:
        return date in self._schedule

    def eligible_for(self, position: str) -> bool:
        position = position.rstrip("0123456789")
        if (position == 'UTIL'): return True
        elif(position == 'G'):
            return 'PG' in self._positions or 'SG' in self._positions
        elif (position == 'F'):
            return 'PF' in self._positions or 'SF' in self._positions
        return position in self._positions
"""
nikola_jokic = Player('Nikola Jokic',['C'],58.5,["2026-11-03", "2026-11-05"],'DEN')
cade_cunningham = Player('Cade Cunningham',['PG','SG'],48.25,["2026-11-04", "2026-11-07"],'DET')

eligible_players = [player for player in players if player.plays_on(date) and player.eligible_for('C')]

highest_eligible_players = sorted_eligible_players[:2]


players = [
    Player("Nikola Jokic", ["C"], 58.5, ["2026-11-05"], "DEN"),
    Player("Victor Wembanyama", ["C", "PF"], 55.2, ["2026-11-05"], "SAS"),
    Player("Cade Cunningham", ["PG", "SG"], 48.25, ["2026-11-05"], "DET"),
    Player("Jalen Brunson", ["PG"], 46.8, ["2026-11-05"], "NYK"),
    Player("Anthony Edwards", ["SG", "SF"], 47.5, ["2026-11-05"], "MIN"),
    Player("Jaylen Brown", ["SG", "SF"], 43.2, ["2026-11-05"], "BOS"),
    Player("Paolo Banchero", ["PF", "SF"], 42.5, ["2026-11-05"], "ORL"),
    Player("Karl-Anthony Towns", ["C", "PF"], 44.3, ["2026-11-05"], "NYK"),
    Player("Jarrett Allen", ["C"], 35.8, ["2026-11-05"], "CLE"),
    Player("Derrick White", ["PG", "SG"], 36.4, ["2026-11-05"], "BOS"),
    Player("Mikal Bridges", ["SG", "SF"], 34.5, ["2026-11-05"], "NYK"),
    Player("Josh Hart", ["SG", "SF"], 33.7, ["2026-11-05"], "NYK"),
]
lineup_slots = [
    "PG", "SG", "G", 
    "SF", "PF", "F",
    "C1", "C2",
    "UTIL1", "UTIL2"
]

date = "2026-11-05"
sorted_eligible_players = sorted(players,key=lambda player: player._projected_points,reverse=True)
selected_players = set()
lineup = {}

for slot in lineup_slots:
    for player in sorted_eligible_players:
        if player.plays_on(date) and player not in selected_players and player.eligible_for(slot):
            lineup[slot] = player
            selected_players.add(player)
            break
for slot, player in lineup.items():
    print(f"{slot}: {player._name}")
"""