from player_class import Player

nikola_jokic = Player('Nikola Jokić',['C'],58.5,["2026-11-03", "2026-11-05"],'DEN')
cade_cunningham = Player('Cade Cunningham',['PG','SG'],48.25,["2026-11-04", "2026-11-07"],'DET')

def test_plays_on_game_day():
    assert nikola_jokic.plays_on("2026-11-03") == True
    assert cade_cunningham.plays_on("2026-11-07") == True

def test_plays_on_off_day():
    assert nikola_jokic.plays_on("2026-11-04") == False
    assert cade_cunningham.plays_on("2026-11-05") == False

def test_plays_on_empty_schedule():
    nikola_jokic = Player('Nikola Jokic',['C'],58.5,[],'DEN')
    cade_cunningham = Player('Cade Cunningham',['PG','SG'],48.25,[],'DET')
    assert nikola_jokic.plays_on("2026-11-04") == False
    assert cade_cunningham.plays_on("2026-11-07") == False

def test_eligible_positions():
    assert cade_cunningham.eligible_for("PG") == True
    assert cade_cunningham.eligible_for("G") == True
    assert cade_cunningham.eligible_for("UTIL") == True

def test_ineliglible_positions():
    assert not cade_cunningham.eligible_for("C")
    assert cade_cunningham.eligible_for("XYZ") == False

def test_numbered_positions():
    assert nikola_jokic.eligible_for("C1")
    assert nikola_jokic.eligible_for("C2")
    assert nikola_jokic.eligible_for("UTIL1")
    assert nikola_jokic.eligible_for("UTIL2")
    assert not nikola_jokic.eligible_for("G")
    assert not nikola_jokic.eligible_for("CXYZ")