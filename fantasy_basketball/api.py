from pathlib import Path
from fastapi import FastAPI, HTTPException
from fantasy_basketball.database import get_connection
from fantasy_basketball.postgres_database import get_postgres_connection
from fantasy_basketball.inference import predict_stored_player_next_game, optimize_projected_day, optimize_projected_week
from fantasy_basketball.schemas import (
    PredictionRequest,
    PredictionResponse,
    DailyLineupRequest,
    DailyLineupResponse,
    ProjectedPlayerResponse,
    WeeklyLineupRequest,
    WeeklyLineupResponse,
    RosterPlayerRequest,
    MatchupRequest,
    MatchupResponse)
from fantasy_basketball.player_class import Player
from datetime import date
from fantasy_basketball.matchup import analyze_projected_matchup
import os
import sqlite3
import psycopg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_URL = os.environ.get("FANTASY_DATABASE_URL", "").strip()
DB_PATH = Path(
    os.environ.get("FANTASY_DB_PATH", str(PROJECT_ROOT / "data" / "fantasy_basketball.db"))).expanduser().resolve()
MODEL_DIR = Path(
    os.environ.get("FANTASY_MODEL_DIR", str(PROJECT_ROOT / "models" / "32_players_3_features"))).expanduser().resolve()
MODEL_CONFIGS = {
    "linear_regression": (MODEL_DIR / "linear_regression.pkl", "sklearn"),
    "random_forest": (MODEL_DIR / "random_forest.pkl", "sklearn"),
    "pytorch": (MODEL_DIR / "neural_network.pt", "pytorch")}

def get_api_connection() -> sqlite3.Connection | psycopg.Connection:
    if DATABASE_URL:
        try:
            return get_postgres_connection(DATABASE_URL)
        except psycopg.OperationalError as error:
            raise HTTPException(status_code=503, detail="Database service is unavailable") from error
    return get_connection(str(DB_PATH))

def find_stored_player(connection: sqlite3.Connection | psycopg.Connection, player_id: int) -> tuple[int] | None:
    placeholder = "%s" if isinstance(connection, psycopg.Connection) else "?"
    query = f"SELECT player_id FROM players WHERE player_id = {placeholder}"
    return connection.execute(query, (player_id,)).fetchone()

def build_api_roster(roster: list[RosterPlayerRequest]) -> tuple[list[Player], dict[str, int]]:
    players = []
    player_ids = {}
    for roster_player in roster:
        player_ids[roster_player.name] = roster_player.player_id
        players.append(
            Player(
                name=roster_player.name,
                positions=roster_player.positions.copy(),
                projected_points=0.0,
                schedule=[day.isoformat() for day in roster_player.schedule],
                team=roster_player.team))
    return players, player_ids

def build_daily_lineup_response(
    lineup: dict[str, Player | None],
    player_ids: dict[str, int],
    lineup_slots: list[str],
    season: str,
    game_date: date,
    model: str,
    as_of_date: date,
    projected_total: float,
) -> DailyLineupResponse:
    response_lineup = {}
    for slot in lineup_slots:
        player = lineup.get(slot)
        if player is None:
            response_lineup[slot] = None
        else:
            response_lineup[slot] = ProjectedPlayerResponse(
                player_id=player_ids[player._name],
                name=player._name,
                positions=player._positions,
                team=player._team,
                projected_fp=player._projected_points)
    return DailyLineupResponse(
        season=season,
        game_date=game_date,
        model=model,
        as_of_date=as_of_date,
        lineup=response_lineup,
        projected_total=projected_total)

def build_weekly_lineup_response(
    weekly_results: dict[str, dict[str, object]],
    player_ids: dict[str, int],
    lineup_slots: list[str],
    season: str,
    start_date: date,
    end_date: date,
    model: str,
    as_of_date: date,
    projected_total: float,
) -> WeeklyLineupResponse:
    daily_results = []
    for game_date, result in sorted(weekly_results.items()):
        daily_results.append(
            build_daily_lineup_response(
                lineup=result["lineup"],
                player_ids=player_ids,
                lineup_slots=lineup_slots,
                season=season,
                game_date=date.fromisoformat(game_date),
                model=model,
                as_of_date=as_of_date,
                projected_total=result["projected_points"]))
    return WeeklyLineupResponse(
        season=season,
        start_date=start_date,
        end_date=end_date,
        model=model,
        as_of_date=as_of_date,
        daily_results=daily_results,
        projected_total=projected_total)

app = FastAPI(title="Fantasy Basketball Analyzer")

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model_path, model_type = MODEL_CONFIGS[request.model]
    if (not DATABASE_URL and not DB_PATH.is_file()) or not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Prediction service data or model is unavailable")
    cutoff = request.as_of_date or request.game_date
    connection = get_api_connection()
    try:
        player = find_stored_player(connection, request.player_id)
        if player is None:
            raise HTTPException(status_code=404, detail="Player not found")
        projection = predict_stored_player_next_game(
            connection,
            request.player_id,
            request.season,
            request.game_date.isoformat(),
            str(model_path),
            model_type,
            as_of_date=cutoff.isoformat())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        connection.close()
    return PredictionResponse(
        player_id=request.player_id,
        season=request.season,
        game_date=request.game_date,
        model=request.model,
        as_of_date=cutoff,
        projected_fp=projection)

@app.post("/lineups/daily", response_model=DailyLineupResponse)
def optimize_daily_lineup(request: DailyLineupRequest) -> DailyLineupResponse:
    model_path, model_type = MODEL_CONFIGS[request.model]
    if (not DATABASE_URL and not DB_PATH.is_file()) or not model_path.is_file():
        raise HTTPException(status_code=503, detail="Prediction service data or model is unavailable")
    cutoff = request.as_of_date or request.game_date
    players, player_ids = build_api_roster(request.players)
    connection = get_api_connection()
    try:
        for roster_player in request.players:
            player = find_stored_player(connection, roster_player.player_id)
            if player is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Player not found: {roster_player.player_id}")
        lineup, projected_total = optimize_projected_day(
            connection,
            players,
            player_ids,
            request.season,
            request.game_date.isoformat(),
            request.lineup_slots,
            str(model_path),
            model_type,
            as_of_date=cutoff.isoformat())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        connection.close()
    return build_daily_lineup_response(
        lineup=lineup,
        player_ids=player_ids,
        lineup_slots=request.lineup_slots,
        season=request.season,
        game_date=request.game_date,
        model=request.model,
        as_of_date=cutoff,
        projected_total=projected_total)

@app.post("/lineups/weekly", response_model=WeeklyLineupResponse)
def optimize_weekly_lineup(request: WeeklyLineupRequest) -> WeeklyLineupResponse:
    model_path, model_type = MODEL_CONFIGS[request.model]
    if (not DATABASE_URL and not DB_PATH.is_file()) or not model_path.is_file():
        raise HTTPException(status_code=503, detail="Prediction service data or model is unavailable")
    cutoff = request.as_of_date or request.start_date
    players, player_ids = build_api_roster(request.players)
    connection = get_api_connection()
    try:
        for roster_player in request.players:
            player = find_stored_player(connection, roster_player.player_id)
            if player is None:
                raise HTTPException(status_code=404, detail=f"Player not found: {roster_player.player_id}")
        weekly_results, projected_total = optimize_projected_week(
            connection,
            players,
            player_ids,
            request.season,
            request.start_date.isoformat(),
            request.end_date.isoformat(),
            request.lineup_slots,
            str(model_path),
            model_type,
            as_of_date=cutoff.isoformat())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        connection.close()
    return build_weekly_lineup_response(
        weekly_results=weekly_results,
        player_ids=player_ids,
        lineup_slots=request.lineup_slots,
        season=request.season,
        start_date=request.start_date,
        end_date=request.end_date,
        model=request.model,
        as_of_date=cutoff,
        projected_total=projected_total)

@app.post("/matchups", response_model=MatchupResponse)
def analyze_matchup_endpoint(request: MatchupRequest) -> MatchupResponse:
    model_path, model_type = MODEL_CONFIGS[request.model]
    if (not DATABASE_URL and not DB_PATH.is_file()) or not model_path.is_file():
        raise HTTPException(status_code=503, detail="Prediction service data or model is unavailable")
    cutoff = request.as_of_date or request.start_date
    team_a, team_a_ids = build_api_roster(request.team_a)
    team_b, team_b_ids = build_api_roster(request.team_b)
    player_ids = team_a_ids | team_b_ids
    connection = get_api_connection()
    try:
        for roster_player in request.team_a + request.team_b:
            player = find_stored_player(connection, roster_player.player_id)
            if player is None:
                raise HTTPException(status_code=404, detail=f"Player not found: {roster_player.player_id}")
        result = analyze_projected_matchup(
            connection,
            team_a,
            team_b,
            player_ids,
            request.season,
            request.start_date.isoformat(),
            request.end_date.isoformat(),
            request.lineup_slots,
            str(model_path),
            model_type,
            as_of_date=cutoff.isoformat())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        connection.close()
    response_a = build_weekly_lineup_response(
        weekly_results=result["team_a"]["weekly_results"],
        player_ids=team_a_ids,
        lineup_slots=request.lineup_slots,
        season=request.season,
        start_date=request.start_date,
        end_date=request.end_date,
        model=request.model,
        as_of_date=cutoff,
        projected_total=result["team_a"]["projected_total"])
    response_b = build_weekly_lineup_response(
        weekly_results=result["team_b"]["weekly_results"],
        player_ids=team_b_ids,
        lineup_slots=request.lineup_slots,
        season=request.season,
        start_date=request.start_date,
        end_date=request.end_date,
        model=request.model,
        as_of_date=cutoff,
        projected_total=result["team_b"]["projected_total"])
    return MatchupResponse(
        season=request.season,
        start_date=request.start_date,
        end_date=request.end_date,
        model=request.model,
        as_of_date=cutoff,
        team_a=response_a,
        team_b=response_b,
        projected_winner=result["projected_winner"],
        point_diff=result["point_diff"])