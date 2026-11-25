from pathlib import Path
from fastapi import FastAPI, HTTPException
from fantasy_basketball.database import get_connection
from fantasy_basketball.inference import predict_stored_player_next_game, optimize_projected_day
from fantasy_basketball.schemas import (
    PredictionRequest,
    PredictionResponse,
    DailyLineupRequest,
    DailyLineupResponse,
    ProjectedPlayerResponse)
from fantasy_basketball.player_class import Player

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fantasy_basketball.db"
MODEL_DIR = PROJECT_ROOT / "models" / "32_players_3_features"
MODEL_CONFIGS = {
    "linear_regression": (MODEL_DIR / "linear_regression.pkl", "sklearn"),
    "random_forest": (MODEL_DIR / "random_forest.pkl", "sklearn"),
    "pytorch": (MODEL_DIR / "neural_network.pt", "pytorch")}

app = FastAPI(title="Fantasy Basketball Analyzer")
@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model_path, model_type = MODEL_CONFIGS[request.model]
    if not DB_PATH.is_file() or not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Prediction service data or model is unavailable")
    cutoff = request.as_of_date or request.game_date
    connection = get_connection(str(DB_PATH))
    try:
        player = connection.execute("SELECT player_id FROM players WHERE player_id = ?",(request.player_id,)).fetchone()
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
    if not DB_PATH.is_file() or not model_path.is_file():
        raise HTTPException(status_code=503,detail="Prediction service data or model is unavailable")
    cutoff = request.as_of_date or request.game_date
    players = []
    player_ids = {}
    for roster_player in request.players:
        player_ids[roster_player.name] = roster_player.player_id
        players.append(
            Player(
                name=roster_player.name,
                positions=roster_player.positions.copy(),
                projected_points=0.0,
                schedule=[day.isoformat() for day in roster_player.schedule],
                team=roster_player.team))
    connection = get_connection(str(DB_PATH))
    try:
        for roster_player in request.players:
            player = connection.execute("SELECT player_id FROM players WHERE player_id = ?",(roster_player.player_id,)).fetchone()
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
    response_lineup = {}
    for slot in request.lineup_slots:
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
        season=request.season,
        game_date=request.game_date,
        model=request.model,
        as_of_date=cutoff,
        lineup=response_lineup,
        projected_total=projected_total)