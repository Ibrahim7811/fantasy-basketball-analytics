from pathlib import Path
from fastapi import FastAPI, HTTPException
from fantasy_basketball.database import get_connection
from fantasy_basketball.inference import predict_stored_player_next_game
from fantasy_basketball.schemas import PredictionRequest, PredictionResponse

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