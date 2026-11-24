from datetime import date
from typing import Literal
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    player_id: int = Field(gt=0)
    season: str = Field(pattern=r"^\d{4}-\d{2}$")
    game_date: date
    model: Literal["linear_regression", "random_forest", "pytorch"] = "linear_regression"
    as_of_date: date | None = None