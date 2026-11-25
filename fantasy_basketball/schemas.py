from datetime import date
from typing import Literal, Self
from pydantic import BaseModel, Field, model_validator

class PredictionRequest(BaseModel):
    player_id: int = Field(gt=0)
    season: str = Field(pattern=r"^\d{4}-\d{2}$")
    game_date: date
    model: Literal["linear_regression", "random_forest", "pytorch"] = "linear_regression"
    as_of_date: date | None = None
    @model_validator(mode="after")
    def validate_date_order(self) -> Self:
        if self.as_of_date is not None and self.as_of_date > self.game_date:
            raise ValueError("as_of_date cannot be after game_date")
        return self

class PredictionResponse(BaseModel):
    player_id: int = Field(gt=0)
    season: str = Field(pattern=r"^\d{4}-\d{2}$")
    game_date: date
    model: Literal["linear_regression", "random_forest", "pytorch"]
    as_of_date: date
    projected_fp: float = Field(allow_inf_nan=False)