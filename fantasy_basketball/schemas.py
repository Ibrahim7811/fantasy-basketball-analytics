from datetime import date
from typing import Literal, Self
from pydantic import BaseModel, Field, field_validator, model_validator

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

class RosterPlayerRequest(BaseModel):
    player_id: int = Field(gt=0)
    name: str = Field(min_length=1)
    positions: list[Literal["PG", "SG", "SF", "PF", "C"]] = Field(min_length=1)
    team: str = Field(pattern=r"^[A-Z]{3}$")
    schedule: list[date] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Player name cannot be blank")
        return value

    @field_validator("positions")
    @classmethod
    def validate_positions(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Duplicate player positions")
        return value

    @field_validator("schedule")
    @classmethod
    def validate_schedule(cls, value: list[date]) -> list[date]:
        if len(value) != len(set(value)):
            raise ValueError("Duplicate schedule dates")
        return sorted(value)

    @field_validator("team")
    @classmethod
    def validate_team(cls, value: str) -> str:
        valid_teams = {
            "ATL", "BOS", "BKN", "CHA", "CHI",
            "CLE", "DAL", "DEN", "DET", "GSW",
            "HOU", "IND", "LAC", "LAL", "MEM",
            "MIA", "MIL", "MIN", "NOP", "NYK",
            "OKC", "ORL", "PHI", "PHX", "POR",
            "SAC", "SAS", "TOR", "UTA", "WAS"}
        if value not in valid_teams:
            raise ValueError("Unknown NBA team abbreviation")
        return value

class DailyLineupRequest(BaseModel):
    players: list[RosterPlayerRequest] = Field(min_length=1)
    season: str = Field(pattern=r"^\d{4}-\d{2}$")
    game_date: date
    lineup_slots: list[Literal["PG", "SG", "G", "SF", "PF", "F", "C1", "C2", "UTIL1", "UTIL2"]] = Field(min_length=1)
    model: Literal["linear_regression", "random_forest", "pytorch"] = "linear_regression"
    as_of_date: date | None = None

    @field_validator("lineup_slots")
    @classmethod
    def validate_lineup_slots(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Duplicate lineup slots")
        return value

    @model_validator(mode="after")
    def validate_roster_and_dates(self) -> Self:
        player_ids = [player.player_id for player in self.players]
        names = [player.name.casefold() for player in self.players]
        if len(player_ids) != len(set(player_ids)):
            raise ValueError("Duplicate player IDs")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate player names")
        if self.as_of_date is not None and self.as_of_date > self.game_date:
            raise ValueError("as_of_date cannot be after game_date")
        return self

class ProjectedPlayerResponse(BaseModel):
    player_id: int = Field(gt=0)
    name: str
    positions: list[Literal["PG", "SG", "SF", "PF", "C"]]
    team: str
    projected_fp: float = Field(allow_inf_nan=False)

class DailyLineupResponse(BaseModel):
    season: str
    game_date: date
    model: Literal["linear_regression", "random_forest", "pytorch"]
    as_of_date: date
    lineup: dict[str, ProjectedPlayerResponse | None]
    projected_total: float = Field(allow_inf_nan=False)

class WeeklyLineupRequest(BaseModel):
    players: list[RosterPlayerRequest] = Field(min_length=1)
    season: str = Field(pattern=r"^\d{4}-\d{2}$")
    start_date: date
    end_date: date
    lineup_slots: list[Literal["PG", "SG", "G", "SF", "PF", "F", "C1", "C2", "UTIL1", "UTIL2"]] = Field(min_length=1)
    model: Literal["linear_regression", "random_forest", "pytorch"] = "linear_regression"
    as_of_date: date | None = None

    @field_validator("lineup_slots")
    @classmethod
    def validate_lineup_slots(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Duplicate lineup slots")
        return value

    @model_validator(mode="after")
    def validate_roster_and_dates(self) -> Self:
        player_ids = [player.player_id for player in self.players]
        names = [player.name.casefold() for player in self.players]
        if len(player_ids) != len(set(player_ids)):
            raise ValueError("Duplicate player IDs")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate player names")
        if self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        if self.as_of_date is not None and self.as_of_date > self.start_date:
            raise ValueError("as_of_date cannot be after start_date")
        return self

class WeeklyLineupResponse(BaseModel):
    season: str
    start_date: date
    end_date: date
    model: Literal["linear_regression", "random_forest", "pytorch"]
    as_of_date: date
    daily_results: list[DailyLineupResponse]
    projected_total: float = Field(allow_inf_nan=False)

class MatchupRequest(BaseModel):
    team_a: list[RosterPlayerRequest] = Field(min_length=1)
    team_b: list[RosterPlayerRequest] = Field(min_length=1)
    season: str = Field(pattern=r"^\d{4}-\d{2}$")
    start_date: date
    end_date: date
    lineup_slots: list[Literal["PG", "SG", "G", "SF", "PF", "F", "C1", "C2", "UTIL1", "UTIL2"]] = Field(min_length=1)
    model: Literal["linear_regression", "random_forest", "pytorch"] = "linear_regression"
    as_of_date: date | None = None

    @field_validator("lineup_slots")
    @classmethod
    def validate_lineup_slots(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Duplicate lineup slots")
        return value

    @model_validator(mode="after")
    def validate_rosters_and_dates(self) -> Self:
        players = self.team_a + self.team_b
        player_ids = [player.player_id for player in players]
        names = [player.name.casefold() for player in players]
        if len(player_ids) != len(set(player_ids)):
            raise ValueError("Duplicate player IDs in matchup rosters")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate player names in matchup rosters")
        if self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        if self.as_of_date is not None and self.as_of_date > self.start_date:
            raise ValueError("as_of_date cannot be after start_date")
        return self

class MatchupResponse(BaseModel):
    season: str
    start_date: date
    end_date: date
    model: Literal["linear_regression", "random_forest", "pytorch"]
    as_of_date: date
    team_a: WeeklyLineupResponse
    team_b: WeeklyLineupResponse
    projected_winner: Literal["Team A", "Team B", "Tie"]
    point_diff: float = Field(allow_inf_nan=False)