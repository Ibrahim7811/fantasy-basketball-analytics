import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

@pytest.mark.parametrize("use_overrides", [False, True])
def test_api_paths_follow_environment_configuration(tmp_path, use_overrides):
    project_root = Path(__file__).resolve().parent.parent
    environment = os.environ.copy()
    environment.pop("FANTASY_DB_PATH", None)
    environment.pop("FANTASY_MODEL_DIR", None)
    if use_overrides:
        expected_db = tmp_path / "custom_data" / "games.db"
        expected_models = tmp_path / "custom_models"
        environment["FANTASY_DB_PATH"] = str(expected_db)
        environment["FANTASY_MODEL_DIR"] = str(expected_models)
    else:
        expected_db = project_root / "data" / "fantasy_basketball.db"
        expected_models = project_root / "models" / "32_players_3_features"
    script = """
import json
from fantasy_basketball.api import DB_PATH, MODEL_DIR, MODEL_CONFIGS
print(json.dumps({
    "db_path": str(DB_PATH),
    "model_dir": str(MODEL_DIR),
    "models": {name: [str(path), model_type] for name, (path, model_type) in MODEL_CONFIGS.items()},
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=60)
    config = json.loads(result.stdout)
    assert config["db_path"] == str(expected_db.resolve())
    assert config["model_dir"] == str(expected_models.resolve())
    assert config["models"] == {
        "linear_regression": [str(expected_models.resolve() / "linear_regression.pkl"), "sklearn"],
        "random_forest": [str(expected_models.resolve() / "random_forest.pkl"), "sklearn"],
        "pytorch": [str(expected_models.resolve() / "neural_network.pt"), "pytorch"]}