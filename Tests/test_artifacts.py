import hashlib
import pytest
from fantasy_basketball.artifacts import build_model_manifest, package_models
import json
from zipfile import ZipFile

@pytest.fixture
def model_files(tmp_path):
    contents = {
        "linear_regression.pkl": b"linear model",
        "random_forest.pkl": b"forest model",
        "neural_network.pt": b"neural model"}
    for filename, content in contents.items():
        (tmp_path / filename).write_bytes(content)
    return tmp_path, contents

def test_build_model_manifest_records_sizes_and_checksums(model_files):
    directory, contents = model_files
    manifest = build_model_manifest(str(directory))
    assert manifest["format_version"] == 1
    assert set(manifest["files"]) == set(contents)
    for filename, content in contents.items():
        assert manifest["files"][filename] == {"size_bytes": len(content),"sha256": hashlib.sha256(content).hexdigest()}

def test_build_model_manifest_does_not_change_files(model_files):
    directory, contents = model_files
    original_names = {path.name for path in directory.iterdir()}
    build_model_manifest(str(directory))
    assert {path.name for path in directory.iterdir()} == original_names
    for filename, content in contents.items():
        assert (directory / filename).read_bytes() == content

def test_build_model_manifest_is_repeatable(model_files):
    directory, _ = model_files
    first = build_model_manifest(str(directory))
    second = build_model_manifest(str(directory))
    assert first == second

def test_build_model_manifest_detects_changed_contents(model_files):
    directory, _ = model_files
    first = build_model_manifest(str(directory))
    (directory / "linear_regression.pkl").write_bytes(b"LINEAR MODEL")
    second = build_model_manifest(str(directory))
    assert first["files"]["linear_regression.pkl"]["sha256"] != second["files"]["linear_regression.pkl"]["sha256"]
    assert first["files"]["random_forest.pkl"] == second["files"]["random_forest.pkl"]
    assert first["files"]["neural_network.pt"] == second["files"]["neural_network.pt"]

@pytest.mark.parametrize("filename", ["linear_regression.pkl", "random_forest.pkl", "neural_network.pt"])
def test_build_model_manifest_rejects_missing_file(model_files, filename):
    directory, _ = model_files
    (directory / filename).unlink()
    with pytest.raises(FileNotFoundError, match = filename.replace(".", r"\.")):
        build_model_manifest(str(directory))

@pytest.mark.parametrize("filename", ["linear_regression.pkl", "random_forest.pkl", "neural_network.pt"])
def test_build_model_manifest_rejects_empty_file(model_files, filename):
    directory, _ = model_files
    (directory / filename).write_bytes(b"")
    with pytest.raises(ValueError, match = "empty"):
        build_model_manifest(str(directory))

def test_build_model_manifest_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match = "Model file is missing"):
        build_model_manifest(str(tmp_path / "missing"))

def test_build_model_manifest_ignores_unrelated_files(model_files):
    directory, contents = model_files
    (directory / "notes.txt").write_text("Not a model", encoding = "utf-8")
    manifest = build_model_manifest(str(directory))
    assert set(manifest["files"]) == set(contents)

def test_package_models_contains_models_and_manifest(model_files, tmp_path):
    directory, contents = model_files
    destination = tmp_path / "release" / "models.zip"
    result = package_models(str(directory), str(destination))
    assert result == str(destination)
    assert destination.is_file()
    with ZipFile(destination) as archive:
        assert len(archive.namelist()) == 4
        assert set(archive.namelist()) == {*contents, "manifest.json"}
        assert archive.testzip() is None
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format_version"] == 1
        assert set(manifest["files"]) == set(contents)
        for filename, original_content in contents.items():
            packaged_content = archive.read(filename)
            assert packaged_content == original_content
            assert manifest["files"][filename] == {
                "size_bytes": len(packaged_content),
                "sha256": hashlib.sha256(packaged_content).hexdigest()}

def test_package_models_preserves_source_files(model_files, tmp_path):
    directory, contents = model_files
    package_models(str(directory), str(tmp_path / "models.zip"))
    for filename, content in contents.items():
        assert (directory / filename).read_bytes() == content

def test_package_models_refuses_to_overwrite_existing_file(model_files, tmp_path):
    directory, _ = model_files
    destination = tmp_path / "models.zip"
    destination.write_bytes(b"existing package")
    with pytest.raises(FileExistsError, match="already exists"):
        package_models(str(directory), str(destination))
    assert destination.read_bytes() == b"existing package"

def test_package_models_excludes_unrelated_files(model_files, tmp_path):
    directory, contents = model_files
    (directory / "notes.txt").write_text("Not part of the release", encoding = "utf-8")
    destination = tmp_path / "models.zip"
    package_models(str(directory), str(destination))
    with ZipFile(destination) as archive:
        assert set(archive.namelist()) == {*contents, "manifest.json"}

@pytest.mark.parametrize("filename", ["linear_regression.pkl", "random_forest.pkl", "neural_network.pt"])
def test_package_models_rejects_missing_models(model_files, tmp_path, filename):
    directory, _ = model_files
    (directory / filename).unlink()
    destination = tmp_path / "release" / "models.zip"
    with pytest.raises(FileNotFoundError, match="Model file is missing"):
        package_models(str(directory), str(destination))
    assert not destination.exists()
    assert not destination.parent.exists()

@pytest.mark.parametrize("filename", ["linear_regression.pkl", "random_forest.pkl", "neural_network.pt"])
def test_package_models_rejects_empty_models(model_files, tmp_path, filename):
    directory, _ = model_files
    (directory / filename).write_bytes(b"")
    destination = tmp_path / "release" / "models.zip"
    with pytest.raises(ValueError, match = "Model file is empty"):
        package_models(str(directory), str(destination))
    assert not destination.exists()

def test_package_models_rejects_changes_after_manifest_creation(model_files, tmp_path, monkeypatch):
    directory, _ = model_files
    manifest = build_model_manifest(str(directory))
    (directory / "linear_regression.pkl").write_bytes(b"changed model")
    monkeypatch.setattr("fantasy_basketball.artifacts.build_model_manifest", lambda _: manifest)
    destination = tmp_path / "release" / "models.zip"
    with pytest.raises(ValueError, match="Model changed during packaging"):
        package_models(str(directory), str(destination))
    assert not destination.exists()