import hashlib
import pytest
from fantasy_basketball.artifacts import build_model_manifest, package_models, install_models
import json
from zipfile import ZipFile, BadZipFile

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

@pytest.fixture
def model_package(model_files, tmp_path):
    directory, contents = model_files
    package_path = tmp_path / "models.zip"
    package_models(str(directory), str(package_path))
    return package_path, contents

def test_install_models_verifies_and_installs_package(model_package, tmp_path):
    package_path, contents = model_package
    destination = tmp_path / "installed" / "models"
    result = install_models(str(package_path), str(destination))
    assert result == str(destination)
    assert {path.name for path in destination.iterdir()} == {*contents, "manifest.json"}
    manifest = json.loads((destination / "manifest.json").read_text(encoding = "utf-8"))
    for filename, content in contents.items():
        installed = (destination / filename).read_bytes()
        assert installed == content
        assert manifest["files"][filename]["size_bytes"] == len(installed)
        assert manifest["files"][filename]["sha256"] == hashlib.sha256(installed).hexdigest()

def test_install_models_preserves_package(model_package, tmp_path):
    package_path, _ = model_package
    original = package_path.read_bytes()
    install_models(str(package_path), str(tmp_path / "installed"))
    assert package_path.read_bytes() == original

def test_install_models_refuses_existing_destination(model_package, tmp_path):
    package_path, _ = model_package
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("preserve", encoding = "utf-8")
    with pytest.raises(FileExistsError, match = "already exists"):
        install_models(str(package_path), str(destination))
    assert marker.read_text(encoding = "utf-8") == "preserve"
    assert {path.name for path in destination.iterdir()} == {"keep.txt"}

@pytest.mark.parametrize("case", ["changed_content", "wrong_size", "wrong_checksum", "empty_model"])
def test_install_models_rejects_invalid_model_contents(model_package, tmp_path, case):
    package_path, _ = model_package
    with ZipFile(package_path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    filename = "linear_regression.pkl"
    if case == "changed_content":
        entries[filename] = b"changed model"
    elif case == "wrong_size":
        manifest["files"][filename]["size_bytes"] += 1
    elif case == "wrong_checksum":
        manifest["files"][filename]["sha256"] = "0" * 64
    else:
        entries[filename] = b""
    entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
    invalid_package = tmp_path / "invalid.zip"
    with ZipFile(invalid_package, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    destination = tmp_path / "release" / "models"
    with pytest.raises(ValueError, match = "verification failed|empty"):
        install_models(str(invalid_package), str(destination))
    assert not destination.exists()
    assert not destination.parent.exists()

@pytest.mark.parametrize("case", ["missing_model", "missing_manifest", "extra_file", "unsafe_path"])
def test_install_models_rejects_unexpected_archive_entries(model_package, tmp_path, case):
    package_path, _ = model_package
    with ZipFile(package_path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    if case == "missing_model":
        del entries["linear_regression.pkl"]
    elif case == "missing_manifest":
        del entries["manifest.json"]
    elif case == "extra_file":
        entries["notes.txt"] = b"extra"
    else:
        entries["../escaped.txt"] = entries.pop("linear_regression.pkl")
    invalid_package = tmp_path / "invalid.zip"
    with ZipFile(invalid_package, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    destination = tmp_path / "release" / "models"
    with pytest.raises(ValueError, match = "exactly three models and a manifest"):
        install_models(str(invalid_package), str(destination))
    assert not destination.exists()
    assert not (tmp_path / "escaped.txt").exists()

@pytest.mark.parametrize("case", ["invalid_json", "not_dictionary", "unsupported_version", "missing_files", "wrong_filenames", "invalid_entry"])
def test_install_models_rejects_invalid_manifest(model_package, tmp_path, case):
    package_path, _ = model_package
    with ZipFile(package_path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    if case == "invalid_json":
        entries["manifest.json"] = b"not JSON"
    else:
        if case == "not_dictionary":
            manifest = []
        elif case == "unsupported_version":
            manifest["format_version"] = 99
        elif case == "missing_files":
            del manifest["files"]
        elif case == "wrong_filenames":
            manifest["files"]["unexpected.pkl"] = manifest["files"].pop("linear_regression.pkl")
        else:
            manifest["files"]["linear_regression.pkl"] = "invalid"
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
    invalid_package = tmp_path / "invalid.zip"
    with ZipFile(invalid_package, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    destination = tmp_path / "release" / "models"
    with pytest.raises(ValueError):
        install_models(str(invalid_package), str(destination))
    assert not destination.exists()
    assert not destination.parent.exists()

def test_install_models_rejects_non_zip_file(tmp_path):
    package_path = tmp_path / "invalid.zip"
    package_path.write_bytes(b"not a ZIP archive")
    destination = tmp_path / "installed"
    with pytest.raises(BadZipFile):
        install_models(str(package_path), str(destination))
    assert not destination.exists()

def test_install_models_cleans_up_after_write_failure(model_package, tmp_path, monkeypatch):
    package_path, _ = model_package
    destination = tmp_path / "release" / "models"
    def fail_write(self, content):
        raise OSError("Simulated write failure")
    monkeypatch.setattr(type(destination), "write_bytes", fail_write)
    with pytest.raises(OSError, match = "Simulated write failure"):
        install_models(str(package_path), str(destination))
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []