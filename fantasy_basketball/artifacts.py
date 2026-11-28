import hashlib
from pathlib import Path
import json
from zipfile import ZIP_DEFLATED, ZipFile
from tempfile import TemporaryDirectory

def build_model_manifest(model_dir: str) -> dict[str, object]:
    directory = Path(model_dir)
    filenames = ["linear_regression.pkl","random_forest.pkl","neural_network.pt"]
    files = {}
    for filename in filenames:
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Model file is missing: {filename}")
        size_bytes = path.stat().st_size
        if size_bytes == 0:
            raise ValueError(f"Model file is empty: {filename}")
        checksum = hashlib.sha256()
        with path.open("rb") as model_file:
            while chunk := model_file.read(1024 * 1024):
                checksum.update(chunk)
        files[filename] = {"size_bytes": size_bytes,"sha256": checksum.hexdigest()}
    return {"format_version": 1,"files": files}

def package_models(model_dir: str, output_path: str) -> str:
    directory = Path(model_dir)
    destination = Path(output_path)
    manifest = build_model_manifest(model_dir)
    if destination.exists():
        raise FileExistsError(f"Package already exists: {destination}")
    model_contents = {}
    for filename, details in manifest["files"].items():
        content = (directory / filename).read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        if len(content) != details["size_bytes"] or checksum != details["sha256"]:
            raise ValueError(f"Model changed during packaging: {filename}")
        model_contents[filename] = content
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, mode="x", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent = 2, sort_keys = True))
        for filename, content in model_contents.items():
            archive.writestr(filename, content)
    return str(destination)

def install_models(package_path: str, output_dir: str) -> str:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    filenames = {"linear_regression.pkl", "random_forest.pkl", "neural_network.pt"}
    model_contents = {}
    with ZipFile(package_path) as archive:
        expected_names = filenames | {"manifest.json"}
        if len(archive.namelist()) != 4 or set(archive.namelist()) != expected_names:
            raise ValueError("Package must contain exactly three models and a manifest")
        manifest_content = archive.read("manifest.json")
        manifest = json.loads(manifest_content)
        if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
            raise ValueError("Unsupported model manifest")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != filenames:
            raise ValueError("Manifest must describe exactly three models")
        for filename in sorted(filenames):
            details = files[filename]
            if not isinstance(details, dict):
                raise ValueError(f"Invalid manifest entry: {filename}")
            content = archive.read(filename)
            if not content:
                raise ValueError(f"Model file is empty: {filename}")
            checksum = hashlib.sha256(content).hexdigest()
            if len(content) != details.get("size_bytes") or checksum != details.get("sha256"):
                raise ValueError(f"Model verification failed: {filename}")
            model_contents[filename] = content
    destination.parent.mkdir(parents = True, exist_ok = True)
    with TemporaryDirectory(dir = destination.parent, prefix = "models-") as temporary_dir:
        staging = Path(temporary_dir)
        for filename, content in model_contents.items():
            (staging / filename).write_bytes(content)
        (staging / "manifest.json").write_bytes(manifest_content)
        staging.rename(destination)
    return str(destination)