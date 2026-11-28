import hashlib
from pathlib import Path
import json
from zipfile import ZIP_DEFLATED, ZipFile

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