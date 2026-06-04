"""
Upload exp2c artifacts to HuggingFace Hub.

Run once:
    python upload_to_hf.py

Requires:
    pip install huggingface_hub
    huggingface-cli login   (or set HF_TOKEN env var)
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "Nazim112/nl-energy-forecaster"
ARTIFACTS = Path("outputs/exp2c_encoder_decoder/exp2_standard/artifacts")
CONFIG = Path("outputs/exp2c_encoder_decoder/exp2_standard/resolved_config.yaml")

FILES = [
    # (local_path, path_in_repo)
    (ARTIFACTS / "best_model.pt",      "best_model.pt"),
    (ARTIFACTS / "scalers.joblib",     "scalers.joblib"),
    (CONFIG,                           "resolved_config.yaml"),
    (Path("predict.py"),               "predict.py"),
    (Path("src/models/encoder_decoder_transformer.py"), "src/models/encoder_decoder_transformer.py"),
    (Path("src/models/layers.py"),     "src/models/layers.py"),
    (Path("src/utils/config.py"),      "src/utils/config.py"),
]


def main():
    api = HfApi()

    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{REPO_ID}")

    for local, remote in FILES:
        local = Path(local)
        if not local.exists():
            print(f"  SKIP (missing): {local}")
            continue
        print(f"  Uploading {local} -> {remote} ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print("done")

    # create __init__.py stubs so src/ is importable from the HF snapshot
    for stub in ["src/__init__.py", "src/models/__init__.py", "src/utils/__init__.py"]:
        api.upload_file(
            path_or_fileobj=b"",
            path_in_repo=stub,
            repo_id=REPO_ID,
            repo_type="model",
        )

    print(f"\nDone. Model card: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
