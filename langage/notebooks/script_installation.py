from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Qwen/Qwen3.5-27B"
DEFAULT_LOCAL_DIR = Path("./modeles/Qwen3.5-27B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge le modèle Qwen dans un dossier local prêt à l'emploi."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Identifiant Hugging Face du modèle (défaut: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--local-dir",
        default=str(DEFAULT_LOCAL_DIR),
        help=f"Dossier de destination local (défaut: {DEFAULT_LOCAL_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or True

    print(f"Téléchargement de {args.repo_id} vers {local_dir} ...")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"Terminé: {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())