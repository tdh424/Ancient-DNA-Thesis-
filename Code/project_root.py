from pathlib import Path
import os


def project_root() -> Path:
    override = os.environ.get('ANCIENT_DNA_ROOT')
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent.parent