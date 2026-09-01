from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE / "src"))

def _ensure_interpreter() -> None:

    try:
        import pydantic  # noqa: F401
        import ifcopenshell  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("_IFC2APDL_REEXEC"):
        return
    home = Path.home()
    candidates = [os.environ.get("IFC2APDL_PYTHON")] + [
        str(root / "envs" / "py312" / "python.exe")
        for root in (home / "AppData/Local/anaconda3", home / "anaconda3",
                     home / "miniconda3", home / "AppData/Local/miniconda3",
                     Path("C:/ProgramData/anaconda3"))
    ]
    for cand in candidates:
        if cand and Path(cand).exists() and Path(cand) != Path(sys.executable):
            print(f"note: switching interpreter to {cand}", file=sys.stderr)
            os.environ["_IFC2APDL_REEXEC"] = "1"
            os.execv(cand, [cand, str(HERE / "main.py")] + sys.argv[1:])
    # nothing found: fall through and let the import error surface


_ensure_interpreter()

from ifc_to_apdl.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(output_root=HERE / "outputs"))
