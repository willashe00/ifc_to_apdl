"""Command-line entry point.

    ifc2apdl model.ifc [-o DIR] [--mesh-size S] [--vertical-axis y|z] [--verify]

Converts one (federated) IFC file into one preprocessing-only APDL macro
(``<stem>_<system>.txt``) per classified system. By default the files land in
``<output root>/<ifc stem>/``; ``-o`` names an explicit folder. Coverage is
summarised on the terminal only; no audit files are written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConversionConfig
from .pipeline import convert_and_verify, convert_file


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ifc2apdl",
        description="Multi-system IFC -> Mechanical APDL conversion "
                    "(preprocessing-only decks, one per system)")
    ap.add_argument("ifc", type=Path, help="input IFC file")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output folder (default: <output root>/<ifc stem>/)")
    ap.add_argument("--mesh-size", type=float, default=None,
                    help="global target element size [m]")
    ap.add_argument("--vertical-axis", choices=["y", "z"], default=None,
                    help="APDL vertical axis (default: y)")
    ap.add_argument("--verify", action="store_true",
                    help="after conversion, batch-solve each deck under PyMAPDL QA "
                         "(gravity + modal screens issued at run time; requires Ansys)")
    ap.add_argument("--modes", type=int, default=None,
                    help="number of modes for the --verify modal screen")
    return ap


def main(argv: list[str] | None = None, output_root: Path | None = None) -> int:
    """Run the converter. ``output_root`` anchors the default output folder
    (``main.py`` passes its own ``outputs/`` directory; falls back to CWD)."""
    args = build_parser().parse_args(argv)

    ifc: Path = args.ifc
    if not ifc.exists():
        print(f"error: IFC file not found: {ifc}", file=sys.stderr)
        return 2
    if ifc.suffix.lower() != ".ifc":
        print(f"warning: {ifc.name} does not have an .ifc extension", file=sys.stderr)

    root = output_root if output_root is not None else Path.cwd() / "outputs"
    out_dir: Path = args.out if args.out is not None else root / ifc.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ConversionConfig()
    if args.mesh_size:
        cfg.mesh.target_size = args.mesh_size
    if args.vertical_axis:
        cfg.vertical_axis = args.vertical_axis
    if args.modes:
        cfg.verify.modal_modes = args.modes

    print(f"Input:  {ifc}")
    print(f"Output: {out_dir}")

    if args.verify:
        results, reports = convert_and_verify(ifc, out_dir, cfg)
    else:
        results, reports = convert_file(ifc, out_dir, cfg), []

    _print_summary(results)
    if args.verify:
        print()
        for rep in reports:
            print(rep.summary())
        return 0 if reports and all(r.passed for r in reports) else 1
    return 0


def _print_summary(results) -> None:
    if not results:
        print("No convertible systems were classified in this model.")
        return
    for r in results:
        m = r.model
        print(f"  {r.system.name} ({r.system.domain}): "
              f"{len(m.members)} members, {len(m.surfaces)} surfaces, "
              f"{len(m.volumes)} volumes -> {r.deck.name}")
    summary = results[0].audit.counts()
    if summary:
        print("Coverage: " + ", ".join(f"{k}: {v}" for k, v in summary.items()))
    for extra in ("flagged", "unhandled"):
        n = summary.get(extra, 0)
        if n:
            print(f"NOTE: {n} product(s) {extra}")


if __name__ == "__main__":
    sys.exit(main())
