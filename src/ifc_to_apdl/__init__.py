"""ifc_to_apdl — multi-system semantic conversion of IFC architectural models
into analysis-ready Mechanical APDL models.

Pipeline (see PLAN.md):
    Phase 0  ingest      — normalize units, precision, placements, representations
    Phase 1  classify    — directed relationship graph -> domain-labeled systems
    Phase 2  assign      — element formulations + material evidence cascade
    Phase 3  geometry    — sections & centerlines (parametric and tessellated)
    Phase 4  assemble    — analytical topology + domain compatibility
    Phase 5  emit        — deterministic, preprocessing-only APDL deck from the IR
    Phase 6  verify      — optional PyMAPDL batch QA (solves issued at run time)

Entry points: ``python main.py model.ifc`` at the repo root, the ``ifc2apdl``
console script, or ``convert_file(ifc, out_dir, ConversionConfig())``.
"""

__version__ = "0.1.0"

from .config import ConversionConfig
from .pipeline import convert_file

__all__ = ["ConversionConfig", "convert_file", "__version__"]
