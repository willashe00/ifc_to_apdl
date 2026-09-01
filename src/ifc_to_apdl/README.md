# ifc_to_apdl

Multi-system semantic conversion of IFC architectural models into
analysis-ready Mechanical APDL decks. Implements the four-phase methodology
of dissertation Chapter 3 with the refinements documented in `../../PLAN.md`.

## Usage

```bash
# drag-and-drop GUI (from the repo root)
python main.py

# convert (one deck per classified system + audit report)
python -m ifc_to_apdl.cli convert model.ifc -o out/

# convert and batch-solve every deck under PyMAPDL QA
python -m ifc_to_apdl.cli convert model.ifc -o out/ --verify
```

Python API:

```python
from ifc_to_apdl import ConversionConfig, convert_file
results = convert_file("model.ifc", "out/")
for r in results:
    print(r.system.domain, r.deck, r.model.expected_mass())
```

## Package layout

| package | phase | role |
|---|---|---|
| `ingest` | 0 | unit/precision normalization, placement + mapped-item resolution |
| `classify` | 1 | relationship graph (aggregation/ports/proximity), domain scoring |
| `assign` | 2 | element formulation table, material evidence cascade + dense retrieval |
| `geometry` | 3 | profile/swept-solid recovery, axis-rep centerlines, tessellated recovery |
| `assemble` | 4 | per-domain topology + compatibility (NodePool, snapping, partitioning) |
| `emit` | 5 | deterministic APDL writer (CM components, parameterized deck) |
| `verify` | 6 | PyMAPDL batch QA: mass reconciliation, mechanism + gravity screens |
| `model` | — | the solver-neutral analytical IR |
| `report` | — | conversion audit ledger (JSON + Markdown) |

## Key design rules

- Every derived quantity records its evidence path (declared IFC data ->
  parametric geometry -> computational recovery -> flagged for review).
- Every `IfcProduct` ends as converted / excluded-with-rule / unhandled /
  flagged — never silently dropped.
- The APDL writer emits every number from the IR exactly once; mesh sizes,
  tolerances and mode counts are `*SET` parameters, not literals.
- Supports select tracked CM components, never coordinate-window `NSEL`.
- Generated decks are runnable as-is and are batch-verified when
  `--verify` is passed.
