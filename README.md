# ifc_to_apdl

Converts a (federated) IFC model into one Mechanical APDL
macro per classified system. Decks contain element types, materials, sections, geometry, mesh
and boundary conditions only.

## Run

```
python main.py path\to\model.ifc
```

Outputs are directed to `outputs/<model stem>/`.

## Supported systems

- **Buildings**: beams, columns, braces, walls, slabs (BEAM188 / SHELL181).
- **Piping**: segments, elbows, tees, hangers (PIPE288 / ELBOW290 / COMBIN14).
- **Nuclear containment**: cylinder walls, base-slab discs and hemispherical
  domes