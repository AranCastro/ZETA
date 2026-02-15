# ZETA

Smart Mineral Formula Calculator for microprobe-style oxide data.

## Features

1. Structural formula calculations (APFU) from CSV/Excel oxide tables.
2. Auto mineral detection (`garnet`, `pyroxene`, `amphibole`) with QC flags.
3. Batch processing + export tables (publication, ternary, GeoRoc-style).
4. `ZETA-Plot` discrimination diagrams:
   - Pyroxene quadrilateral (Wo-En-Fs)
   - Amphibole Leake-style proxy
   - Feldspar ternary (An-Ab-Or)
   - Garnet composition plots
5. Advanced power tools:
   - Naming/classification outputs
   - Zoning profile analysis + diffusion estimate
   - Recalculation workflows (Fe2+/Fe3+, renormalizations)
   - Thermobarometry proxies + Fe-Mg equilibrium filters
6. Built-in sample data box in UI (microprobe/formula/traverse).

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run zeta_app.py
```

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```
