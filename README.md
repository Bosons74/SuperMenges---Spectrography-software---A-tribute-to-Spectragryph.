# SuperMenges V1.17

A serious, open-source spectroscopy workbench for **UV-Vis, NIR, FTIR, and Raman**.

This software is named in tribute to **Dr Friedrich Menges (1974–2024)**, the creator of Spectragryph whose work shaped a generation of vibrational-spectroscopy practice.

> **Independent project — not affiliated with Spectragryph or Spectroscopy Ninja.** No proprietary code copied or reverse-engineered. Any third-party readers (such as `spc-spectra`, `brukeropusreader`) are permissively licensed open-source dependencies.

The internal Python package name remains `openspectra_workbench` for compatibility with existing imports, scripts, and projects.

## What's new in V1.17

- Renamed **OpenSpectra Workbench → SuperMenges** (display only, package unchanged)
- New **Help → About SuperMenges** with tribute and non-affiliation notice
- Soft-tinted palette on the light and dark skins:
  - Menubar in vert d'eau (light) / deep seafoam (dark)
  - Details / Results panel in pêche (light) / deep peach (dark)
  - Peaks / FWHM table in lavande (light) / deep lavender (dark)
- **⛶ Reset zoom (auto-scale)** at the very top of the plot's right-click menu, bold and emphasized
- Removed sepia and solarized skins (4 skins remain: light, dark, high_contrast, oceanographic)
- Removed the JWS diagnostic dialog and `inspect-jws` CLI (the `.jws` file reader itself is kept)

## Launch

```bash
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
.venv\Scripts\activate              # Windows
pip install -r requirements.txt
python -m openspectra_workbench.app
```

## What this software does (V1.5 → V1.17 cumulative)

### Reading
JCAMP-DX (`.jdx`/`.dx`), CSV/TXT/TSV/ASC/DAT, MSA/EMSA, RRUFF, JASCO `.jws`, Renishaw, Thermo Omnic, Ocean Optics. Optional readers for SPC and Bruker OPUS via `spc-spectra` and `brukeropusreader`.

### Spectral processing
- Smoothing, despiking, normalisation
- Baselines: rubberband, polynomial, ALS, arPLS, anchor-points (linear / cubic / PCHIP)
- Axis transforms: nm ↔ cm⁻¹ ↔ eV ↔ THz, Raman shift converter from laser nm
- ATR correction for FTIR
- Atmospheric H₂O / CO₂ background subtraction
- Kubelka-Munk forward and inverse

### Analysis
- Peak finding with prominence, height, FWHM
- Multi-peak fitting (Gaussian / Lorentzian / pseudo-Voigt / Voigt)
- Region-weighted compare (FTIR fingerprint / Raman defaults)
- A−B difference view with rolling RMSE

### Quantification
- Beer-Lambert calibration with R² / LOD / LOQ per ICH Q2
- PLS regression with cross-validated component selection
- CIE color from spectra under D65 / A illuminants

### Hyperspectral imaging
- ENVI cube reader
- ASCII cube reader
- Band-integral chemical image
- k-means classification
- Mean spectrum

### Chemometrics (V1.15 + V1.16)
**Pretreatments**: SNV, MSC, EMSC (with polynomial baseline + interferent projection), OSC, detrending, autoscale, Pareto, range, Poisson, mean-centre. Each returns a fit-state for replay on validation data.

**Modelling**: PLS-DA classification with VIP and Selectivity Ratio diagnostics; PCR regression; SIMCA classification with built-in outlier detection (Hotelling T² + Q-residual diagnostics with proper F-distribution and Jackson-Mudholkar critical values).

**Region & variable selection**: iPLS (scan / forward / backward) and CARS Monte Carlo selection.

**Validation**: Kennard-Stone calibration/validation split, permutation testing for empirical p-values, Y-randomisation with Eriksson-style intercept regression for chance-correlation detection.

### Spectral library matching
Multi-library SQLite system. Each library has a name, an optional source folder for re-indexing, and an enable/disable flag. Library Manager dialog with searchable, sortable spectra table. Match reports in Markdown / CSV / HTML formats.

### Macros and plugins
JSON-based macros for replayable workflows, plus a plugin loader for custom processing operations.

## CLI

```bash
# File conversion
openspectra-workbench convert source.jdx --out out.csv --format csv --baseline als

# Library
openspectra-workbench library-create Wiley_ATR_IR --db lib.sqlite --folder Wiley_refs/
openspectra-workbench library-match unknown.jdx --db lib.sqlite --report match.md

# Chemometrics
openspectra-workbench chemo-pls-da labels.csv --vip-csv vip.csv
openspectra-workbench chemo-simca labels.csv --predict unknown.jdx --alpha 0.01
openspectra-workbench chemo-ipls concentrations.csv --mode forward --n-intervals 10
openspectra-workbench chemo-cars concentrations.csv --n-iterations 50

# FTIR
openspectra-workbench auto-ftir raw.jdx --out clean.jdx --baseline als --mask-atmosphere
openspectra-workbench ftir-qc sample.jdx
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

182 / 182 pass.

## Skin selection

Use **View → Skin** to switch at runtime. The choice persists across sessions in the platform-appropriate config location.
