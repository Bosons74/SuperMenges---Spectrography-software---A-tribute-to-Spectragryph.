# SuperMenges V1.17 — soft palette, rebrand, simplifications

## What changed

V1.17 is a polish release focused on the user-experience details. All scientific functionality from V1.16 is retained.

### Rebranding to SuperMenges

The product is now called **SuperMenges**, named in tribute to Dr Friedrich Menges (1974–2024), the creator of Spectragryph whose work shaped a generation of vibrational-spectroscopy practice.

This is a name change only. The internal Python package remains `openspectra_workbench` for compatibility — every existing import and CLI invocation continues to work unchanged.

A new **Help → About SuperMenges** dialog states the tribute *and* the non-affiliation note clearly: *"Independent project — not affiliated with Spectragryph or Spectroscopy Ninja. No proprietary code copied or reverse-engineered."*

### Soft tinted palette (light skin)

The light skin received a gentle three-region accent palette:

- **Menubar** at top: vert d'eau (seafoam, `#dbece4`)
- **Details / Results** panel on the right: pêche (soft peach, `#fbe6d4`)
- **Peaks / FWHM** table at bottom-right: lavande (`#e8def4`)

The dark skin gets proportional darker analogues so it still feels coherent (deep seafoam menubar `#26342f`, deep peach details `#34291f`, deep lavender peaks `#2c2538`).

The high-contrast and oceanographic skins are unchanged — they have specific accessibility/aesthetic purposes.

### Right-click → Reset zoom (auto-scale)

The plot's right-click context menu now starts with a bold **⛶ Reset zoom (auto-scale)** action at the very top, separated from the rest by a divider so it's the first thing users see. It calls `vb.autoRange()` to fit all visible data.

### Removed

- **Sepia** and **solarized** skins. The remaining four (light / dark / high_contrast / oceanographic) cover the practical needs without theme bloat.
- **JWS diagnostic** workflow:
  - File menu entry "JWS diagnostic report..." gone
  - Tools menu entry "JWS/JASCO diagnostic report..." gone
  - Toolbar button "JWS diag" gone
  - CLI subcommand `inspect-jws` gone
  - `JWS_FORMAT_NOTES.md` deleted
- The actual `.jws` file **reader** in `openspectra_workbench/io/jws.py` is **kept** — opening JASCO `.jws` spectra continues to work normally. Only the diagnostic-report layer has been removed.

## Tests

182 / 182 pass — same coverage as V1.16. No tests touched JWS diagnostics (it was a data-shape inspector with no test coverage before V1.17, and removing it leaves the test count unchanged).

## Files changed

```
openspectra_workbench/__init__.py            # version 1.17.0
openspectra_workbench/app/themes.py          # 6 → 4 skins, soft tint fields on Theme
openspectra_workbench/app/main_window.py     # rename, About dialog, reset zoom, no JWS diag
openspectra_workbench/app/__main__.py        # setApplicationName("SuperMenges")
openspectra_workbench/cli.py                 # inspect-jws subcommand removed
openspectra_workbench/core/project.py        # Project format string renamed
openspectra_workbench/export/exporters.py    # Header strings renamed
openspectra_workbench/export/templates.py    # PDF footer renamed
openspectra_workbench/export/report_markdown.py  # Markdown title renamed
JWS_FORMAT_NOTES.md                          # deleted
```

## Backwards compatibility

- Existing project JSON files (with `"format": "OpenSpectra Workbench Project"`) still load — the loader doesn't validate this string strictly, it's metadata.
- Existing user library SQLite files load unchanged.
- All existing `openspectra_workbench` imports continue to resolve.
- Preferences file (the saved skin name etc.) reads as before. If a user had picked `sepia` or `solarized` in V1.16, V1.17 falls back to the default `light` skin gracefully (no crash) because `THEMES` no longer contains those keys.
