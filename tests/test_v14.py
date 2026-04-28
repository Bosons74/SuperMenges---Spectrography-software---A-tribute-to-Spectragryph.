"""V1.14 tests — multi-library SQLite, folder indexing, match reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openspectra_workbench.core.spectrum import Spectrum
from openspectra_workbench.library.database import (
    DEFAULT_SCANNABLE_EXTENSIONS,
    SpectralLibrary,
)
from openspectra_workbench.export.match_report import write_match_report


def _make_spec(name: str, peak_x: float = 1500.0) -> Spectrum:
    """Synthetic FTIR-like spectrum with a single peak at ``peak_x``."""
    x = np.linspace(400, 4000, 1801)
    y = np.exp(-((x - peak_x) / 25) ** 2) + 0.05
    return Spectrum(
        x=x, y=y, name=name, technique="FTIR", x_unit="cm-1", y_unit="absorbance",
        source_path="", metadata={}, annotations=[], regions=[],
    )


# ---------------------------------------------------------------------------
# Library CRUD
# ---------------------------------------------------------------------------

class TestLibraryCRUD:
    def test_default_library_present(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        libs = lib.list_libraries()
        assert any(l.name == "default" and l.id == 1 for l in libs)

    def test_create_named_library(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        lib_id = lib.create_library("my_refs", folder="", description="for testing")
        assert lib_id != 1
        assert "my_refs" in [l.name for l in lib.list_libraries()]

    def test_create_duplicate_name_raises(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        lib.create_library("foo")
        with pytest.raises(Exception):
            lib.create_library("foo")

    def test_create_empty_name_raises(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        with pytest.raises(ValueError):
            lib.create_library("")

    def test_rename_library(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        lid = lib.create_library("oldname")
        lib.rename_library(lid, "newname")
        names = [l.name for l in lib.list_libraries()]
        assert "newname" in names and "oldname" not in names

    def test_delete_library_with_spectra(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        lid = lib.create_library("temp")
        sid = lib.add_spectrum(_make_spec("a"), library_id=lid)
        assert lib.count(library_id=lid) == 1
        lib.delete_library(lid, delete_spectra=True)
        assert lid not in [l.id for l in lib.list_libraries()]
        with pytest.raises(KeyError):
            lib.get_spectrum(sid)

    def test_delete_library_reassign(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        lid = lib.create_library("temp")
        sid = lib.add_spectrum(_make_spec("a"), library_id=lid)
        lib.delete_library(lid, delete_spectra=False)
        assert lib.get_row(sid)["library_id"] == 1

    def test_cannot_delete_default_library(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        with pytest.raises(ValueError):
            lib.delete_library(1)

    def test_enable_disable(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        lid = lib.create_library("toggle")
        lib.set_enabled(lid, False)
        assert next(l for l in lib.list_libraries() if l.id == lid).enabled is False
        lib.set_enabled(lid, True)
        assert next(l for l in lib.list_libraries() if l.id == lid).enabled is True


# ---------------------------------------------------------------------------
# Spectra in named libraries
# ---------------------------------------------------------------------------

class TestSpectraScoping:
    def test_spectrum_assigned_to_library(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        a = lib.create_library("A"); b = lib.create_library("B")
        lib.add_spectrum(_make_spec("a1"), library_id=a)
        lib.add_spectrum(_make_spec("a2"), library_id=a)
        lib.add_spectrum(_make_spec("b1"), library_id=b)
        assert lib.count(library_id=a) == 2
        assert lib.count(library_id=b) == 1
        assert lib.count() == 3

    def test_list_all_filtered_by_library(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        a = lib.create_library("A"); b = lib.create_library("B")
        lib.add_spectrum(_make_spec("a1"), library_id=a)
        lib.add_spectrum(_make_spec("b1"), library_id=b)
        assert {r["name"] for r in lib.list_all(library_ids=[a])} == {"a1"}
        assert {r["name"] for r in lib.list_all(library_ids=[b])} == {"b1"}

    def test_search_within_libraries(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        a = lib.create_library("A"); b = lib.create_library("B")
        lib.add_spectrum(_make_spec("acetone"), library_id=a)
        lib.add_spectrum(_make_spec("acetone"), library_id=b)
        assert len(lib.search("acetone")) == 2
        rows = lib.search("acetone", library_ids=[a])
        assert len(rows) == 1 and rows[0]["library_id"] == a

    def test_list_includes_library_name(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "test.sqlite")
        lid = lib.create_library("MyLib")
        lib.add_spectrum(_make_spec("foo"), library_id=lid)
        target = next(r for r in lib.list_all() if r["name"] == "foo")
        assert target["library_name"] == "MyLib"
        assert target["library_id"] == lid


# ---------------------------------------------------------------------------
# Folder indexing
# ---------------------------------------------------------------------------

def _write_csv(path: Path, peak_x: float = 1500.0) -> None:
    x = np.linspace(400, 4000, 200)
    y = np.exp(-((x - peak_x) / 25) ** 2) + 0.05
    with path.open("w") as f:
        f.write("wavenumber,absorbance\n")
        for xv, yv in zip(x, y):
            f.write(f"{xv:.3f},{yv:.6f}\n")


class TestFolderIndexing:
    def test_index_folder(self, tmp_path):
        ref_dir = tmp_path / "refs"; ref_dir.mkdir()
        _write_csv(ref_dir / "acetone.csv", 1715)
        _write_csv(ref_dir / "ethanol.csv", 1050)
        _write_csv(ref_dir / "water.csv", 1640)
        (ref_dir / "readme.txt").write_text("notes")
        lib = SpectralLibrary(tmp_path / "lib.sqlite")
        lid = lib.create_library("WileyCustom")
        result = lib.index_folder(lid, ref_dir, scannable_extensions={".csv"})
        assert result["added"] == 3
        assert result["skipped"] == 0
        assert lib.count(library_id=lid) == 3
        recorded = next(l for l in lib.list_libraries() if l.id == lid)
        assert Path(recorded.folder) == ref_dir

    def test_rebuild_replaces_content(self, tmp_path):
        ref_dir = tmp_path / "refs"; ref_dir.mkdir()
        _write_csv(ref_dir / "a.csv", 1000)
        _write_csv(ref_dir / "b.csv", 1500)
        lib = SpectralLibrary(tmp_path / "lib.sqlite")
        lid = lib.create_library("L")
        lib.index_folder(lid, ref_dir, scannable_extensions={".csv"})
        assert lib.count(library_id=lid) == 2
        _write_csv(ref_dir / "c.csv", 2000)
        lib.index_folder(lid, ref_dir, scannable_extensions={".csv"})
        assert lib.count(library_id=lid) == 5
        lib.index_folder(lid, ref_dir, scannable_extensions={".csv"}, rebuild=True)
        assert lib.count(library_id=lid) == 3

    def test_index_nonexistent_folder_raises(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "lib.sqlite")
        lid = lib.create_library("L")
        with pytest.raises(ValueError):
            lib.index_folder(lid, tmp_path / "does_not_exist")

    def test_default_extensions_include_jdx(self):
        assert ".jdx" in DEFAULT_SCANNABLE_EXTENSIONS
        assert ".csv" in DEFAULT_SCANNABLE_EXTENSIONS


# ---------------------------------------------------------------------------
# Matching with library scopes
# ---------------------------------------------------------------------------

class TestMatchingScopes:
    def test_match_only_enabled_libraries(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "lib.sqlite")
        a = lib.create_library("A"); b = lib.create_library("B")
        lib.add_spectrum(_make_spec("ref_in_A", 1500), library_id=a)
        lib.add_spectrum(_make_spec("ref_in_B", 1500), library_id=b)
        unknown = _make_spec("unknown", 1500)
        rows = lib.match(unknown, enabled_only=True, limit=10)
        assert {r["name"] for r in rows} == {"ref_in_A", "ref_in_B"}
        lib.set_enabled(b, False)
        rows = lib.match(unknown, enabled_only=True, limit=10)
        assert {r["name"] for r in rows} == {"ref_in_A"}

    def test_match_explicit_library_subset(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "lib.sqlite")
        a = lib.create_library("A"); b = lib.create_library("B")
        lib.add_spectrum(_make_spec("ref_a"), library_id=a)
        lib.add_spectrum(_make_spec("ref_b"), library_id=b)
        rows = lib.match(_make_spec("unknown"), library_ids=[a], limit=10)
        assert {r["name"] for r in rows} == {"ref_a"}

    def test_match_returns_library_metadata(self, tmp_path):
        lib = SpectralLibrary(tmp_path / "lib.sqlite")
        lid = lib.create_library("MyLib")
        lib.add_spectrum(_make_spec("ref"), library_id=lid)
        rows = lib.match(_make_spec("unknown"), enabled_only=True, limit=5)
        assert rows[0]["library_name"] == "MyLib"
        assert rows[0]["library_id"] == lid


# ---------------------------------------------------------------------------
# V1.13 → V1.14 migration
# ---------------------------------------------------------------------------

class TestV13Migration:
    def test_old_database_migrated_in_place(self, tmp_path):
        import sqlite3
        path = tmp_path / "v13.sqlite"
        conn = sqlite3.connect(str(path))
        conn.executescript(
            "CREATE TABLE spectra ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " name TEXT NOT NULL, cas TEXT, technique TEXT,"
            " x_unit TEXT, y_unit TEXT, source TEXT,"
            " tags TEXT, notes TEXT,"
            " x_json TEXT NOT NULL, y_json TEXT NOT NULL, metadata_json TEXT);"
        )
        conn.execute(
            "INSERT INTO spectra(name, x_json, y_json) VALUES (?, ?, ?)",
            ("legacy_spectrum", json.dumps([1.0, 2.0]), json.dumps([0.5, 0.6])),
        )
        conn.commit(); conn.close()
        lib = SpectralLibrary(path)
        assert "default" in [l.name for l in lib.list_libraries()]
        rows = lib.list_all(library_ids=[1])
        assert any(r["name"] == "legacy_spectrum" for r in rows)


# ---------------------------------------------------------------------------
# Match report writer
# ---------------------------------------------------------------------------

class TestMatchReports:
    def _rows(self) -> list[dict]:
        return [
            {"id": 1, "name": "Acetone", "library_name": "Wiley_ATR_IR",
             "technique": "FTIR", "cas": "67-64-1",
             "final_v1_9_score": 0.987, "diagnostic_score": 0.95,
             "derivative_diagnostic_score": 0.9},
            {"id": 5, "name": "Ethanol", "library_name": "Wiley_ATR_IR",
             "technique": "FTIR", "cas": "64-17-5",
             "final_v1_9_score": 0.812},
            {"id": 9, "name": "Methanol", "library_name": "InHouse",
             "technique": "FTIR", "cas": "67-56-1",
             "final_v1_9_score": 0.654},
        ]

    def test_markdown_report(self, tmp_path):
        out = tmp_path / "report.md"
        write_match_report(self._rows(), out, unknown_name="my_unknown")
        text = out.read_text()
        assert "OpenSpectra match report" in text
        assert "my_unknown" in text
        assert "Acetone" in text and "Wiley_ATR_IR" in text
        assert "0.9870" in text
        assert "|" in text  # markdown table

    def test_csv_report(self, tmp_path):
        out = tmp_path / "report.csv"
        write_match_report(self._rows(), out, unknown_name="x")
        lines = out.read_text().splitlines()
        assert lines[0].startswith("Rank,Reference,Library")
        assert len(lines) == 4

    def test_html_report(self, tmp_path):
        out = tmp_path / "report.html"
        write_match_report(self._rows(), out, unknown_name="my_unknown")
        text = out.read_text()
        assert text.lstrip().startswith("<!DOCTYPE html>")
        assert "<table>" in text
        assert "Acetone" in text

    def test_format_chosen_by_extension(self, tmp_path):
        for ext, marker in [(".md", "|"), (".csv", "Rank,"), (".html", "<table>")]:
            out = tmp_path / f"r{ext}"
            write_match_report(self._rows(), out, unknown_name="x")
            assert marker in out.read_text()
