# -*- coding: utf-8 -*-
"""Pruebas para scripts/generate_study_area_map.py."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from scripts import generate_study_area_map as gsam


def test_bbox_matches_specification():
    assert gsam.BBOX == (-82.0, -19.0, -34.0, 0.0)


def test_aeronet_stations_are_complete_and_correct():
    stations = {s["abbr"]: s for s in gsam.AERONET_STATIONS}
    assert len(gsam.AERONET_STATIONS) == 5
    assert set(stations) == {"HYO", "LPZ", "RB", "MNS", "ARA"}

    expected = {
        "HYO": ("Huancayo", -75.3, -12.0),
        "LPZ": ("La Paz", -68.15, -16.5),
        "RB": ("Rio Branco", -67.87, -9.96),
        "MNS": ("Manaus", -60.02, -3.1),
        "ARA": ("Arica", -70.3, -18.48),
    }
    for abbr, (site_name, lon, lat) in expected.items():
        station = stations[abbr]
        assert station["site_name"] == site_name
        assert station["longitude"] == pytest.approx(lon)
        assert station["latitude"] == pytest.approx(lat)


def test_all_stations_are_within_bbox():
    min_lon, min_lat, max_lon, max_lat = gsam.BBOX
    for station in gsam.AERONET_STATIONS:
        assert min_lon <= station["longitude"] <= max_lon
        assert min_lat <= station["latitude"] <= max_lat


def test_required_country_labels_present():
    required = {
        "PERU",
        "BOLIVIA",
        "BRAZIL",
        "COLOMBIA",
        "SURINAME",
        "GUYANA",
        "FRENCH GUIANA",
        "VENEZUELA",
    }
    labeled = {name for name, _, _ in gsam.COUNTRY_LABELS}
    assert required.issubset(labeled)


def test_ocean_labels_present_in_english():
    labeled = {name for name, *_ in gsam.OCEAN_LABELS}
    assert "Pacific Ocean" in labeled
    assert "Atlantic Ocean" in labeled


def test_download_elevation_data_degrades_gracefully_without_network(tmp_path, monkeypatch):
    """Si no hay red disponible, debe devolver None sin lanzar excepciones."""

    def _boom(*args, **kwargs):
        raise ConnectionError("sin conexión (simulado)")

    monkeypatch.setattr(gsam, "ELEVATION_SOURCES", ["https://example.invalid/{min_lat}/{max_lat}"])

    import requests

    monkeypatch.setattr(requests, "get", _boom)

    result = gsam.download_elevation_data(cache_dir=tmp_path)
    assert result is None


def test_download_elevation_data_uses_cache(tmp_path):
    import numpy as np

    cache_path = tmp_path / "elevation_cache.npz"
    lon = np.linspace(-82, -34, 4)
    lat = np.linspace(-19, 0, 3)
    elevation = np.zeros((3, 4))
    np.savez_compressed(cache_path, lon=lon, lat=lat, elevation=elevation)

    result = gsam.download_elevation_data(cache_dir=tmp_path)
    assert result is not None
    assert list(result["lon"]) == pytest.approx(list(lon))


def _read_png_dpi(path: Path):
    """Lee el pHYs chunk de un PNG y devuelve el dpi (x, y)."""
    with open(path, "rb") as fh:
        data = fh.read()
    idx = data.find(b"pHYs")
    assert idx != -1, "El PNG no contiene metadatos de resolución (pHYs)."
    x_ppu, y_ppu, unit = struct.unpack(">IIB", data[idx + 4: idx + 13])
    assert unit == 1  # metros
    to_dpi = lambda ppu: round(ppu * 0.0254)
    return to_dpi(x_ppu), to_dpi(y_ppu)


def test_generate_study_area_map_creates_png_at_170_dpi(tmp_path):
    img_dir = tmp_path / "img"
    data_dir = tmp_path / "data"

    output = gsam.generate_study_area_map(
        img_dir=img_dir,
        data_dir=data_dir,
        elevation_data=None,  # evita intentos de red durante la prueba
    )

    assert output == img_dir / gsam.OUTPUT_FILENAME
    assert output.exists()
    assert output.stat().st_size > 0

    dpi_x, dpi_y = _read_png_dpi(output)
    assert dpi_x == gsam.DPI
    assert dpi_y == gsam.DPI


def test_generate_study_area_map_accepts_custom_output_path(tmp_path):
    custom_output = tmp_path / "custom_name.png"

    output = gsam.generate_study_area_map(
        output_path=custom_output,
        img_dir=tmp_path / "unused_img_dir",
        data_dir=tmp_path / "data",
        elevation_data=None,
    )

    assert output == custom_output
    assert custom_output.exists()
