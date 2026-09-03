#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera el mapa cartográfico del área de estudio MODIS-AERONET.

Este script produce ``study_area_map.png`` con todas las características
cartográficas solicitadas para el área de estudio en Sudamérica:

* Orografía/relieve del terreno (datos de elevación tipo GEBCO/ETOPO, con
  descarga automática y caché local).
* Delimitación clara del área de estudio mediante el BBOX del proyecto.
* Etiquetas de países: Perú, Bolivia, Brasil, Colombia, Surinam, Guyana,
  Guyana Francesa y Venezuela.
* Etiquetas de océanos ("Pacific Ocean" / "Atlantic Ocean").
* Las 5 estaciones AERONET fijas del proyecto (HYO, LPZ, RB, MNS, ARA).
* Escala gráfica en km, rosa de los vientos, grilla de coordenadas y
  leyenda con las estaciones.
* Límites políticos (Natural Earth) y líneas costeras.
* Exportación en alta resolución (170 dpi, PNG).

El módulo puede ejecutarse de forma independiente::

    python scripts/generate_study_area_map.py

o importarse desde el pipeline existente::

    from scripts.generate_study_area_map import generate_study_area_map
    generate_study_area_map()
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("generate_study_area_map")

# ---------------------------------------------------------------------------
# Configuración del área de estudio (integrable con el pipeline existente
# mediante variables de entorno).
# ---------------------------------------------------------------------------

#: BBOX del área de estudio: (min_lon, min_lat, max_lon, max_lat).
BBOX: Tuple[float, float, float, float] = (-82.0, -19.0, -34.0, 0.0)

#: Directorio de salida de imágenes del pipeline.
IMG_DIR = Path(os.environ.get("IMG_DIR", "output/img"))

#: Directorio de caché para datos auxiliares (elevación, etc.).
DATA_DIR = Path(os.environ.get("DATA_DIR", "output/data"))

#: Nombre del archivo de salida del mapa del área de estudio.
OUTPUT_FILENAME = "study_area_map.png"

#: Resolución de salida en puntos por pulgada.
DPI = 170

#: Estaciones AERONET fijas del área de estudio.
AERONET_STATIONS: List[Dict[str, Any]] = [
    {"abbr": "HYO", "site_name": "Huancayo", "longitude": -75.3, "latitude": -12.0},
    {"abbr": "LPZ", "site_name": "La Paz", "longitude": -68.15, "latitude": -16.5},
    {"abbr": "RB", "site_name": "Rio Branco", "longitude": -67.87, "latitude": -9.96},
    {"abbr": "MNS", "site_name": "Manaus", "longitude": -60.02, "latitude": -3.1},
    {"abbr": "ARA", "site_name": "Arica", "longitude": -70.3, "latitude": -18.48},
]

#: Etiquetas de países visibles dentro/en el borde del BBOX: (nombre, lon, lat).
COUNTRY_LABELS: List[Tuple[str, float, float]] = [
    ("PERU", -75.5, -9.7),
    ("BOLIVIA", -61.5, -18.3),
    ("BRAZIL", -54.5, -10.5),
    ("COLOMBIA", -72.0, -1.5),
    ("VENEZUELA", -66.5, -0.3),
    ("GUYANA", -59.0, -0.85),
    ("SURINAME", -55.3, -0.3),
    ("FRENCH GUIANA", -51.7, -0.85),
]

#: Etiquetas de océanos: (nombre, lon, lat, tamaño de fuente, ángulo).
OCEAN_LABELS: List[Tuple[str, float, float, float, float]] = [
    ("Pacific Ocean", -79.0, -14.0, 13.0, 0.0),
    ("Atlantic Ocean", -37.5, -8.0, 13.0, 0.0),
]

#: URLs públicas de datos de elevación (GEBCO/ETOPO) servidas vía ERDDAP,
#: intentadas en orden hasta que una responda correctamente.
ELEVATION_SOURCES: List[str] = [
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/GEBCO_2020.nc"
    "?elevation[({min_lat}):({max_lat})][({min_lon}):({max_lon})]",
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc"
    "?altitude[({min_lat}):({max_lat})][({min_lon}):({max_lon})]",
]

# Nombre de la variable de elevación esperada dentro del NetCDF, por fuente.
_ELEVATION_VARS = ("elevation", "altitude", "z")


# ---------------------------------------------------------------------------
# Descarga de datos de elevación (orografía)
# ---------------------------------------------------------------------------

def download_elevation_data(
    bbox: Tuple[float, float, float, float] = BBOX,
    cache_dir: Path = DATA_DIR,
    sources: Sequence[str] = ELEVATION_SOURCES,
    timeout: int = 60,
) -> Optional[Dict[str, np.ndarray]]:
    """Descarga (o recupera de caché) una grilla de elevación para el BBOX.

    Intenta, en orden, las fuentes GEBCO/ETOPO indicadas en ``sources``. El
    resultado se almacena en caché local (``cache_dir``) para evitar
    descargas repetidas. Si ninguna fuente está disponible (por ejemplo, sin
    conexión a internet), se devuelve ``None`` y el mapa se genera igualmente
    usando un respaldo sin orografía detallada.

    Returns
    -------
    dict con claves ``lon``, ``lat`` (1D) y ``elevation`` (2D, shape
    ``(len(lat), len(lon))``), o ``None`` si no fue posible obtener datos.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "elevation_cache.npz"

    if cache_path.exists():
        try:
            with np.load(cache_path) as cached:
                LOGGER.info("Usando datos de elevación en caché: %s", cache_path)
                return {key: cached[key] for key in ("lon", "lat", "elevation")}
        except Exception:  # pragma: no cover - caché corrupta
            LOGGER.warning("Caché de elevación corrupta, se descartará: %s", cache_path)

    try:
        import requests
    except ImportError:  # pragma: no cover
        LOGGER.warning("El paquete 'requests' no está disponible; se omite la orografía.")
        return None

    try:
        import xarray as xr
    except ImportError:  # pragma: no cover
        LOGGER.warning("El paquete 'xarray' no está disponible; se omite la orografía.")
        return None

    for template in sources:
        url = template.format(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)
        try:
            LOGGER.info("Descargando datos de elevación desde %s", url)
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            with xr.open_dataset(_bytes_to_tempfile(response.content)) as dataset:
                var_name = next((v for v in _ELEVATION_VARS if v in dataset.variables), None)
                if var_name is None:
                    LOGGER.warning("Fuente %s no contiene variable de elevación reconocida.", url)
                    continue
                elevation = np.asarray(dataset[var_name].values, dtype=float)
                lon = np.asarray(dataset["longitude"].values if "longitude" in dataset else dataset["lon"].values)
                lat = np.asarray(dataset["latitude"].values if "latitude" in dataset else dataset["lat"].values)

            np.savez_compressed(cache_path, lon=lon, lat=lat, elevation=elevation)
            LOGGER.info("Datos de elevación guardados en caché: %s", cache_path)
            return {"lon": lon, "lat": lat, "elevation": elevation}
        except Exception as exc:  # noqa: BLE001 - se degrada de forma controlada
            LOGGER.warning("No se pudo descargar elevación desde %s (%s)", url, exc)
            continue

    LOGGER.warning(
        "No fue posible descargar datos de elevación de ninguna fuente; "
        "el mapa se generará sin orografía detallada."
    )
    return None


def _bytes_to_tempfile(data: bytes):
    """Escribe ``data`` en un archivo temporal y devuelve su ruta (str)."""
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    try:
        tmp.write(data)
    finally:
        tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Capas del mapa
# ---------------------------------------------------------------------------

def add_topography(ax, elevation: Optional[Dict[str, np.ndarray]]) -> None:
    """Dibuja el relieve del terreno (sombreado + colormap tipo terreno)."""
    import cartopy.crs as ccrs
    from matplotlib.colors import LightSource
    import matplotlib.pyplot as plt

    if elevation is not None:
        try:
            lon, lat, z = elevation["lon"], elevation["lat"], elevation["elevation"]
            ls = LightSource(azdeg=315, altdeg=45)
            cmap = plt.get_cmap("gist_earth")
            rgb = ls.shade(
                z,
                cmap=cmap,
                blend_mode="soft",
                vert_exag=50,
                dx=1,
                dy=1,
            )
            ax.imshow(
                rgb,
                extent=(lon.min(), lon.max(), lat.min(), lat.max()),
                origin="lower",
                transform=ccrs.PlateCarree(),
                interpolation="bilinear",
                zorder=0,
            )
            return
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Fallo al renderizar la orografía descargada (%s)", exc)

    # Respaldo: imagen base de relieve incluida con cartopy (baja resolución).
    try:
        ax.stock_img()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("No fue posible dibujar el respaldo de orografía (%s)", exc)
        ax.set_facecolor("#eaf3fb")


def add_base_layers(ax) -> None:
    """Agrega costas y límites políticos (Natural Earth)."""
    import cartopy.feature as cfeature

    for feature, style in (
        (
            cfeature.COASTLINE.with_scale("50m"),
            dict(linewidth=0.9, edgecolor="black", zorder=3),
        ),
        (
            cfeature.BORDERS.with_scale("50m"),
            dict(linewidth=0.8, edgecolor="dimgray", linestyle="--", zorder=3),
        ),
    ):
        try:
            # Cartopy resuelve (y descarga, si hace falta) las geometrías de
            # Natural Earth de forma perezosa al momento de dibujar. Se
            # fuerza aquí la resolución para poder degradar de forma
            # controlada si no hay conexión a internet, en lugar de fallar
            # tardíamente durante el guardado de la figura.
            list(feature.geometries())
            ax.add_feature(feature, **style)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "No fue posible descargar los datos de Natural Earth (%s): %s. "
                "Verifique la conexión a internet.",
                feature.name,
                exc,
            )


def add_country_labels(ax, labels: Sequence[Tuple[str, float, float]] = COUNTRY_LABELS) -> None:
    """Añade las etiquetas de país sobre el mapa."""
    import cartopy.crs as ccrs

    for name, lon, lat in labels:
        ax.text(
            lon,
            lat,
            name,
            transform=ccrs.PlateCarree(),
            fontsize=9,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            zorder=5,
            path_effects=_text_halo(),
        )


def add_ocean_labels(ax, labels: Sequence[Tuple[str, float, float, float, float]] = OCEAN_LABELS) -> None:
    """Añade las etiquetas de océanos en inglés."""
    import cartopy.crs as ccrs

    for name, lon, lat, size, angle in labels:
        ax.text(
            lon,
            lat,
            name,
            transform=ccrs.PlateCarree(),
            fontsize=size,
            fontstyle="italic",
            fontweight="bold",
            color="#0b3d66",
            ha="center",
            va="center",
            rotation=angle,
            zorder=5,
            path_effects=_text_halo(),
        )


def _text_halo():
    """Efecto de halo blanco para mejorar la legibilidad de las etiquetas."""
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=2.5, foreground="white")]


def add_stations(ax, stations: Sequence[Dict[str, Any]] = AERONET_STATIONS) -> None:
    """Dibuja las estaciones AERONET y construye la leyenda."""
    import cartopy.crs as ccrs

    marker_handle = ax.plot(
        [],
        [],
        marker="^",
        markersize=9,
        markerfacecolor="red",
        markeredgecolor="black",
        linestyle="None",
        label="Estación AERONET",
    )[0]

    for station in stations:
        lon = station["longitude"]
        lat = station["latitude"]
        abbr = station["abbr"]
        site_name = station.get("site_name", abbr)

        ax.plot(
            lon,
            lat,
            marker="^",
            markersize=9,
            markerfacecolor="red",
            markeredgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        ax.text(
            lon + 0.6,
            lat,
            f"{abbr} ({site_name})",
            transform=ccrs.PlateCarree(),
            fontsize=8,
            color="black",
            ha="left",
            va="center",
            zorder=6,
            path_effects=_text_halo(),
        )

    ax.legend(handles=[marker_handle], loc="lower left", framealpha=0.9, fontsize=9)


def add_gridlines(ax) -> None:
    """Agrega la grilla de coordenadas con etiquetas."""
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.6, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False


def add_north_arrow(ax, location: Tuple[float, float] = (0.94, 0.92), size: float = 0.06) -> None:
    """Dibuja una rosa de los vientos simplificada apuntando al norte."""
    x, y = location
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - size),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        arrowprops=dict(facecolor="black", width=4, headwidth=12, headlength=10),
        zorder=10,
    )


def add_scale_bar(ax, bbox: Tuple[float, float, float, float] = BBOX) -> None:
    """Agrega una escala gráfica en kilómetros en la esquina inferior derecha."""
    import cartopy.crs as ccrs
    from pyproj import Geod

    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0

    geod = Geod(ellps="WGS84")
    _, _, total_width_m = geod.inv(min_lon, mid_lat, max_lon, mid_lat)
    total_width_km = total_width_m / 1000.0

    # Elige una longitud "redonda" para la barra, aproximadamente 1/4 del ancho.
    candidates_km = [2000, 1000, 500, 250, 200, 100, 50, 25, 10]
    target = total_width_km / 4.0
    bar_km = min(candidates_km, key=lambda c: abs(c - target))

    # Convierte la longitud de la barra (km) a grados de longitud en mid_lat.
    lon_end, _, _ = geod.fwd(min_lon, mid_lat, 90, bar_km * 1000.0)
    bar_width_deg = lon_end - min_lon

    bar_lon_start = min_lon + (max_lon - min_lon) * 0.62
    bar_lat = min_lat + (max_lat - min_lat) * 0.06

    ax.plot(
        [bar_lon_start, bar_lon_start + bar_width_deg],
        [bar_lat, bar_lat],
        color="black",
        linewidth=2.5,
        transform=ccrs.PlateCarree(),
        zorder=10,
        solid_capstyle="butt",
    )
    for x in (bar_lon_start, bar_lon_start + bar_width_deg):
        ax.plot(
            [x, x],
            [bar_lat - 0.15, bar_lat + 0.15],
            color="black",
            linewidth=2.0,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
    ax.text(
        bar_lon_start + bar_width_deg / 2.0,
        bar_lat + 0.4,
        f"{bar_km} km",
        transform=ccrs.PlateCarree(),
        fontsize=9,
        ha="center",
        va="bottom",
        zorder=10,
        path_effects=_text_halo(),
    )


# ---------------------------------------------------------------------------
# Función principal reutilizable
# ---------------------------------------------------------------------------

def generate_study_area_map(
    output_path: Optional[Path] = None,
    bbox: Tuple[float, float, float, float] = BBOX,
    stations: Sequence[Dict[str, Any]] = AERONET_STATIONS,
    country_labels: Sequence[Tuple[str, float, float]] = COUNTRY_LABELS,
    ocean_labels: Sequence[Tuple[str, float, float, float, float]] = OCEAN_LABELS,
    dpi: int = DPI,
    img_dir: Path = IMG_DIR,
    data_dir: Path = DATA_DIR,
    elevation_data: Optional[Dict[str, np.ndarray]] = "auto",  # type: ignore[assignment]
    figsize: Tuple[float, float] = (14, 6),
) -> Path:
    """Genera y guarda el mapa del área de estudio.

    Parameters
    ----------
    output_path:
        Ruta completa del PNG de salida. Si es ``None`` se usa
        ``img_dir / OUTPUT_FILENAME``.
    bbox:
        Área de estudio ``(min_lon, min_lat, max_lon, max_lat)``.
    stations:
        Lista de estaciones AERONET (dicts con ``abbr``, ``site_name``,
        ``longitude`` y ``latitude``).
    elevation_data:
        Datos de elevación ya cargados (dict con ``lon``/``lat``/``elevation``),
        ``None`` para omitir la descarga, o ``"auto"`` (por defecto) para
        intentar descargarlos automáticamente.
    dpi:
        Resolución de exportación (170 dpi por defecto, según especificación).

    Returns
    -------
    Path
        Ruta del archivo PNG generado.
    """
    import matplotlib

    matplotlib.use("Agg")  # backend no interactivo, apto para pipelines/CI
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path) if output_path is not None else img_dir / OUTPUT_FILENAME

    if elevation_data == "auto":
        elevation_data = download_elevation_data(bbox=bbox, cache_dir=data_dir)

    min_lon, min_lat, max_lon, max_lat = bbox

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    add_topography(ax, elevation_data)
    add_base_layers(ax)
    add_gridlines(ax)
    add_ocean_labels(ax, ocean_labels)
    add_country_labels(ax, country_labels)
    add_stations(ax, stations)
    add_scale_bar(ax, bbox)
    add_north_arrow(ax)

    ax.set_title("Área de estudio MODIS-AERONET — Sudamérica", fontsize=13, fontweight="bold")

    # NOTA: se evita bbox_inches="tight" porque, con algunas versiones de
    # matplotlib/cartopy, el cálculo del bbox ajustado de las gridlines con
    # etiquetas puede expandirse incorrectamente a todo el lienzo. En su
    # lugar se reservan márgenes explícitos para un resultado consistente.
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.05)
    fig.savefig(output_path, dpi=dpi, format="png")
    plt.close(fig)

    LOGGER.info("Mapa del área de estudio guardado en %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Ruta de salida del PNG (por defecto: IMG_DIR/{OUTPUT_FILENAME}).",
    )
    parser.add_argument("--dpi", type=int, default=DPI, help="Resolución de salida en dpi (por defecto 170).")
    parser.add_argument(
        "--no-elevation",
        action="store_true",
        help="Omite la descarga de datos de elevación (usa el respaldo de relieve).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activa mensajes de registro detallados (DEBUG).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output = generate_study_area_map(
        output_path=args.output,
        dpi=args.dpi,
        elevation_data=None if args.no_elevation else "auto",
    )
    print(f"Mapa generado: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
