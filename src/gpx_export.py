"""Writing an estimated trajectory back out as GPX.

The trajectory is in local metres; a GPX file wants latitude and longitude. The
inverse of the projection used to read the ground truth gets us there: an
azimuthal-equidistant frame anchored at a known starting fix, so distances from
that anchor are preserved exactly.

Points are thinned to at most one per second. GPX is a format for GPS logs, and
tools that read it expect roughly that rate; writing 30 points per second would
produce a file most viewers choke on for no added information.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

GPX_NAMESPACE = "http://www.topografix.com/GPX/1/1"


def local_metres_to_latlon(
    x: np.ndarray, y: np.ndarray, origin_lat: float, origin_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local metric offsets back to WGS84 latitude and longitude.

    Args:
        x, y: Eastward and northward offsets in metres from the origin.
        origin_lat, origin_lon: The anchor point, in degrees.
    """
    from pyproj import CRS, Transformer

    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(local, CRS.from_epsg(4326), always_xy=True)
    lons, lats = transformer.transform(x, y)
    return np.asarray(lats), np.asarray(lons)


def trajectory_to_gpx(
    txyz: np.ndarray,
    destination: Path,
    origin_lat: float,
    origin_lon: float,
    start_time: Optional[float] = None,
    name: str = "video_to_trajectory estimate",
    max_rate_hz: float = 1.0,
) -> Path:
    """Write *txyz* to *destination* as a GPX 1.1 track.

    Args:
        txyz: ``(N, 4+)`` array ``[time, x, y, z]`` in local metres.
        destination: Output ``.gpx`` path.
        origin_lat, origin_lon: WGS84 position of the trajectory's first point.
        start_time: Unix timestamp of the first pose. When given, absolute
            ``<time>`` elements are written.
        name: Track name recorded in the file.
        max_rate_hz: Points are thinned to at most this rate.

    Returns:
        *destination*.
    """
    keep = _thin(txyz[:, 0], max_rate_hz)
    sampled = txyz[keep]
    lats, lons = local_metres_to_latlon(sampled[:, 1], sampled[:, 2],
                                        origin_lat, origin_lon)

    ElementTree.register_namespace("", GPX_NAMESPACE)
    gpx = ElementTree.Element(f"{{{GPX_NAMESPACE}}}gpx",
                              {"version": "1.1", "creator": "video_to_trajectory"})
    track = ElementTree.SubElement(gpx, f"{{{GPX_NAMESPACE}}}trk")
    ElementTree.SubElement(track, f"{{{GPX_NAMESPACE}}}name").text = name
    segment = ElementTree.SubElement(track, f"{{{GPX_NAMESPACE}}}trkseg")

    has_elevation = sampled.shape[1] > 3
    for i in range(len(sampled)):
        point = ElementTree.SubElement(
            segment, f"{{{GPX_NAMESPACE}}}trkpt",
            {"lat": f"{lats[i]:.8f}", "lon": f"{lons[i]:.8f}"},
        )
        if has_elevation:
            ElementTree.SubElement(
                point, f"{{{GPX_NAMESPACE}}}ele"
            ).text = f"{sampled[i, 3]:.2f}"
        if start_time is not None:
            moment = datetime.fromtimestamp(start_time + sampled[i, 0], timezone.utc)
            ElementTree.SubElement(
                point, f"{{{GPX_NAMESPACE}}}time"
            ).text = moment.strftime("%Y-%m-%dT%H:%M:%SZ")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = ElementTree.ElementTree(gpx)
    ElementTree.indent(tree, space="  ")
    tree.write(destination, encoding="UTF-8", xml_declaration=True)
    return destination


def _thin(times: np.ndarray, max_rate_hz: float) -> np.ndarray:
    """Indices keeping at most *max_rate_hz* samples per second."""
    if max_rate_hz <= 0 or len(times) < 2:
        return np.arange(len(times))
    minimum_gap = 1.0 / max_rate_hz
    keep = [0]
    last = times[0]
    for i in range(1, len(times)):
        if times[i] - last >= minimum_gap:
            keep.append(i)
            last = times[i]
    if keep[-1] != len(times) - 1:
        keep.append(len(times) - 1)
    return np.asarray(keep)
