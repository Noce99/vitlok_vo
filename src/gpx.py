"""Ground truth: reading a GPS track and putting it in the same frame as an estimate.

``gpx_evaluation.py`` accepts ground truth in three forms.

**A GPX file.** Parsed for ``<trkpt lat lon><time>``, cropped to the window the
video covers, and projected from WGS84 into a local metric frame. Two projections
are available and they are not interchangeable:

* with a georeferenced map (``--map``/``--world``), points are projected into the
  map's own CRS (``--epsg``) so that estimate, ground truth and map pixels all
  share one coordinate system;
* without a map, an azimuthal-equidistant projection is anchored at the track's
  first point, which preserves distance from that point exactly and is the right
  choice when all we want is metres.

**A GPS telemetry CSV** (``--csv``), as extracted from GoPro GPMF metadata (e.g.
a ``*-GPS5.csv`` file): a ``date`` column of ISO 8601 timestamps and
``GPS (Lat.) [deg]`` / ``GPS (Long.) [deg]`` columns. Cropped and projected the
same way as a GPX track.

**A pre-computed trajectory file** (``--gt-trajectory``), four columns
``time x y z`` already in local metres, as produced for synthetic sequences.

Either way the result is an ``(N, 3)`` array of ``[time, x, y]`` with **x east
and y north**, which is what :mod:`src.alignment` and :mod:`src.metrics` consume.

That axis order matters and is easy to get wrong. Alignment applies a rotation
and a translation but never a reflection, so ground truth handed over in the
opposite handedness cannot be aligned at all -- the metrics come out several
times too large rather than failing outright. Simulator ground truth is commonly
**NED** (x north, y east, z down), which is the opposite handedness to ENU; pass
``axes="ned"`` for those files and they will be swapped on the way in.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

#: WGS84, what a GPS receiver reports and what GPX files store.
WGS84_EPSG = "4326"


def read_gpx(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a GPX file into latitudes, longitudes and Unix timestamps.

    The XML namespace is ignored, since GPX 1.0 and 1.1 differ in it and both
    appear in the wild.

    Returns:
        ``(lats, lons, times)``; *times* are seconds since the Unix epoch, NaN
        for points that carry no ``<time>``.

    Raises:
        ValueError: if the file contains no track points.
    """
    root = ElementTree.parse(Path(path)).getroot()

    lats: list[float] = []
    lons: list[float] = []
    times: list[float] = []
    for element in root.iter():
        if not element.tag.endswith("trkpt"):
            continue
        lats.append(float(element.attrib["lat"]))
        lons.append(float(element.attrib["lon"]))
        stamp = float("nan")
        for child in element:
            if child.tag.endswith("time") and child.text:
                stamp = _parse_time(child.text.strip())
        times.append(stamp)

    if not lats:
        raise ValueError(f"{path}: no <trkpt> elements found")
    return np.asarray(lats), np.asarray(lons), np.asarray(times)


def read_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a GoPro-style GPS telemetry CSV into latitudes, longitudes and Unix timestamps.

    Expects a ``date`` column (ISO 8601) and columns whose names contain "lat"
    and "long" -- ``GPS (Lat.) [deg]`` / ``GPS (Long.) [deg]``, as written by
    GPMF extractors such as ``gopro2gpx``. Other columns (2D/3D speed, fix,
    precision, ...) are ignored.

    Returns:
        ``(lats, lons, times)``; *times* are seconds since the Unix epoch, NaN
        for rows with no parseable ``date``.

    Raises:
        ValueError: if the file has no data rows or lacks a lat/lon column.
    """
    import csv as csv_module

    with Path(path).open(newline="") as f:
        reader = csv_module.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        lat_col = _find_column(reader.fieldnames, "lat", path)
        lon_col = _find_column(reader.fieldnames, "long", path)

        lats: list[float] = []
        lons: list[float] = []
        times: list[float] = []
        for row in reader:
            lats.append(float(row[lat_col]))
            lons.append(float(row[lon_col]))
            stamp = (row.get("date") or "").strip()
            times.append(_parse_time(stamp) if stamp else float("nan"))

    if not lats:
        raise ValueError(f"{path}: no data rows found")
    return np.asarray(lats), np.asarray(lons), np.asarray(times)


def _find_column(fieldnames: list[str], needle: str, path: Path) -> str:
    """The first CSV column whose name contains *needle*, case-insensitively."""
    for name in fieldnames:
        if needle in name.lower():
            return name
    raise ValueError(f"{path}: no column containing {needle!r}; found {fieldnames}")


def _parse_time(text: str) -> float:
    """Parse a GPX timestamp into Unix seconds."""
    cleaned = text.replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(cleaned)
    except ValueError:
        return float("nan")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def transform_coordinates(
    lats: np.ndarray, lons: np.ndarray, from_epsg: str, to_epsg: str
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject latitude/longitude into *to_epsg*.

    Returns:
        ``(x, y)`` easting and northing in the target CRS.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(
        f"EPSG:{from_epsg}", f"EPSG:{to_epsg}", always_xy=True
    )
    x, y = transformer.transform(lons, lats)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def project_local_metres(
    lats: np.ndarray, lons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project to metres with an azimuthal-equidistant frame at the first point.

    Distances measured from that first point are exact, which is what matters
    when comparing against a trajectory whose own origin is its first pose.
    """
    from pyproj import CRS, Transformer

    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lats[0]} +lon_0={lons[0]} +datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(
        CRS.from_epsg(int(WGS84_EPSG)), local, always_xy=True
    )
    x, y = transformer.transform(lons, lats)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def load_ground_truth(
    gpx_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    gt_trajectory_path: Optional[Path] = None,
    start_time: Optional[float] = None,
    duration_s: Optional[float] = None,
    target_epsg: Optional[str] = None,
    gps_epsg: str = WGS84_EPSG,
    axes: str = "enu",
) -> np.ndarray:
    """Load ground truth as ``(N, 3)`` of ``[time, x, y]`` in metres.

    Args:
        gpx_path: GPX track to read.
        csv_path: GPS telemetry CSV to read (see :func:`read_csv`), as an
            alternative to *gpx_path*.
        gt_trajectory_path: Alternative to both -- a ``time x y z`` text file
            already in local metres. Takes precedence if more than one is given.
        start_time: Unix timestamp of the video's first frame. Points before it
            are dropped; without it the whole track is used.
        duration_s: Video duration; points after ``start_time + duration_s`` are
            dropped.
        target_epsg: Project into this CRS (use the map's, when there is a map).
            When ``None``, an azimuthal-equidistant local frame is used instead.
        gps_epsg: CRS the GPX/CSV coordinates are in; virtually always WGS84.
        axes: Axis convention of *gt_trajectory_path*: ``"enu"`` (x east,
            y north) or ``"ned"`` (x north, y east), the latter being what
            simulators such as TartanAir export. Ignored for GPX/CSV input,
            which is always projected to ENU.

    Raises:
        ValueError: if no source is given, or the crop leaves nothing.
    """
    if gt_trajectory_path is not None:
        data = np.loadtxt(gt_trajectory_path, ndmin=2)
        if data.shape[1] < 3:
            raise ValueError(
                f"{gt_trajectory_path}: expected columns 'time x y [z]', "
                f"found {data.shape[1]}"
            )
        if axes not in ("enu", "ned"):
            raise ValueError(f"axes must be 'enu' or 'ned', got {axes!r}")
        data = data - data[0]
        # NED is the opposite handedness to ENU: swapping the two horizontal
        # columns converts between them.
        columns = [0, 1, 2] if axes == "enu" else [0, 2, 1]
        return data[:, columns]

    if gpx_path is not None:
        track_path, lats, lons, times = gpx_path, *read_gpx(gpx_path)
    elif csv_path is not None:
        track_path, lats, lons, times = csv_path, *read_csv(csv_path)
    else:
        raise ValueError("give --gpx, --csv or --gt-trajectory")

    timed = ~np.isnan(times)
    if start_time is not None:
        if not timed.any():
            raise ValueError(
                f"{track_path}: --start-time was given but no track point has a "
                "time to crop against"
            )
        lats, lons, times = lats[timed], lons[timed], times[timed]
        keep = times >= start_time
        if duration_s is not None:
            keep &= times <= start_time + duration_s
        if not keep.any():
            raise ValueError(
                f"{track_path}: no track points fall inside the video's window "
                f"[{start_time}, {start_time + (duration_s or 0)}]. "
                "Is --start-time correct, and in UTC?"
            )
        lats, lons, times = lats[keep], lons[keep], times[keep]
    elif timed.any():
        lats, lons, times = lats[timed], lons[timed], times[timed]
    else:
        # Untimed track: fall back to a nominal 1 Hz so the series is ordered.
        times = np.arange(len(lats), dtype=float)

    if target_epsg is not None:
        x, y = transform_coordinates(lats, lons, gps_epsg, target_epsg)
    else:
        x, y = project_local_metres(lats, lons)

    return np.column_stack([times.astype(np.float64), x, y])


def video_duration_s(path: Path) -> float:
    """Duration of a video in seconds, from its frame count and rate."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        capture.release()
    return float(frames / fps) if fps else 0.0
