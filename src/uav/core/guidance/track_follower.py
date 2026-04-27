"""L1-style track follower — the "religiously-follow-the-ribbon" engine.

The ribbon is a polyline of PathPoints. At each tick this module:

  1. Projects the plane's current (lat, lon) onto the polyline → finds the
     active segment (i, i+1), along-track distance, and SIGNED perpendicular
     cross-track distance.
  2. Advances the active segment when the plane passes the endpoint.
  3. Computes an L1 lookahead point: a point on the ribbon a tunable
     distance AHEAD of the plane's projection.
  4. Returns a commanded heading = bearing from plane → lookahead point.

Why L1 (Park/Deyst/How, MIT 2004) instead of heading-to-point or
proportional cross-track correction:

  • **Cross-track capture is automatic.** If the plane is off-track, the
    lookahead sits on the line, so the bearing to it has a lateral
    component that pulls the plane back.
  • **Turn anticipation is automatic.** If the ribbon bends ahead, the
    lookahead moves around the bend before the plane reaches it, so the
    plane starts turning early. No explicit turn-anticipation distance
    needed.
  • **Past-waypoint fallback is automatic.** The lookahead advances along
    the polyline; there's no "past this aim_at" degenerate case. The
    plane always steers toward something ahead, never toward a point
    behind it.

Replaces: the aim_at bearing-to-point logic in flight_engine._resolve()
and its past-aim fallback guard. Those become unnecessary once heading
is driven by track projection.

Usage:

    follower = TrackFollower(ribbon.points)
    state = follower.update(telemetry)
    commanded_heading = state.heading_deg

The follower is stateless except for a scan-start-index cache
(`_last_seg_idx`) used to avoid O(N) projection work when the ribbon
has many points. Projection math is flat-earth (local tangent plane
around each segment's midpoint) — exact to millimetres at <200 nm,
which is well beyond any SF50 hop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from uav.nav.flight_plan_v2 import PathPoint
from uav.nav.geo import bearing_deg
from uav.sim.types import Telemetry


# ── L1 tuning ────────────────────────────────────────────────────────
# L1_PERIOD governs lookahead time. Ardupilot default is ~18s. For a
# small jet (SF50) that maneuvers faster than a GA piston we want
# shorter lookahead so the plane tracks tight turns without cutting
# corners. 12 s at 150 kts = 0.5 nm lookahead — approximately one
# segment in the current ribbon discretization.
L1_PERIOD_SEC = 12.0
L1_MIN_NM = 0.3     # floor (taxi / flare at low speed)
L1_MAX_NM = 3.0     # ceiling (guards against wild lookahead if GS is noisy)


@dataclass
class TrackState:
    """One-tick output of the follower. Consumed by flight_engine and
    also published in telemetry so the app can render cross_track_nm."""
    heading_deg: float              # commanded heading (0..360)
    cross_track_nm: float           # signed; positive = plane right of track
    along_track_nm: float           # cumulative along-track from ribbon start
    segment_idx: int                # active segment (points[i] → points[i+1])
    segment_progress: float         # 0..1 within active segment
    lookahead_lat: float            # for map display / debug
    lookahead_lon: float
    ribbon_length_nm: float         # total polyline length


# ── Flat-earth projection helpers ───────────────────────────────────

def _project_on_segment(
    plane_lat: float,
    plane_lon: float,
    p0: PathPoint,
    p1: PathPoint,
) -> Tuple[float, float, float]:
    """Project (plane_lat, plane_lon) onto segment p0→p1.

    Returns (along_nm, cross_nm, seg_len_nm):
      along_nm: signed distance from p0 along segment direction.
        Can be negative (plane behind p0) or > seg_len (past p1).
      cross_nm: signed perpendicular (positive = plane to the RIGHT of
        track when facing p0→p1).
      seg_len_nm: length of segment.
    """
    mean_lat_rad = math.radians((p0.lat + p1.lat) / 2.0)
    cos_lat = math.cos(mean_lat_rad)

    # Segment vector (east, north) in nm
    seg_e = (p1.lon - p0.lon) * 60.0 * cos_lat
    seg_n = (p1.lat - p0.lat) * 60.0
    seg_len = math.hypot(seg_e, seg_n)

    if seg_len < 1e-9:
        return 0.0, 0.0, 0.0

    # Unit vector along the segment
    ue = seg_e / seg_len
    un = seg_n / seg_len

    # Plane's vector from p0 (east, north) in nm
    dp_e = (plane_lon - p0.lon) * 60.0 * cos_lat
    dp_n = (plane_lat - p0.lat) * 60.0

    # Along-track: dot product with unit-along vector
    along = dp_e * ue + dp_n * un
    # Cross-track: signed perpendicular. Right-of-track vector is
    # (un, -ue) (rotate along-track -90°). Dot product with plane vector:
    cross = dp_e * un - dp_n * ue

    return along, cross, seg_len


def _interp_on_ribbon(
    points: List[PathPoint],
    cum_lengths: List[float],
    target_along_nm: float,
) -> Tuple[float, float]:
    """Return (lat, lon) at the given absolute along-track distance.

    Clamps to endpoints if out of range. Uses linear interpolation in
    lat/lon space — fine at segment scales of a few nm.
    """
    if target_along_nm <= 0.0:
        return points[0].lat, points[0].lon
    total = cum_lengths[-1]
    if target_along_nm >= total:
        return points[-1].lat, points[-1].lon

    # Binary search would be faster; linear is fine at N≈20
    for i in range(len(points) - 1):
        a0 = cum_lengths[i]
        a1 = cum_lengths[i + 1]
        if a0 <= target_along_nm <= a1:
            seg_len = a1 - a0
            if seg_len < 1e-9:
                return points[i].lat, points[i].lon
            t = (target_along_nm - a0) / seg_len
            lat = points[i].lat + t * (points[i + 1].lat - points[i].lat)
            lon = points[i].lon + t * (points[i + 1].lon - points[i].lon)
            return lat, lon
    # Fallback — shouldn't reach here with valid cum_lengths
    return points[-1].lat, points[-1].lon


class TrackFollower:
    """Stateful L1 follower over a static ribbon polyline.

    One instance per flight (rebuild when the ribbon is regenerated).
    Thread-safe for single-writer access from the flight engine tick.
    """

    def __init__(self, points: List[PathPoint]) -> None:
        if len(points) < 2:
            raise ValueError(
                "TrackFollower needs at least 2 points to define a segment"
            )
        self.points = points
        self._last_seg_idx = 0

        # Precompute cumulative arc lengths so lookahead interpolation
        # is O(N) once, not per-tick.
        self.cum_lengths: List[float] = [0.0]
        for i in range(len(points) - 1):
            _, _, seg_len = _project_on_segment(
                points[i + 1].lat, points[i + 1].lon,
                points[i], points[i + 1],
            )
            self.cum_lengths.append(self.cum_lengths[-1] + seg_len)

    # ── Segment search ──────────────────────────────────────────────

    def _find_segment(self, lat: float, lon: float) -> int:
        """Return the active segment index. Monotonically advances:
        once past a segment, never goes back (GPS jitter or a huge lateral
        offset won't regress the ribbon progress).

        Scans forward from last known segment; advances while plane's
        projection is past the segment end. Stops at the first segment
        where the plane is within OR before (i.e. hasn't reached yet).
        """
        idx = self._last_seg_idx
        last_idx = len(self.points) - 2  # last valid segment index

        while idx < last_idx:
            p0 = self.points[idx]
            p1 = self.points[idx + 1]
            along, _cross, seg_len = _project_on_segment(lat, lon, p0, p1)

            if seg_len < 1e-9:
                idx += 1  # degenerate segment, skip
                continue

            if along >= seg_len:
                # Past this segment's end — advance
                idx += 1
                continue

            # Plane is within (0 ≤ along < seg_len) or before (along < 0).
            # Either way, this is our active segment.
            return idx

        # Fall-through: past the last segment. Pin to last valid index so
        # the follower still has something to project on and L1 lookahead
        # clamps to the endpoint.
        return last_idx

    # ── Main per-tick update ────────────────────────────────────────

    def update(self, telemetry: Telemetry) -> TrackState:
        """Compute commanded heading + cross/along telemetry."""
        # L1 lookahead scales with groundspeed. We use airspeed as a
        # proxy because telemetry doesn't always carry groundspeed;
        # airspeed ≈ GS in calm air and the error gets absorbed by the
        # next tick's projection.
        gs_kts = max(0.0, telemetry.airspeed_kts)
        gs_nm_per_sec = gs_kts / 3600.0
        L1_nm = max(L1_MIN_NM, min(L1_MAX_NM, gs_nm_per_sec * L1_PERIOD_SEC))

        # Find/advance active segment
        seg_idx = self._find_segment(telemetry.lat_deg, telemetry.lon_deg)
        self._last_seg_idx = seg_idx

        p0 = self.points[seg_idx]
        p1 = self.points[seg_idx + 1]
        along, cross, seg_len = _project_on_segment(
            telemetry.lat_deg, telemetry.lon_deg, p0, p1,
        )

        # Absolute along-track on the full polyline. Clamp to segment
        # bounds for the absolute-along computation (we don't want a
        # negative "along" to shove the lookahead backward into a prior
        # segment — we always look AHEAD on the ribbon).
        along_in_seg = max(0.0, min(seg_len, along))
        absolute_along = self.cum_lengths[seg_idx] + along_in_seg

        # L1 lookahead point: L1_nm ahead of projection
        la_along = absolute_along + L1_nm
        total_len = self.cum_lengths[-1]

        if la_along >= total_len:
            # Lookahead has clamped to the last ribbon point. Using
            # bearing-to-last-point would command a U-turn if the plane
            # is past the endpoint. Hold the last segment's direction
            # instead — that is always "straight ahead" relative to the
            # ribbon's final heading (e.g. runway axis for rollout).
            last_p0 = self.points[-2]
            last_p1 = self.points[-1]
            heading = bearing_deg(last_p0.lat, last_p0.lon,
                                  last_p1.lat, last_p1.lon)
            la_lat, la_lon = last_p1.lat, last_p1.lon
        else:
            la_lat, la_lon = _interp_on_ribbon(
                self.points, self.cum_lengths, la_along,
            )
            # Commanded heading = bearing from plane to lookahead
            heading = bearing_deg(
                telemetry.lat_deg, telemetry.lon_deg, la_lat, la_lon,
            )

        progress = along_in_seg / seg_len if seg_len > 1e-9 else 1.0

        return TrackState(
            heading_deg=heading,
            cross_track_nm=cross,
            along_track_nm=absolute_along,
            segment_idx=seg_idx,
            segment_progress=progress,
            lookahead_lat=la_lat,
            lookahead_lon=la_lon,
            ribbon_length_nm=self.cum_lengths[-1],
        )

    # ── Convenience ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset scan cache. Call on flight reset if the same follower
        instance is reused (we don't currently — we rebuild per flight)."""
        self._last_seg_idx = 0
