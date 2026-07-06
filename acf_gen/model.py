"""Parametric aircraft model - the single source of truth (requirements doc §3).

A declarative description of a fixed-wing aircraft: lifting surfaces, bodies,
mass, propulsion, gear, control. The flight-model emitter reads THIS and writes
the .acf; nothing downstream invents numbers that aren't derived from here.

COORDINATE FRAME (matches X-Plane and the OBJ8 visual so they share one space):
    +X = right wing      +Y = up      +Z = aft (toward tail)
    nose points toward -Z. Units: METERS, kilograms, degrees.
    The datum (origin) is the aircraft reference point; pick the nose tip or a
    fixed structural point and keep the Fusion model on the same origin.

Mass properties (CG, inertia) are COMPUTED from the mass components here, never
typed in - so they can't contradict the layout (principle #3 / "derive, don't
declare").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Vec3 = tuple[float, float, float]

SurfaceRole = Literal["main_wing", "h_stab", "v_stab", "canard", "custom"]
ControlType = Literal["aileron", "elevator", "rudder", "flap", "spoiler", "all_moving"]
EngineType = Literal["electric", "piston", "turboprop", "turbojet", "hybrid"]


# --------------------------------------------------------------------- geometry
@dataclass
class Section:
    """One spanwise station of a lifting surface."""
    y: float                      # spanwise distance from the surface root (m)
    chord: float                  # local chord (m)
    airfoil: str                  # .afl filename (stock or custom)
    incidence: float = 0.0        # local twist/incidence (deg, +nose-up)


@dataclass
class ControlSurface:
    type: ControlType
    chord_fraction: float          # 0-1; 1.0 = all-moving / full-flying surface
    span_range: tuple[float, float]  # (inboard, outboard) as fraction of semi-span
    deflection_limits: tuple[float, float]  # (min, max) deg
    hinge_axis: Optional[Vec3] = None        # pivot line for all-moving surfaces
    # Which flight axes this surface drives, and how. Each entry is
    # (axis, mode) where axis in {roll,pitch,yaw,flap} and mode in
    # {differential, symmetric, direct}. Supports over-actuated mixing.
    drives: list = field(default_factory=list)


@dataclass
class LiftingSurface:
    role: SurfaceRole
    root_position: Vec3            # root LE position in the aircraft frame (m)
    sections: list[Section]        # root -> tip, >= 2
    symmetric: bool = True         # auto-mirror left/right (wings, h-stab)
    control: Optional[ControlSurface] = None

    @property
    def semi_span(self) -> float:
        return max(s.y for s in self.sections)

    @property
    def area(self) -> float:
        """Planform area of ONE side (trapezoidal sum over sections)."""
        a = 0.0
        for i in range(len(self.sections) - 1):
            s0, s1 = self.sections[i], self.sections[i + 1]
            a += 0.5 * (s0.chord + s1.chord) * (s1.y - s0.y)
        return a

    @property
    def mac(self) -> float:
        """Mean aerodynamic chord (area-weighted), one side."""
        num = den = 0.0
        for i in range(len(self.sections) - 1):
            s0, s1 = self.sections[i], self.sections[i + 1]
            seg_area = 0.5 * (s0.chord + s1.chord) * (s1.y - s0.y)
            num += seg_area * 0.5 * (s0.chord + s1.chord)
            den += seg_area
        return num / den if den else 0.0


@dataclass
class BodyStation:
    x: float                       # station position along the body axis (m)
    width: float                   # full width (m)
    height: float                  # full height (m)


@dataclass
class Body:
    name: str
    stations: list[BodyStation]    # nose -> tail


# --------------------------------------------------------------- mass / power
@dataclass
class MassComponent:
    name: str
    mass: float                    # kg
    position: Vec3                 # CG of this component (m)
    is_payload: bool = False       # True = removable mission payload (not empty weight)


@dataclass
class Propeller:
    diameter: float                # m
    num_blades: int = 2
    pitch: Literal["fixed", "variable"] = "fixed"
    direction: int = 1             # +1 = clockwise seen from behind


@dataclass
class Propulsion:
    type: EngineType
    position: Vec3                 # thrust point (m)
    max_power_kw: float = 0.0      # for power engines (electric/piston/turboprop)
    max_thrust_n: float = 0.0      # for jets
    propeller: Optional[Propeller] = None
    battery_wh: float = 0.0        # battery capacity (Wh) for electric
    design_rpm: float = 0.0


@dataclass
class GearLeg:
    name: str
    position: Vec3                 # wheel contact point (m)
    leg_length: float              # m (strut travel reference)
    tire_radius: float             # m
    steerable: bool = False
    retractable: bool = False


@dataclass
class OperatingLimits:
    vne_kts: float = 0.0
    vno_kts: float = 0.0
    vs_kts: float = 0.0
    vfe_kts: float = 0.0
    g_pos: float = 4.0
    g_neg: float = -2.0
    ceiling_m: float = 0.0


# ------------------------------------------------------------------- aircraft
@dataclass
class AircraftSpec:
    name: str
    author: str = ""
    description: str = ""
    surfaces: list[LiftingSurface] = field(default_factory=list)
    bodies: list[Body] = field(default_factory=list)
    masses: list[MassComponent] = field(default_factory=list)
    propulsion: list[Propulsion] = field(default_factory=list)
    gear: list[GearLeg] = field(default_factory=list)
    limits: OperatingLimits = field(default_factory=OperatingLimits)

    # ---- derived mass properties (computed, never declared) ----
    @property
    def empty_mass(self) -> float:
        """Empty weight = everything EXCEPT removable mission payload."""
        return sum(m.mass for m in self.masses if not m.is_payload)

    @property
    def payload_mass(self) -> float:
        return sum(m.mass for m in self.masses if m.is_payload)

    @property
    def total_mass(self) -> float:
        """Operating (loaded) weight = empty + payload."""
        return sum(m.mass for m in self.masses)

    @property
    def cg(self) -> Vec3:
        # Operating CG (loaded with payload) — this is the flight design point.
        M = self.total_mass
        if M <= 0:
            return (0.0, 0.0, 0.0)
        cx = sum(m.mass * m.position[0] for m in self.masses) / M
        cy = sum(m.mass * m.position[1] for m in self.masses) / M
        cz = sum(m.mass * m.position[2] for m in self.masses) / M
        return (cx, cy, cz)

    @property
    def inertia(self) -> tuple[float, float, float]:
        """Principal moments of inertia about the CG (kg·m²) from point masses,
        returned as the tensor diagonal (Ixx, Iyy, Izz) in THIS model frame
        (x=right/lateral, y=up/vertical, z=aft/longitudinal). The axis→rotation
        mapping in this frame is NOT the X-Plane letter order:
            Ixx (about lateral x)      → PITCH inertia
            Iyy (about vertical y)     → YAW   inertia
            Izz (about longitudinal z) → ROLL  inertia
        X-Plane's _Jxx/_Jyy/_Jzz_unitmass are ROLL/PITCH/YAW per unit mass, so the
        emitter maps by PHYSICAL axis: Jxx←Izz, Jyy←Ixx, Jzz←Iyy (each ÷ total mass)."""
        cx, cy, cz = self.cg
        ixx = iyy = izz = 0.0
        for m in self.masses:
            x, y, z = m.position[0] - cx, m.position[1] - cy, m.position[2] - cz
            ixx += m.mass * (y * y + z * z)   # about lateral x      -> PITCH
            iyy += m.mass * (x * x + z * z)   # about vertical y     -> YAW
            izz += m.mass * (x * x + y * y)   # about longitudinal z -> ROLL
        return (ixx, iyy, izz)

    @property
    def main_wing(self) -> Optional[LiftingSurface]:
        for s in self.surfaces:
            if s.role == "main_wing":
                return s
        return None


if __name__ == "__main__":
    # Placeholder Peregrine-ish test article (sane numbers, NOT the real CAD).
    # Replaced by measurements off the Fusion model once it exists.
    WING = "NACA 2412 (popular).afl"
    TAIL = "NACA 0009 (symmetrical).afl"
    spec = AircraftSpec(
        name="Peregrine Placeholder",
        author="Idriss",
        description="Engine certification test article - placeholder geometry",
        surfaces=[
            LiftingSurface("main_wing", (0.0, 0.10, 0.55), [
                Section(0.0, 0.34, WING, 2.0),
                Section(1.30, 0.24, WING, 1.0),
            ]),
            LiftingSurface("h_stab", (0.0, 0.05, 1.55), [
                Section(0.0, 0.20, TAIL), Section(0.45, 0.14, TAIL),
            ], control=ControlSurface("all_moving", 1.0, (0.0, 1.0), (-15, 15))),
            LiftingSurface("v_stab", (0.0, 0.0, 1.55), [
                Section(0.0, 0.22, TAIL), Section(0.40, 0.14, TAIL),
            ], symmetric=False,
               control=ControlSurface("all_moving", 1.0, (0.0, 1.0), (-20, 20))),
        ],
        bodies=[Body("fuselage", [
            BodyStation(0.0, 0.05, 0.05), BodyStation(0.35, 0.18, 0.20),
            BodyStation(1.0, 0.12, 0.14), BodyStation(1.75, 0.03, 0.04),
        ])],
        masses=[
            MassComponent("structure", 3.2, (0.0, 0.05, 0.75)),
            MassComponent("battery", 2.0, (0.0, 0.02, 0.45)),
            MassComponent("avionics+payload", 1.8, (0.0, 0.05, 0.30)),
        ],
        propulsion=[Propulsion("electric", (0.0, 0.05, 0.0), max_power_kw=2.5,
                               propeller=Propeller(0.46, 2), battery_wh=600,
                               design_rpm=7000)],
        gear=[
            GearLeg("nose", (0.0, -0.18, 0.25), 0.18, 0.05, steerable=True),
            GearLeg("left", (-0.35, -0.18, 0.70), 0.18, 0.06),
            GearLeg("right", (0.35, -0.18, 0.70), 0.18, 0.06),
        ],
        limits=OperatingLimits(vne_kts=90, vno_kts=70, vs_kts=22, g_pos=6, g_neg=-3),
    )
    jx, jy, jz = spec.inertia
    print(f"name        : {spec.name}")
    print(f"empty mass  : {spec.empty_mass:.2f} kg ({spec.empty_mass*2.2046:.2f} lb)")
    print(f"CG (x,y,z)  : ({spec.cg[0]:.3f}, {spec.cg[1]:.3f}, {spec.cg[2]:.3f}) m")
    print(f"inertia kgm2: Jxx={jx:.3f} Jyy={jy:.3f} Jzz={jz:.3f}")
    w = spec.main_wing
    print(f"main wing   : span={2*w.semi_span:.2f} m  area={2*w.area:.3f} m²  MAC={w.mac:.3f} m")
    print(f"surfaces    : {[s.role for s in spec.surfaces]}")
    print(f"all-moving  : {[s.role for s in spec.surfaces if s.control and s.control.type=='all_moving']}")
