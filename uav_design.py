import numpy as np
import scipy.optimize
import math

class Config:
    def __init__(self, payload_kg=2.0, battery_wh=310, max_span_m=2.7,
                 cruise_ms=16.0, mission_km=55, endurance_hr=2.0, flying_wing=False):
        self.payload_kg   = float(payload_kg)
        self.battery_wh   = float(battery_wh)
        self.max_span_m   = float(max_span_m)
        self.cruise_ms    = float(cruise_ms)
        self.mission_km   = float(mission_km)
        self.endurance_hr = float(endurance_hr)
        self.flying_wing  = bool(flying_wing)

class Design:
    def __init__(self, wingspan, aspect_ratio, taper_ratio, t_c_ratio,
                 twist_deg=0.0, winglet_pres=0.0):
        self.wingspan     = float(wingspan)
        self.aspect_ratio = float(aspect_ratio)
        self.taper_ratio  = float(taper_ratio)
        self.t_c_ratio    = float(t_c_ratio)
        self.twist_deg    = float(twist_deg)
        self.winglet_pres = float(winglet_pres)


# ── AIRFOIL DATABASE ─────────────────────────────────────────────────────────
# Each entry contains real aerodynamic properties for feasibility checking
AIRFOIL_DB = {
    "naca4412":  {"name": "NACA 4412",  "Cl_max": 1.60, "Cd_min": 0.0080, "Re_min": 200000,
                  "best_use": "General purpose UAV, good all-rounder"},
    "naca2412":  {"name": "NACA 2412",  "Cl_max": 1.45, "Cd_min": 0.0075, "Re_min": 150000,
                  "best_use": "Low camber, good at higher speeds"},
    "naca0012":  {"name": "NACA 0012",  "Cl_max": 1.20, "Cd_min": 0.0070, "Re_min": 100000,
                  "best_use": "Symmetric — aerobatics, tail surfaces, flying wing with twist"},
    "naca6409":  {"name": "NACA 6409",  "Cl_max": 1.85, "Cd_min": 0.0090, "Re_min": 300000,
                  "best_use": "High lift, slow-speed UAV, heavy payload"},
    "naca23012": {"name": "NACA 23012", "Cl_max": 1.60, "Cd_min": 0.0078, "Re_min": 200000,
                  "best_use": "Reflex camber — flying wing, naturally stable"},
    "naca4415":  {"name": "NACA 4415",  "Cl_max": 1.65, "Cd_min": 0.0085, "Re_min": 250000,
                  "best_use": "Thick — structural strength, large wingspan builds"},
    "clark_y":   {"name": "Clark Y",    "Cl_max": 1.55, "Cd_min": 0.0082, "Re_min": 150000,
                  "best_use": "Classic UAV flat-bottom, easy to hand-build"},
    "e387":      {"name": "Eppler E387","Cl_max": 1.50, "Cd_min": 0.0060, "Re_min": 100000,
                  "best_use": "High-efficiency glider, long-range UAV"},
    "s1223":     {"name": "Selig S1223","Cl_max": 2.20, "Cd_min": 0.0120, "Re_min": 200000,
                  "best_use": "Extreme high-lift, very slow / heavy payload"},
    "naca63412": {"name": "NACA 63-412","Cl_max": 1.60, "Cd_min": 0.0055, "Re_min": 500000,
                  "best_use": "Laminar flow — high-speed efficient cruise"},
}


def check_feasibility(cfg, airfoil_key="naca4412"):
    """
    Physics-based feasibility check BEFORE running optimizer.
    Returns (is_feasible: bool, warnings: list[str], errors: list[str])
    """
    warnings = []
    errors   = []

    af = AIRFOIL_DB.get(airfoil_key, AIRFOIL_DB["naca4412"])
    Cl_max   = af["Cl_max"]
    Re_min   = af["Re_min"]

    rho      = 1.225   # kg/m³ sea level
    b        = cfg.max_span_m
    V        = cfg.cruise_ms

    # Estimate minimum wing area needed to support payload + battery + structure
    # Conservative: assume ~35% of MTOW is structure+battery
    battery_mass = cfg.battery_wh / 145.0
    est_struct   = 0.14 * (b ** 1.6) * (10.0 ** 0.4)  # rough estimate at AR=10
    mtow_est     = cfg.payload_kg + battery_mass + est_struct
    W_est        = mtow_est * 9.81

    # Min wing area to fly at cruise: L = 0.5*rho*V²*S*Cl_cruise
    # Use Cl_cruise = 0.6 * Cl_max (typical efficient cruise point)
    Cl_cruise    = 0.6 * Cl_max
    S_min_cruise = W_est / (0.5 * rho * V**2 * Cl_cruise) if V > 0 else 999
    S_min_stall  = W_est / (0.5 * rho * (V*0.7)**2 * Cl_max) if V > 0 else 999  # stall at 70% cruise
    S_min        = max(S_min_cruise, S_min_stall)

    # Max available wing area from span constraint (assume AR=5 as lower bound)
    S_max_available = b**2 / 4.0  # AR=4 gives max area for given span

    # Max AR achievable
    AR_max = b**2 / S_min if S_min > 0 else 999

    # Reynolds number at cruise (use mean chord estimate)
    nu = 1.5e-5   # kinematic viscosity of air
    chord_est = S_min / b if b > 0 else 0.1
    Re_est    = V * chord_est / nu

    # ── HARD ERRORS (physically impossible) ──────────────────────────
    if S_min > S_max_available * 1.5:
        errors.append(
            f"Wing area required ({S_min:.2f} m²) far exceeds what a {b:.1f} m wingspan can provide "
            f"({S_max_available:.2f} m² max). This aircraft cannot generate enough lift. "
            f"Try: increase wingspan, reduce payload, or use a high-lift airfoil like S1223."
        )

    if AR_max < 3.0:
        errors.append(
            f"Aspect ratio would need to be {AR_max:.1f} — below 3.0 is aerodynamically unstable "
            f"and unbuildable as a fixed-wing UAV. Increase wingspan or reduce payload drastically."
        )

    if V < 5.0:
        errors.append(
            f"Cruise speed {V:.1f} m/s is below the minimum controllable airspeed for any fixed-wing UAV (5 m/s)."
        )

    if cfg.payload_kg > 0.6 * mtow_est:
        errors.append(
            f"Payload ({cfg.payload_kg:.1f} kg) is {100*cfg.payload_kg/mtow_est:.0f}% of estimated MTOW "
            f"({mtow_est:.2f} kg). Maximum practical payload fraction is ~60%. Reduce payload or increase battery/span."
        )

    # ── SOFT WARNINGS (flyable but compromised) ───────────────────────
    if Re_est < Re_min and not errors:
        warnings.append(
            f"Estimated Reynolds number (~{Re_est:.0f}) is below {af['name']}'s minimum Re ({Re_min:,}). "
            f"At this speed/scale the airfoil may stall early. Consider E387 or NACA 0012 for low-Re flight."
        )

    if cfg.battery_wh / W_est < 20:
        warnings.append(
            f"Battery energy density is low relative to weight ({cfg.battery_wh:.0f} Wh for {W_est:.1f} N aircraft). "
            f"Estimated range may be very short. Consider increasing battery capacity."
        )

    stall_est = math.sqrt((2 * W_est) / (rho * S_min_stall * Cl_max)) if S_min_stall > 0 else 0
    if stall_est > V * 0.85:
        warnings.append(
            f"Estimated stall speed ({stall_est:.1f} m/s) is close to cruise speed ({V:.1f} m/s). "
            f"Stall margin is dangerously low. Increase wingspan or reduce cruise speed."
        )

    if cfg.flying_wing and airfoil_key not in ("naca23012", "naca0012", "naca4412"):
        warnings.append(
            f"{af['name']} has positive camber without reflex — it will cause pitch-up instability "
            f"on a flying wing. Recommended: NACA 23012 (reflex) or NACA 0012 (symmetric)."
        )

    if cfg.max_span_m > 4.0 and af["name"] in ("Clark Y", "NACA 4412"):
        warnings.append(
            f"For a {b:.1f} m wingspan, consider NACA 4415 or NACA 63-412 for better structural depth "
            f"and reduced wing flex under load."
        )

    is_feasible = len(errors) == 0
    return is_feasible, warnings, errors


def analyse(design, cfg):
    b        = min(design.wingspan, cfg.max_span_m)
    AR       = design.aspect_ratio
    taper    = design.taper_ratio
    tc       = design.t_c_ratio
    twist    = design.twist_deg
    winglets = design.winglet_pres
    fw       = cfg.flying_wing

    S          = (b ** 2) / AR if AR > 0 else 0.1
    root_chord = (2 * S) / (b * (1 + taper)) if b > 0 else 0.1
    tip_chord  = root_chord * taper

    AR_eff = AR * (1.0 + 1.9 * (tip_chord * 0.15 / b)) if winglets > 0.5 else AR

    if fw:
        C_Do         = 0.016 + (0.005 * tc) + (0.003 if winglets > 0.5 else 0.0)
        trim_penalty = 0.08
    else:
        C_Do         = 0.019 + (0.006 * tc) + (0.002 if winglets > 0.5 else 0.0)
        trim_penalty = 0.0

    k_twist = max(0.88, 1.0 - (twist * 0.015))
    C_Di    = (0.38 * k_twist) / (math.pi * AR_eff) if AR_eff > 0 else 0.4
    ld_raw  = 0.45 / (C_Do + C_Di) if (C_Do + C_Di) > 0 else 5.0
    ld      = ld_raw * (1.0 - trim_penalty)

    fw_mass_factor = 0.85 if fw else 1.0
    mass_struct    = fw_mass_factor * 0.14 * (b ** 1.6) * (AR ** 0.4) * \
                     (1.0 + (0.08 if winglets > 0.5 else 0.0))
    mass_total     = cfg.payload_kg + (cfg.battery_wh / 145.0) + mass_struct
    weight         = mass_total * 9.81

    stall_speed = math.sqrt((2 * weight) / (1.225 * S * 1.15)) if S > 0 else 10.0
    range_km    = (cfg.battery_wh * 0.65 * ld * 3.6) / weight if weight > 0 else 0
    endurance   = range_km / (cfg.cruise_ms * 3.6) if cfg.cruise_ms > 0 else 0
    safety      = max(1.0, (tc * 150.0) / (b + 0.5))

    if fw:
        stability_penalty = max(0.0, (8.0 - AR) * 1.5) if AR < 8.0 else 0.0
    else:
        stability_penalty = 0.0

    range_score     = min(25.0, (range_km / cfg.mission_km) * 20.0)
    endurance_score = min(25.0, (endurance / cfg.endurance_hr) * 20.0)
    safety_score    = min(25.0, (safety / 2.0) * 15.0)
    ld_score        = min(25.0, (ld / 18.0) * 20.0)

    score = min(100.0, max(5.0,
        range_score + endurance_score + safety_score + ld_score - stability_penalty))

    return {
        "score":        round(score, 2),
        "ld":           round(ld, 3),
        "range_km":     round(range_km, 2),
        "endurance_hr": round(endurance, 3),
        "stall_speed":  round(stall_speed, 2),
        "mass_total":   round(mass_total, 3),
        "weight_n":     round(weight, 3),
        "wingspan":     round(b, 4),
        "aspect_ratio": round(AR, 4),
        "taper_ratio":  round(taper, 4),
        "t_c":          round(tc, 4),
        "root_chord":   round(root_chord, 4),
        "tip_chord":    round(tip_chord, 4),
        "twist_deg":    round(twist, 3),
        "winglet_pres": round(winglets, 3),
        "wing_area":    round(S, 4),
        "C_Do":         round(C_Do, 5),
        "C_Di":         round(C_Di, 5),
        "flying_wing":  fw,
    }


def optimize(cfg):
    fw = cfg.flying_wing

    if fw:
        bounds = [
            (0.5,  cfg.max_span_m),
            (7.0,  22.0),
            (0.2,   0.7),
            (0.09,  0.18),
            (0.0,   3.0),
            (0.0,   1.0),
        ]
    else:
        bounds = [
            (0.5,  cfg.max_span_m),
            (4.0,  20.0),
            (0.1,   1.0),
            (0.06,  0.20),
            (0.0,   6.0),
            (0.0,   1.0),
        ]

    def objective_wrapper(flat_x):
        d = Design(
            wingspan     = flat_x[0],
            aspect_ratio = flat_x[1],
            taper_ratio  = flat_x[2],
            t_c_ratio    = flat_x[3],
            twist_deg    = flat_x[4],
            winglet_pres = flat_x[5],
        )
        return -analyse(d, cfg)["score"]

    res = scipy.optimize.differential_evolution(
        objective_wrapper, bounds, maxiter=200, popsize=15,
        tol=1e-6, seed=42, polish=True
    )

    best_x = res.x
    best_design = Design(
        wingspan     = best_x[0],
        aspect_ratio = best_x[1],
        taper_ratio  = best_x[2],
        t_c_ratio    = best_x[3],
        twist_deg    = best_x[4],
        winglet_pres = best_x[5],
    )
    return best_design, -res.fun
