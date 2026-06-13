import numpy as np
import scipy.optimize
import math

XFOIL_AVAILABLE = False  # Not available on Render free tier

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

AIRFOIL_DB = {
    "naca4412":  {"name": "NACA 4412",  "Cl_max": 1.60, "Cd_min": 0.0080, "Re_min": 200000},
    "naca2412":  {"name": "NACA 2412",  "Cl_max": 1.45, "Cd_min": 0.0075, "Re_min": 150000},
    "naca0012":  {"name": "NACA 0012",  "Cl_max": 1.20, "Cd_min": 0.0070, "Re_min": 100000},
    "naca6409":  {"name": "NACA 6409",  "Cl_max": 1.85, "Cd_min": 0.0090, "Re_min": 300000},
    "naca23012": {"name": "NACA 23012", "Cl_max": 1.60, "Cd_min": 0.0078, "Re_min": 200000},
    "naca4415":  {"name": "NACA 4415",  "Cl_max": 1.65, "Cd_min": 0.0085, "Re_min": 250000},
    "clark_y":   {"name": "Clark Y",    "Cl_max": 1.55, "Cd_min": 0.0082, "Re_min": 150000},
    "e387":      {"name": "Eppler E387","Cl_max": 1.50, "Cd_min": 0.0060, "Re_min": 100000},
    "s1223":     {"name": "Selig S1223","Cl_max": 2.20, "Cd_min": 0.0120, "Re_min": 200000},
    "naca63412": {"name": "NACA 63-412","Cl_max": 1.60, "Cd_min": 0.0055, "Re_min": 500000},
}

def check_feasibility(cfg, airfoil_key="naca4412"):
    warnings = []
    errors   = []

    af     = AIRFOIL_DB.get(airfoil_key, AIRFOIL_DB["naca4412"])
    Cl_max = af["Cl_max"]
    Re_min = af["Re_min"]
    rho    = 1.225
    b      = cfg.max_span_m
    V      = cfg.cruise_ms

    battery_mass = cfg.battery_wh / 145.0
    est_struct   = 0.14 * (b ** 1.6) * (7.0 ** 0.4)
    mtow_est     = cfg.payload_kg + battery_mass + est_struct
    W_est        = mtow_est * 9.81

    # Realistic AR for the given span — cap at 10 for feasibility check
    AR_check   = min(10.0, 8.0)
    S_est      = (b ** 2) / AR_check
    Cl_cruise  = 0.6 * Cl_max

    S_min_cruise = W_est / (0.5 * rho * V**2 * Cl_cruise) if V > 0 else 999
    S_min_stall  = W_est / (0.5 * rho * (V * 0.7)**2 * Cl_max) if V > 0 else 999
    S_min        = max(S_min_cruise, S_min_stall)
    S_max_available = b**2 / 4.0
    AR_max = b**2 / S_min if S_min > 0 else 999

    nu        = 1.5e-5
    chord_est = S_min / b if b > 0 else 0.1
    Re_est    = V * chord_est / nu

    # Stall speed at a realistic AR=8 wing
    stall_est = math.sqrt((2 * W_est) / (rho * S_est * Cl_max)) if S_est > 0 else 99

    # ── HARD ERRORS ──────────────────────────────────────────────────
    if S_min > S_max_available * 1.5:
        errors.append(
            f"Wing area required ({S_min:.2f} m²) far exceeds what a {b:.1f} m wingspan can provide "
            f"({S_max_available:.2f} m² max). Increase wingspan or reduce payload."
        )
    if AR_max < 3.0:
        errors.append(
            f"Required aspect ratio ({AR_max:.1f}) is below 3.0 — aerodynamically unstable. "
            f"Increase wingspan or reduce payload."
        )
    if V < 5.0:
        errors.append(
            f"Cruise speed {V:.1f} m/s is below the minimum controllable airspeed for any fixed-wing UAV (5 m/s)."
        )
    if V < stall_est * 1.15:
        errors.append(
            f"Cruise speed ({V:.1f} m/s) is below the estimated stall speed ({stall_est:.1f} m/s) "
            f"for this aircraft weight and wingspan. Increase cruise speed to at least {stall_est * 1.2:.1f} m/s, "
            f"or increase wingspan / reduce payload to lower stall speed."
        )
    if cfg.payload_kg > 0.6 * mtow_est:
        errors.append(
            f"Payload ({cfg.payload_kg:.1f} kg) exceeds 60% of estimated MTOW ({mtow_est:.2f} kg). "
            f"Reduce payload or increase battery/wingspan."
        )

    # ── SOFT WARNINGS ────────────────────────────────────────────────
    warnings.append("⚙️ Using estimated aerodynamics — XFOIL simulation not available on this server.")

    if Re_est < Re_min and not errors:
        warnings.append(
            f"Estimated Reynolds number (~{Re_est:.0f}) is below {af['name']}'s minimum Re ({Re_min:,}). "
            f"Airfoil may stall early at this scale/speed."
        )
    if cfg.battery_wh / W_est < 20:
        warnings.append(
            f"Battery energy is low relative to aircraft weight ({cfg.battery_wh:.0f} Wh / {W_est:.1f} N). "
            f"Range may be very short."
        )
    if cfg.flying_wing and airfoil_key not in ("naca23012", "naca0012", "naca4412"):
        warnings.append(
            f"{af['name']} may cause pitch-up instability on a flying wing. "
            f"Recommended: NACA 23012 (reflex) or NACA 0012 (symmetric)."
        )

    is_feasible = len(errors) == 0
    return is_feasible, warnings, errors


def analyse(design, cfg, airfoil_key="naca4412"):
    af = AIRFOIL_DB.get(airfoil_key, AIRFOIL_DB["naca4412"])
    b        = min(design.wingspan, cfg.max_span_m)
    # Hard cap on AR to realistic UAV values
    AR       = min(design.aspect_ratio, 10.0 if cfg.flying_wing else 9.0)
    taper    = design.taper_ratio
    tc       = design.t_c_ratio
    twist    = min(abs(design.twist_deg), 3.0)
    winglets = design.winglet_pres
    fw       = cfg.flying_wing

    S          = (b ** 2) / AR if AR > 0 else 0.1
    root_chord = (2 * S) / (b * (1 + taper)) if b > 0 else 0.1
    tip_chord  = root_chord * taper
    AR_eff     = AR * (1.0 + 1.9 * (tip_chord * 0.15 / b)) if winglets > 0.5 else AR

    Cd_min = af["Cd_min"]
    Cl_max = af["Cl_max"]

    if fw:
        C_Do         = Cd_min + (0.005 * tc)
        trim_penalty = 0.08
    else:
        C_Do         = Cd_min + (0.006 * tc)
        trim_penalty = 0.0

    k_twist   = max(0.94, 1.0 - (twist * 0.01))
    C_Di      = (0.38 * k_twist) / (math.pi * AR_eff) if AR_eff > 0 else 0.4
    Cl_cruise = 0.7 * Cl_max
    ld_raw    = (Cl_cruise / (C_Do + C_Di)) * 0.55 if (C_Do + C_Di) > 0 else 5.0
    ld        = ld_raw * (1.0 - trim_penalty)
    fw_mass_factor = 0.85 if fw else 1.0
    mass_struct    = fw_mass_factor * 0.14 * (b ** 1.6) * (AR ** 0.4) * \
                     (1.0 + (0.08 if winglets > 0.5 else 0.0))
    mass_total     = cfg.payload_kg + (cfg.battery_wh / 145.0) + mass_struct
    weight         = mass_total * 9.81

    stall_speed = math.sqrt((2 * weight) / (1.225 * S * 1.15)) if S > 0 else 10.0
    range_km    = (cfg.battery_wh * 0.65 * ld * 3.6) / weight if weight > 0 else 0
    endurance   = range_km / (cfg.cruise_ms * 3.6) if cfg.cruise_ms > 0 else 0
    safety      = max(1.0, (tc * 150.0) / (b + 0.5))
    stability_penalty = max(0.0, (8.0 - AR) * 1.5) if (fw and AR < 8.0) else 0.0

    # Penalise designs where stall speed exceeds cruise speed
    stall_penalty = max(0.0, (stall_speed - cfg.cruise_ms * 0.85) * 5.0)

    range_score     = min(25.0, (range_km / (cfg.mission_km * 3.0)) * 25.0)
    endurance_score = min(25.0, (endurance / (cfg.endurance_hr * 3.0)) * 25.0)
    safety_score    = min(25.0, (safety / 4.0) * 25.0)
    ld_score        = min(25.0, (ld / 14.0) * 25.0)
    score = min(100.0, max(5.0,
        range_score + endurance_score + safety_score + ld_score
        - stability_penalty - stall_penalty))

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
        "xfoil_run":    False,
    }


def optimize(cfg):
    fw = cfg.flying_wing
    if fw:
        bounds = [
            (0.5,  cfg.max_span_m),
            (6.0,  10.0),   # AR 6–10 for flying wing
            (0.2,   0.6),
            (0.09,  0.16),
            (0.0,   3.0),
            (0.0,   1.0),
        ]
    else:
        bounds = [
            (0.5,  cfg.max_span_m),
            (4.0,   9.0),   # AR 4–9 for conventional UAV
            (0.3,   0.8),
            (0.08,  0.16),
            (0.0,   3.0),
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
    best_design = Design(
        wingspan     = res.x[0],
        aspect_ratio = res.x[1],
        taper_ratio  = res.x[2],
        t_c_ratio    = res.x[3],
        twist_deg    = res.x[4],
        winglet_pres = res.x[5],
    )
    return best_design, -res.fun
