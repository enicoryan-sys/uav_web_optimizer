import numpy as np
import scipy.optimize
import math

try:
    from xfoil import XFoil
    from xfoil.model import Airfoil
    XFOIL_AVAILABLE = True
except ImportError:
    XFOIL_AVAILABLE = False

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
    "naca4412":  {"name": "NACA 4412",  "Cl_max": 1.60, "Cd_min": 0.0080, "Re_min": 200000, "naca": "4412"},
    "naca2412":  {"name": "NACA 2412",  "Cl_max": 1.45, "Cd_min": 0.0075, "Re_min": 150000, "naca": "2412"},
    "naca0012":  {"name": "NACA 0012",  "Cl_max": 1.20, "Cd_min": 0.0070, "Re_min": 100000, "naca": "0012"},
    "naca6409":  {"name": "NACA 6409",  "Cl_max": 1.85, "Cd_min": 0.0090, "Re_min": 300000, "naca": "6409"},
    "naca23012": {"name": "NACA 23012", "Cl_max": 1.60, "Cd_min": 0.0078, "Re_min": 200000, "naca": "23012"},
    "naca4415":  {"name": "NACA 4415",  "Cl_max": 1.65, "Cd_min": 0.0085, "Re_min": 250000, "naca": "4415"},
    "clark_y":   {"name": "Clark Y",    "Cl_max": 1.55, "Cd_min": 0.0082, "Re_min": 150000, "naca": None},
    "e387":      {"name": "Eppler E387","Cl_max": 1.50, "Cd_min": 0.0060, "Re_min": 100000, "naca": None},
    "s1223":     {"name": "Selig S1223","Cl_max": 2.20, "Cd_min": 0.0120, "Re_min": 200000, "naca": None},
    "naca63412": {"name": "NACA 63-412","Cl_max": 1.60, "Cd_min": 0.0055, "Re_min": 500000, "naca": "63412"},
}

# Coordinate data for non-NACA airfoils
AIRFOIL_COORDS = {
    "clark_y": [
        (0.0,0.0),(0.025,0.0385),(0.05,0.0576),(0.10,0.0825),(0.15,0.0988),
        (0.20,0.1100),(0.30,0.1215),(0.40,0.1265),(0.50,0.1240),(0.60,0.1150),
        (0.70,0.1005),(0.80,0.0820),(0.90,0.0555),(0.95,0.0395),(1.0,0.0125),
        (0.95,0.0027),(0.90,0.0033),(0.80,0.0018),(0.70,-0.002),(0.60,-0.0068),
        (0.50,-0.012),(0.40,-0.0172),(0.30,-0.0218),(0.20,-0.0245),(0.10,-0.0234),
        (0.05,-0.0197),(0.025,-0.015),(0.0,0.0),
    ],
    "e387": [
        (0.0,0.0),(0.025,0.0393),(0.05,0.0545),(0.10,0.0759),(0.20,0.1018),
        (0.30,0.1117),(0.40,0.1115),(0.50,0.1047),(0.60,0.0919),(0.70,0.0732),
        (0.80,0.0504),(0.90,0.0256),(1.0,0.0),
        (0.90,0.001),(0.80,0.0005),(0.70,-0.0031),(0.60,-0.0082),(0.50,-0.0144),
        (0.40,-0.0218),(0.30,-0.0296),(0.20,-0.0355),(0.10,-0.0356),
        (0.05,-0.0295),(0.025,-0.0226),(0.0,0.0),
    ],
    "s1223": [
        (0.0,0.0),(0.01,0.034),(0.025,0.052),(0.05,0.075),(0.10,0.107),
        (0.20,0.147),(0.30,0.165),(0.40,0.164),(0.50,0.150),(0.60,0.127),
        (0.70,0.097),(0.80,0.064),(0.90,0.031),(1.0,0.0),
        (0.90,0.003),(0.80,-0.002),(0.70,-0.011),(0.60,-0.023),(0.50,-0.038),
        (0.40,-0.053),(0.30,-0.065),(0.20,-0.069),(0.10,-0.053),
        (0.05,-0.034),(0.025,-0.021),(0.0,0.0),
    ],
}

def run_xfoil_simulation(airfoil_key, reynolds, alpha_start=-2, alpha_end=14, alpha_step=0.5):
    """
    Run XFOIL simulation for a given airfoil at a given Reynolds number.
    Returns dict with polar data: alphas, cls, cds, cms, cl_max, cd_min, ld_max, cl_cruise, cd_cruise
    Returns None if XFOIL not available or simulation fails.
    """
    if not XFOIL_AVAILABLE:
        return None

    af_info = AIRFOIL_DB.get(airfoil_key, AIRFOIL_DB["naca4412"])
    naca_code = af_info.get("naca")

    try:
        xf = XFoil()
        xf.max_iter = 100
        xf.Re = max(50000, min(reynolds, 5000000))
        xf.M = 0.0
        xf.n_crit = 9.0

        if naca_code:
            xf.naca(naca_code)
        elif airfoil_key in AIRFOIL_COORDS:
            coords = np.array(AIRFOIL_COORDS[airfoil_key])
            xf.airfoil = Airfoil(x=coords[:,0], y=coords[:,1])
        else:
            return None

        xf.repanel()

        a, cl, cd, cm, cp = xf.aseq(alpha_start, alpha_end, alpha_step)

        # Filter out failed points (NaN)
        valid = ~np.isnan(cl) & ~np.isnan(cd) & (cd > 0)
        if valid.sum() < 3:
            return None

        a, cl, cd, cm = a[valid], cl[valid], cd[valid], cm[valid]

        ld = cl / cd
        ld_max_idx = np.argmax(ld)
        cl_max = float(np.max(cl))
        cd_min = float(np.min(cd))
        ld_max = float(ld[ld_max_idx])
        cl_at_ld_max = float(cl[ld_max_idx])
        cd_at_ld_max = float(cd[ld_max_idx])
        alpha_at_ld_max = float(a[ld_max_idx])

        # Cruise point: 60% of cl_max
        cl_cruise_target = 0.6 * cl_max
        idx_cruise = np.argmin(np.abs(cl - cl_cruise_target))
        cl_cruise = float(cl[idx_cruise])
        cd_cruise = float(cd[idx_cruise])
        alpha_cruise = float(a[idx_cruise])

        return {
            "xfoil_run":      True,
            "reynolds":       int(xf.Re),
            "cl_max":         round(cl_max, 4),
            "cd_min":         round(cd_min, 6),
            "ld_max":         round(ld_max, 2),
            "cl_at_ld_max":   round(cl_at_ld_max, 4),
            "cd_at_ld_max":   round(cd_at_ld_max, 6),
            "alpha_at_ld_max":round(alpha_at_ld_max, 2),
            "cl_cruise":      round(cl_cruise, 4),
            "cd_cruise":      round(cd_cruise, 6),
            "alpha_cruise":   round(alpha_cruise, 2),
            "polar_alpha":    [round(float(x),2) for x in a],
            "polar_cl":       [round(float(x),4) for x in cl],
            "polar_cd":       [round(float(x),6) for x in cd],
            "polar_ld":       [round(float(x),2) for x in ld],
        }
    except Exception as e:
        return None


def check_feasibility(cfg, airfoil_key="naca4412"):
    warnings = []
    errors   = []
    af = AIRFOIL_DB.get(airfoil_key, AIRFOIL_DB["naca4412"])
    Cl_max = af["Cl_max"]
    Re_min = af["Re_min"]
    rho    = 1.225
    b      = cfg.max_span_m
    V      = cfg.cruise_ms
    battery_mass = cfg.battery_wh / 145.0
    est_struct   = 0.14 * (b ** 1.6) * (7.0 ** 0.4)
    mtow_est     = cfg.payload_kg + battery_mass + est_struct
    W_est        = mtow_est * 9.81
    Cl_cruise    = 0.6 * Cl_max
    S_min_cruise = W_est / (0.5 * rho * V**2 * Cl_cruise) if V > 0 else 999
    S_min_stall  = W_est / (0.5 * rho * (V*0.7)**2 * Cl_max) if V > 0 else 999
    S_min        = max(S_min_cruise, S_min_stall)
    S_max_available = b**2 / 4.0
    AR_max = b**2 / S_min if S_min > 0 else 999
    nu = 1.5e-5
    chord_est = S_min / b if b > 0 else 0.1
    Re_est    = V * chord_est / nu
    if S_min > S_max_available * 1.5:
        errors.append(f"Wing area required ({S_min:.2f} m²) far exceeds what a {b:.1f} m wingspan can provide.")
    if AR_max < 3.0:
        errors.append(f"Aspect ratio would need to be {AR_max:.1f} — below 3.0 is aerodynamically unstable.")
    if V < 5.0:
        errors.append(f"Cruise speed {V:.1f} m/s is below the minimum controllable airspeed (5 m/s).")
    if cfg.payload_kg > 0.6 * mtow_est:
        errors.append(f"Payload ({cfg.payload_kg:.1f} kg) is too high relative to estimated MTOW ({mtow_est:.2f} kg).")
    if Re_est < Re_min and not errors:
        warnings.append(f"Estimated Reynolds number (~{Re_est:.0f}) is below {af['name']}'s minimum Re ({Re_min:,}).")
    if cfg.battery_wh / W_est < 20:
        warnings.append(f"Battery energy is low relative to aircraft weight. Range may be very short.")
    if cfg.flying_wing and airfoil_key not in ("naca23012", "naca0012", "naca4412"):
        warnings.append(f"{af['name']} may cause pitch instability on a flying wing. Recommended: NACA 23012 or NACA 0012.")
    is_feasible = len(errors) == 0
    return is_feasible, warnings, errors


def analyse(design, cfg, xfoil_data=None):
    b        = min(design.wingspan, cfg.max_span_m)
    AR       = min(design.aspect_ratio, 16.0 if cfg.flying_wing else 12.0)
    taper    = design.taper_ratio
    tc       = design.t_c_ratio
    twist    = min(abs(design.twist_deg), 4.0)
    winglets = design.winglet_pres
    fw       = cfg.flying_wing

    S          = (b ** 2) / AR if AR > 0 else 0.1
    root_chord = (2 * S) / (b * (1 + taper)) if b > 0 else 0.1
    tip_chord  = root_chord * taper
    AR_eff     = AR * (1.0 + 1.9 * (tip_chord * 0.15 / b)) if winglets > 0.5 else AR

    if xfoil_data and xfoil_data.get("xfoil_run"):
        # Use real XFOIL values
        Cd_profile  = xfoil_data["cd_cruise"]
        Cl_cruise   = xfoil_data["cl_cruise"]
        ld_2d       = xfoil_data["ld_max"]
        C_Do        = Cd_profile * (1.1 if not fw else 1.0)
        C_Di        = (Cl_cruise ** 2) / (math.pi * AR_eff * 0.85) if AR_eff > 0 else 0.05
        ld          = Cl_cruise / (C_Do + C_Di) if (C_Do + C_Di) > 0 else ld_2d * 0.7
        if fw: ld *= 0.92
    else:
        # Fallback estimated values
        if fw:
            C_Do         = 0.016 + (0.005 * tc)
            trim_penalty = 0.08
        else:
            C_Do         = 0.019 + (0.006 * tc)
            trim_penalty = 0.0
        k_twist = max(0.92, 1.0 - (twist * 0.01))
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
    stability_penalty = max(0.0, (8.0 - AR) * 1.5) if (fw and AR < 8.0) else 0.0

    range_score     = min(25.0, (range_km / cfg.mission_km) * 20.0)
    endurance_score = min(25.0, (endurance / cfg.endurance_hr) * 20.0)
    safety_score    = min(25.0, (safety / 2.0) * 15.0)
    ld_score        = min(25.0, (ld / 18.0) * 20.0)
    score = min(100.0, max(5.0,
        range_score + endurance_score + safety_score + ld_score - stability_penalty))

    result = {
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
        "xfoil_run":    bool(xfoil_data and xfoil_data.get("xfoil_run")),
    }
    if xfoil_data and xfoil_data.get("xfoil_run"):
        result["xfoil"] = xfoil_data
    return result


def optimize(cfg):
    fw = cfg.flying_wing
    if fw:
        bounds = [(0.5,cfg.max_span_m),(7.0,16.0),(0.2,0.7),(0.09,0.18),(0.0,3.0),(0.0,1.0)]
    else:
        bounds = [(0.5,cfg.max_span_m),(4.0,12.0),(0.3,0.9),(0.08,0.18),(0.0,4.0),(0.0,1.0)]

    def objective_wrapper(flat_x):
        d = Design(wingspan=flat_x[0], aspect_ratio=flat_x[1], taper_ratio=flat_x[2],
                   t_c_ratio=flat_x[3], twist_deg=flat_x[4], winglet_pres=flat_x[5])
        return -analyse(d, cfg)["score"]

    res = scipy.optimize.differential_evolution(
        objective_wrapper, bounds, maxiter=200, popsize=15,
        tol=1e-6, seed=42, polish=True
    )
    best_design = Design(wingspan=res.x[0], aspect_ratio=res.x[1], taper_ratio=res.x[2],
                         t_c_ratio=res.x[3], twist_deg=res.x[4], winglet_pres=res.x[5])
    return best_design, -res.fun
