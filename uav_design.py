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

def analyse(design, cfg):
    b        = min(design.wingspan, cfg.max_span_m)
    AR       = design.aspect_ratio
    taper    = design.taper_ratio
    tc       = design.t_c_ratio
    twist    = design.twist_deg
    winglets = design.winglet_pres

    S          = (b ** 2) / AR if AR > 0 else 0.1
    root_chord = (2 * S) / (b * (1 + taper)) if b > 0 else 0.1
    tip_chord  = root_chord * taper

    AR_eff  = AR * (1.0 + 1.9 * (tip_chord * 0.15 / b)) if winglets > 0.5 else AR
    C_Do    = 0.019 + (0.006 * tc) + (0.002 if winglets > 0.5 else 0.0)
    k_twist = max(0.88, 1.0 - (twist * 0.015))
    C_Di    = (0.38 * k_twist) / (math.pi * AR_eff) if AR_eff > 0 else 0.4
    ld      = 0.45 / (C_Do + C_Di) if (C_Do + C_Di) > 0 else 5.0

    mass_struct = 0.14 * (b ** 1.6) * (AR ** 0.4) * (1.0 + (0.08 if winglets > 0.5 else 0.0))
    mass_total  = cfg.payload_kg + (cfg.battery_wh / 145.0) + mass_struct
    weight      = mass_total * 9.81

    stall_speed = math.sqrt((2 * weight) / (1.225 * S * 1.15)) if S > 0 else 10.0
    range_km    = (cfg.battery_wh * 0.65 * ld * 3.6) / weight if weight > 0 else 0
    endurance   = range_km / (cfg.cruise_ms * 3.6) if cfg.cruise_ms > 0 else 0
    safety      = max(1.0, (tc * 150.0) / (b + 0.5))

    range_score     = min(25.0, (range_km / cfg.mission_km) * 20.0)
    endurance_score = min(25.0, (endurance / cfg.endurance_hr) * 20.0)
    safety_score    = min(25.0, (safety / 2.0) * 15.0)
    ld_score        = min(25.0, (ld / 18.0) * 20.0)

    score = min(100.0, max(5.0, range_score + endurance_score + safety_score + ld_score))

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
    }

def optimize(cfg):
    bounds = [
        (0.5,  cfg.max_span_m),  # wingspan
        (4.0,  20.0),             # aspect_ratio
        (0.1,   1.0),             # taper_ratio
        (0.06,  0.20),            # t_c_ratio
        (0.0,   6.0),             # twist_deg
        (0.0,   1.0),             # winglet_pres
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
