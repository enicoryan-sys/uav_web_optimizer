from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import uav_design

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/feasibility', methods=['POST'])
def check_feasibility():
    try:
        data = request.json
        cfg = uav_design.Config(
            payload_kg   = float(data.get('payload',    2.0)),
            battery_wh   = float(data.get('battery',  310.0)),
            max_span_m   = float(data.get('maxspan',    2.7)),
            cruise_ms    = float(data.get('cruisems',  16.0)),
            mission_km   = float(data.get('range',     55.0)),
            endurance_hr = float(data.get('endurance',  2.0)),
            flying_wing  = bool(data.get('flyingwing', False)),
        )
        airfoil = data.get('airfoil', 'naca4412')
        if airfoil == 'clarky': airfoil = 'clark_y'
        is_feasible, warnings, errors = uav_design.check_feasibility(cfg, airfoil)
        return jsonify({"status":"success","feasible":is_feasible,"warnings":warnings,"errors":errors})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 400

@app.route('/api/optimize', methods=['POST'])
def run_optimization():
    try:
        data = request.json
        cfg = uav_design.Config(
            payload_kg   = float(data.get('payload',    2.0)),
            battery_wh   = float(data.get('battery',  310.0)),
            max_span_m   = float(data.get('maxspan',    2.7)),
            cruise_ms    = float(data.get('cruisems',  16.0)),
            mission_km   = float(data.get('range',     55.0)),
            endurance_hr = float(data.get('endurance',  2.0)),
            flying_wing  = bool(data.get('flyingwing', False)),
        )
        airfoil = data.get('airfoil', 'naca4412')
        if airfoil == 'clarky': airfoil = 'clark_y'

        is_feasible, warnings, errors = uav_design.check_feasibility(cfg, airfoil)
        if not is_feasible:
            return jsonify({"status":"infeasible","feasible":False,"warnings":warnings,"errors":errors}), 422

        best_design, score = uav_design.optimize(cfg)
                r = uav_design.analyse(best_design, cfg, airfoil)

        results = {
            "score":       r["score"],
            "ld":          r["ld"],
            "rangekm":     r["range_km"],
            "endurancehr": r["endurance_hr"],
            "stallspeed":  r["stall_speed"],
            "masstotal":   r["mass_total"],
            "weightn":     r["weight_n"],
            "wingspan":    r["wingspan"],
            "aspectratio": r["aspect_ratio"],
            "taperratio":  r["taper_ratio"],
            "tc":          r["t_c"],
            "rootchord":   r["root_chord"],
            "tipchord":    r["tip_chord"],
            "twistdeg":    r["twist_deg"],
            "wingletpres": r["winglet_pres"],
            "wingarea":    r["wing_area"],
            "CDo":         r["C_Do"],
            "CDi":         r["C_Di"],
            "xfoilRun":    False,
            "airfoilkey":  airfoil,
            "airfoilname": uav_design.AIRFOIL_DB.get(airfoil, {}).get("name", airfoil),
        }
        return jsonify({"status":"success","feasible":True,"warnings":warnings,"data":results})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
