from flask import Flask, render_template, request, jsonify
import traceback
import math

# Imports your backend analysis logic
from uav_design import Config, Design, optimize, analyse

app = Flask(__name__)

def sanitize_for_json(obj):
    """Converts hidden numpy types so the browser json reader handles them safely."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/optimize", methods=["POST"])
def optimize_route():
    try:
        data = request.get_json()

        cfg = Config(
            payload_kg    = float(data.get("payload_kg", 2.0)),
            battery_wh    = float(data.get("battery_wh", 310)),
            max_span_m    = float(data.get("max_span_m", 2.7)),
            cruise_ms     = float(data.get("cruise_ms", 16.0)),
            mission_km    = float(data.get("mission_km", 55)),
            endurance_hr  = float(data.get("endurance_hr", 2.0)),
            flying_wing   = bool(data.get("flying_wing", False)),
        )

        # Run optimization matrix
        best_design, _ = optimize(cfg)
        result = analyse(best_design, cfg)

        clean_result = sanitize_for_json(result)
        return jsonify({"ok": True, "result": clean_result})

    except Exception as e:
        print("\n" + "="*50 + " BACKEND CRASH TRACE " + "="*50)
        print(traceback.format_exc())
        print("="*121 + "\n")
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500

@app.route("/evaluate", methods=["POST"])
def evaluate_route():
    try:
        data = request.get_json()

        cfg = Config(
            payload_kg    = float(data.get("payload_kg", 2.0)),
            battery_wh    = float(data.get("battery_wh", 310)),
            max_span_m    = float(data.get("max_span_m", 2.7)),
            cruise_ms     = float(data.get("cruise_ms", 16.0)),
            mission_km    = float(data.get("mission_km", 55)),
            endurance_hr  = float(data.get("endurance_hr", 2.0)),
            flying_wing   = bool(data.get("flying_wing", False)),
        )

        # Reads the classic sliders from your UI, defaulting advanced parameters safely
        design = Design(
            wingspan     = float(data["wingspan"]),
            aspect_ratio = float(data["aspect_ratio"]),
            taper_ratio  = float(data["taper_ratio"]),
            t_c_ratio    = float(data["t_c_ratio"]),
            twist_deg    = float(data.get("twist_deg", 0.0)),
            winglet_pres = float(data.get("winglet_pres", 0.0))
        )

        result = analyse(design, cfg)
        clean_result = sanitize_for_json(result)
        return jsonify({"ok": True, "result": clean_result})

    except Exception as e:
        print("\n" + "="*50 + " BACKEND CRASH TRACE " + "="*50)
        print(traceback.format_exc())
        print("="*121 + "\n")
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)
