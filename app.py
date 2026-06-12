from flask import Flask, render_template, request, jsonify
import uav_design

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/optimize', methods=['POST'])
def run_optimization():
    try:
        data = request.json
        cfg = uav_design.Config(
            payload_kg=float(data.get('payload', 2.0)),
            max_span_m=float(data.get('max_span', 2.7)),
            mission_km=float(data.get('range', 55))
        )
        best_design, score = uav_design.optimize(cfg)
        results = uav_design.analyse(best_design, cfg)
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
