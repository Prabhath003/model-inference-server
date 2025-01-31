from flask import Blueprint, request, jsonify, Flask
from ..manager import createModelEndpoint, inferModelInstance

infer_bp = Blueprint('infer', __name__)

@infer_bp.route("/infer", methods=["POST"])
def infer():
    data = request.get_json()
    model_name = data.get("model_name")
    model_type = data.get("type", "text-generation")

    if not model_name or not model_type:
        return jsonify({"error": "Missing model_name or model_type"}), 400

    try:
        # Perform inference using the model server
        createModelEndpoint(model_name, model_type)
        response = inferModelInstance(data)
        return jsonify(response.json()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def register_routes(app: Flask):
    app.register_blueprint(infer_bp)