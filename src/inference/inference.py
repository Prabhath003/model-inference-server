from flask import request, jsonify
import logging

class InferenceHandler:
    def __init__(self, model_server):
        self.model_server = model_server

    def handle_inference(self):
        data = request.get_json()
        model_name = data.get("model_name")
        input_text = data.get("input_text")
        type = data.get("type", "text-generation")

        if not model_name:
            logging.error("Missing model_name in request")
            return jsonify({"error": "Missing model_name"}), 400

        if type not in self.model_server.processes:
            logging.error(f"Model {model_name} of type {type} is not available.")
            return jsonify({"error": "Model not available"}), 404

        try:
            response = self.model_server.processes[f"{model_name}_{type}"]["model"].infer(input_text)
            return jsonify(response), 200
        except Exception as e:
            logging.error(f"Inference error: {e}")
            return jsonify({"error": str(e)}), 500

def create_inference_route(app, model_server):
    handler = InferenceHandler(model_server)

    @app.route("/infer", methods=["POST"])
    def infer():
        return handler.handle_inference()