from flask import Flask
from .routes.infer import infer_bp
from dotenv import load_dotenv
import os

load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(infer_bp)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "5000")))