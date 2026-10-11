from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from app.config import Config
from app.routes.audio import audio_bp


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": app.config["ALLOWED_ORIGINS"]
            }
        },
        supports_credentials=False,
    )
    app.register_blueprint(audio_bp)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(error):
        return jsonify(error="Uploaded file exceeds the 100 MB limit."), 413

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        app.logger.exception("Unexpected API error")
        return jsonify(error="An unexpected server error occurred."), 500

    return app
