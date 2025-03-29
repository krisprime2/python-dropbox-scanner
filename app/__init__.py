from flask import Flask
from config import Config
import logging


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Logging konfigurieren
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
    log_file = app.config.get('LOG_FILE')

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging konfiguriert")

    # Routen registrieren
    from app.routes import main
    app.register_blueprint(main)

    return app