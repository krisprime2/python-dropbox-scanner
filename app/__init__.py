from flask import Flask
from config import Config
import logging
from flask_apscheduler import APScheduler

# Globale Scheduler-Instanz
scheduler = APScheduler()


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

    # Scheduler initialisieren, wenn SCHEDULER_ENABLED aktiviert ist
    if app.config.get('SCHEDULER_ENABLED', False):
        logger.info("Initialisiere automatischen Scheduler für nächtliche Indexierung")
        # Wichtig: Scheduler zur App initialisieren
        scheduler.init_app(app)
        scheduler.start()

        # Jobs hinzufügen
        from app.scheduled_tasks import configure_scheduled_jobs
        configure_scheduled_jobs(scheduler)  # Übergebe den scheduler, nicht die app

        logger.info("Scheduler gestartet - Nächste Indexierung um 03:00 Uhr")
    else:
        logger.info("Automatischer Scheduler ist deaktiviert")

    return app