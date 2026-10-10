import os

from flask import Flask

import config

from . import dashboard, db, manufacturer, predictions, reviews, routes, store


def create_app(db_path=None, ml_db_path=None, app_db_path=None):
    app = Flask(__name__)
    app.json.sort_keys = False  # keep column order in responses
    app.config["DB_PATH"] = db_path or config.DB_PATH

    if not os.path.exists(app.config["DB_PATH"]):
        raise RuntimeError(
            f"Database not found at {app.config['DB_PATH']}. "
            "Build it first:  python scripts/import_data.py"
        )

    # ML inference reads from the V4.2 database; callers can override via
    # environment variable EHR_ML_DB_PATH or the ml_db_path argument.
    app.config["ML_DB_PATH"] = (
        ml_db_path
        or os.environ.get("EHR_ML_DB_PATH")
        or config.ML_DB_PATH
    )

    # Application database: doctor reviews and logged AI predictions (created on first use).
    app.config["APP_DB_PATH"] = (
        app_db_path or os.environ.get("COGNIOS_APP_DB_PATH") or config.APP_DB_PATH
    )
    store.init_db(app.config["APP_DB_PATH"])

    app.teardown_appcontext(db.close_db)
    app.register_blueprint(routes.bp)
    app.register_blueprint(predictions.bp)
    app.register_blueprint(dashboard.bp)
    app.register_blueprint(reviews.bp)
    app.register_blueprint(manufacturer.bp)
    return app
