"""Keep tests away from the real application database (data/cognios_app.db).

Set at import time, before any module-scoped fixture calls create_app().
"""
import os
import tempfile

os.environ["COGNIOS_APP_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="cognios_test_"), "app.db")
