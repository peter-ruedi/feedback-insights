"""HTTP API. Run with: uvicorn feedback_insights.api:app --reload"""

from .app import Deps, app, create_app

__all__ = ["Deps", "app", "create_app"]
