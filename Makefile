.PHONY: api test embed

api:
	uvicorn feedback_insights.api:app --reload --port 8001

test:
	TEST_DATABASE_URL=postgresql://feedback:feedback@localhost:5432/feedback pytest -v

embed:
	python -m setup.embed
