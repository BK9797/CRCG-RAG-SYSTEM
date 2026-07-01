install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload

build-index:
	python scripts/build_index.py
