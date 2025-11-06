.PHONY: env preprocess db_upload train evaluate predict

env:
	conda env create -f environment.yml || python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

preprocess:
	python src/preprocess.py --input data/raw/customer_churn_raw.csv --output data/interim/customer_clean.csv

db_upload:
	python src/db_utils.py --csv data/interim/customer_clean.csv

train:
	python src/train.py --csv data/interim/customer_clean.csv --out models/model.pkl --experiment churn_experiment

evaluate:
	python src/evaluate.py --model models/model.pkl --csv data/interim/customer_clean.csv

predict:
	python src/predict_function.py
