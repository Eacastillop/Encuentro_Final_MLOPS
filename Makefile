install:
	pip install -r requirements.txt

train:
	python src/preprocessing.py
	python src/train.py
