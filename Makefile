.PHONY: setup generate generate-clean test lint

setup:
	pip install -r requirements.txt

generate:
	cd src && python dataset_builder.py --records 10000 --dup-rate 0.20 --seed 42

generate-clean:
	cd src && python dataset_builder.py --records 10000 --clean-only --seed 42

test:
	pytest tests/ -v --tb=short

lint:
	python -m py_compile src/ine_loader.py src/record_generator.py src/dataset_builder.py
	@echo "Sintaxis OK"
