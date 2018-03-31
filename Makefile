.PHONY: init
init:
	pip install pipenv --upgrade
	pipenv install --dev

.PHONY: test
test:
	pipenv run pytest
	pipenv run mypy ./ledge

.PHONY: publish
publish:
	pipenv run python setup.py sdist bdist_wheel
	pipenv run twine upload dist/*
	rm -rf build dist

.PHONY: docs
docs:
	cd docs && pipenv run make html
