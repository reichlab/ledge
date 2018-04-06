.PHONY: init
init:
	pip install tox pipenv --upgrade

.PHONY: test
test:
	tox

.PHONY: publish
publish:
	pipenv run python setup.py sdist bdist_wheel
	pipenv run twine upload dist/*
	rm -rf build dist

.PHONY: docs
docs:
	cd docs && pipenv run make html
