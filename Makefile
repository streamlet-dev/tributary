build:  ## Build the repository
	python setup.py build 

develop:  ## install to site-packages in editable mode
	python -m pip install --upgrade build pip setuptools twine wheel
	python -m pip install -e .[develop]

tests: ## Clean and Make unit tests
	python -m pytest tributary --cov=tributary --junitxml=python_junit.xml --cov-report=xml --cov-branch

testsci: ## Clean and Make unit tests
	CI=true python -m pytest tributary --cov=tributary --junitxml=python_junit.xml --cov-report=xml --cov-branch

dockerup:
	docker-compose -f ci/docker-compose.yml up -d

dockerdown:
	docker-compose -f ci/docker-compose.yml down

notebooks:  ## test execute the notebooks
	./scripts/test_notebooks.sh

lint: ## run linter
	python -m flake8 tributary setup.py

fix:  ## run black fix
	python -m black tributary/ setup.py

check: checks
checks:  ## run lint and other checks
	check-manifest

clean: ## clean the repository
	find . -name "__pycache__" | xargs  rm -rf 
	find . -name "*.pyc" | xargs rm -rf 
	find . -name ".ipynb_checkpoints" | xargs  rm -rf 
	rm -rf .coverage coverage *.xml build dist *.egg-info lib node_modules .pytest_cache *.egg-info .autoversion .mypy_cache
	rm -rf ./*.gv*
	make -C ./docs clean

install:  ## install to site-packages
	python -m pip install .

docs:  ## make documentation
	make -C ./docs html
	open ./docs/_build/html/index.html

dist:  ## create dists
	rm -rf dist build
	python setup.py sdist bdist_wheel
	python -m twine check dist/*
	
publish: dist  ## dist to pypi
	python -m twine upload dist/* --skip-existing

# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'

.PHONY: testjs testpy tests test lintpy lintjs lint fixpy fixjs fix checks check build develop install labextension dist publishpy publishjs publish docs clean dockerup dockerdown
