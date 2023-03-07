.PHONY: uq mixture

uq:
	pipenv run python3 -m rectflow.uq

mixture:
	pipenv run python3 -m rectflow.mixture
