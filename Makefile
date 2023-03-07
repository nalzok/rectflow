.PHONY: uq mixture

uq:
	JAX_PLATFORMS="cpu" pipenv run python3 -m rectflow.uq

mixture:
	JAX_PLATFORMS="cpu" pipenv run python3 -m rectflow.mixture
