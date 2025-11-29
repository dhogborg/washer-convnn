VENV = .venv
PY   = $(VENV)/bin/python


train:
	$(PY) train.py

.PHONY: docker
docker:
	docker build --platform linux/arm64 -t dhogborg/washer:latest .
	docker save -o docker/image.tar dhogborg/washer:latest