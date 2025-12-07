VENV = .venv
PY   = $(VENV)/bin/python

.PHONY: stash
stash:
	-@mv conv2d_model.pt models/conv2d_model_$(shell date +%Y%m%d_%H%M%S).pt

.PHONY: train
train: stash
	$(PY) conv2d_train.py

.PHONY: docker
docker:
	docker build --platform linux/arm64 -t dhogborg/washer:latest .
	docker save -o docker/image.tar dhogborg/washer:latest
