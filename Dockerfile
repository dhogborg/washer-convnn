FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libportaudiocpp0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.min.txt ./
RUN pip install --no-cache-dir -r requirements.min.txt

COPY monitor_onnx.py \
    export/conv2d_model.onnx \
    export/conv2d_model.onnx.data \
    export/conv2d_model.onnx.meta.json ./

ENV MONITOR_DEVICE_INDEX=0

ENTRYPOINT ["python", "monitor_onnx.py", "--onnx-model", "/app/conv2d_model.onnx"]
