"""ONNX Runtime-based audio monitor streaming probabilities over MQTT."""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import sounddevice as sd
from PIL import Image
from scipy import signal

DEFAULT_IMAGE_SIZE = (171, 286)
SAMPLE_RATE = 44_100
CHANNELS = 1
BLOCK_SECONDS = 10
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_SECONDS
DEFAULT_DEVICE_INDEX = 0

audio_q: "queue.Queue[np.ndarray]" = queue.Queue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor mic input using an ONNX model")
    parser.add_argument("--onnx-model", type=Path, required=True, help="Path to an exported ONNX model")
    parser.add_argument("--mqtt-topic", type=str, required=False, help="MQTT topic to publish JSON payloads")
    parser.add_argument("--mqtt-host", type=str, default="", help="MQTT broker hostname")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--mqtt-keepalive", type=int, default=60, help="MQTT keepalive interval in seconds")
    parser.add_argument("--device-index", type=int, default=DEFAULT_DEVICE_INDEX, help="sounddevice input index to capture")
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=DEFAULT_IMAGE_SIZE,
        metavar=("H", "W"),
        help="Resize applied to spectrograms before inference",
    )
    return parser.parse_args()


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Status:", status)
    data = indata.copy()
    if data.ndim > 1:
        data = np.mean(data, axis=1, keepdims=True)
    audio_q.put(data[:, 0])


def recorder_thread(device_index: int) -> None:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        device=device_index,
        dtype="float32",
    ):
        while True:
            time.sleep(0.1)


def generate_spectrogram_image(samples: np.ndarray, samplerate: int) -> Image.Image:
    _, _, Sxx = signal.spectrogram(samples, fs=samplerate, nperseg=1024, noverlap=512)
    sxx_db = 10 * np.log10(Sxx + 1e-12)

    min_val = np.min(sxx_db)
    max_val = np.max(sxx_db)
    if np.isclose(max_val, min_val):
        normalized = np.zeros_like(sxx_db)
    else:
        normalized = (sxx_db - min_val) / (max_val - min_val)

    image_array = np.flipud((1.0 - normalized) * 255.0).astype(np.uint8)
    return Image.fromarray(image_array, mode="L")


def preprocess_image(image: Image.Image, img_size: Sequence[int]) -> np.ndarray:
    height, width = int(img_size[0]), int(img_size[1])
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr[np.newaxis, np.newaxis, :, :]
    return arr


def read_classes(meta_path: Path) -> Sequence[str]:
    if not meta_path.exists():
        return ("WASHER", "SPIN", "IDLE", "END", "FAN", "DRYER")
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ("WASHER", "SPIN", "IDLE", "END", "FAN", "DRYER")
    classes = data.get("classes")
    if isinstance(classes, list) and classes:
        return tuple(str(label) for label in classes)
    return ("WASHER", "SPIN", "IDLE", "END", "FAN", "DRYER")


def format_payload(classes: Sequence[str], probs: Sequence[float]) -> str:
    timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    payload = {
        "timestamp": timestamp,
        "probabilities": {label: float(prob) for label, prob in zip(classes, probs)},
    }
    return json.dumps(payload)


def connect_mqtt(host: str, port: int, keepalive: int) -> mqtt.Client:
    client = mqtt.Client()
    client.connect(host, port, keepalive)
    client.loop_start()
    return client


def publish_payload(client: mqtt.Client, topic: str, payload: str) -> None:
    result = client.publish(topic, payload)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"MQTT publish failed: {mqtt.error_string(result.rc)}")


def main() -> None:
    args = parse_args()

    env_topic = os.getenv("MONITOR_MQTT_TOPIC")
    if env_topic:
        args.mqtt_topic = env_topic

    env_host = os.getenv("MONITOR_MQTT_HOST")
    if env_host:
        args.mqtt_host = env_host

    env_port = os.getenv("MONITOR_MQTT_PORT")
    if env_port:
        try:
            args.mqtt_port = int(env_port)
        except ValueError:
            print("Invalid MONITOR_MQTT_PORT env; using CLI/default value")

    env_device_idx = os.getenv("MONITOR_DEVICE_INDEX")
    if env_device_idx is not None:
        try:
            args.device_index = int(env_device_idx)
        except ValueError:
            print("Invalid MONITOR_DEVICE_INDEX env; using CLI/default value")

    model_path = args.onnx_model.expanduser().resolve()
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    classes = read_classes(model_path.with_suffix(model_path.suffix + ".meta.json"))

    mqtt_client : mqtt.Client | None = None
    if args.mqtt_host:
        mqtt_client = connect_mqtt(args.mqtt_host, args.mqtt_port, args.mqtt_keepalive)

    t = threading.Thread(target=recorder_thread, args=(args.device_index,), daemon=True)
    t.start()

    ring = np.zeros(BLOCK_SAMPLES, dtype=np.float32)
    pos = 0

    time.sleep(BLOCK_SECONDS)
    while True:
        while not audio_q.empty():
            chunk = audio_q.get()
            n = len(chunk)
            if n >= BLOCK_SAMPLES:
                ring[:] = chunk[-BLOCK_SAMPLES:]
                pos = 0
            else:
                end = pos + n
                if end <= BLOCK_SAMPLES:
                    ring[pos:end] = chunk
                else:
                    first = BLOCK_SAMPLES - pos
                    ring[pos:] = chunk[:first]
                    ring[: n - first] = chunk[first:]
                pos = (pos + n) % BLOCK_SAMPLES

        block_copy = ring.copy()
        image = generate_spectrogram_image(block_copy, SAMPLE_RATE)
        tensor = preprocess_image(image, args.img_size)

        debug_dir = Path("debug")
        if debug_dir.exists():
            image.save(debug_dir / "spectrogram_debug.png")

        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: tensor})[0]
        logits = np.asarray(logits, dtype=np.float32)
        probs = (1.0 / (1.0 + np.exp(-logits))).squeeze(0).tolist()

        payload = format_payload(classes, probs)
        if mqtt_client:
            publish_payload(mqtt_client, args.mqtt_topic, payload)
        else:
            print("\n----")
            for cls, prob in zip(classes, probs):
                print(f"{cls}: {prob:.4f}")

        ring.fill(0.0)
        pos = 0
        time.sleep(BLOCK_SECONDS)


if __name__ == "__main__":
    main()
