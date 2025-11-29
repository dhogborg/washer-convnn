import argparse
import os
import queue
import threading
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
from PIL import Image
from scipy import signal

SAMPLE_RATE = 44100
CHANNELS = 1
BLOCK_SECONDS = 10
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_SECONDS

DEVICE_INDEX = 0  # <- set this to your USB input index

audio_q = queue.Queue()

def parse_args():
    parser = argparse.ArgumentParser(description="Capture rolling spectrogram snapshots")
    return parser.parse_args()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Status:", status)
    # Flatten to mono if needed
    data = indata.copy()
    if data.ndim > 1:
        data = np.mean(data, axis=1, keepdims=True)
    audio_q.put(data[:, 0])  # mono float32

def recorder_thread():
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        device=DEVICE_INDEX,
        dtype="float32",
    ):
        while True:
            time.sleep(0.1)

def generate_spectrogram(samples: np.ndarray, samplerate: int, out_path: str):
    _, _, Sxx = signal.spectrogram(samples, fs=samplerate, nperseg=1024, noverlap=512)
    sxx_db = 10 * np.log10(Sxx + 1e-12)

    min_val = np.min(sxx_db)
    max_val = np.max(sxx_db)
    if np.isclose(max_val, min_val):
        normalized = np.zeros_like(sxx_db)
    else:
        normalized = (sxx_db - min_val) / (max_val - min_val)

    # Map each spectrogram bin to a single grayscale pixel without interpolation.
    image_array = np.flipud((1.0 - normalized) * 255.0).astype(np.uint8)
    bitmap = Image.fromarray(image_array, mode="L")
    bitmap.save(out_path)

def main():
    # Start background recorder
    t = threading.Thread(target=recorder_thread, daemon=True)
    t.start()

    # Simple ring buffer
    ring = np.zeros(BLOCK_SAMPLES, dtype=np.float32)
    pos = 0

    time.sleep(BLOCK_SECONDS)
    while True:
        # Drain queue
        while not audio_q.empty():
            chunk = audio_q.get()
            n = len(chunk)
            if n >= BLOCK_SAMPLES:
                # If the chunk is bigger than our block, just keep the last BLOCK_SAMPLES
                ring[:] = chunk[-BLOCK_SAMPLES:]
                pos = 0
            else:
                end = pos + n
                if end <= BLOCK_SAMPLES:
                    ring[pos:end] = chunk
                else:
                    # wrap around
                    first = BLOCK_SAMPLES - pos
                    ring[pos:] = chunk[:first]
                    ring[: n - first] = chunk[first:]
                pos = (pos + n) % BLOCK_SAMPLES

        # Every 10 seconds, snapshot, write spectrogram, then reset ring
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = f"output/spectrogram_{ts}.png"
        block_copy = ring.copy()
        generate_spectrogram(block_copy, SAMPLE_RATE, out_path)
        print("Wrote", out_path)
        ring.fill(0.0)
        pos = 0
        time.sleep(BLOCK_SECONDS)

if __name__ == "__main__":
    args = parse_args()
    main()
