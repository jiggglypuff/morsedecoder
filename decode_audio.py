#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from scipy import signal

from main import Net, num_tags, prediction_to_str
from morse import SAMPLE_FREQ, get_spectrogram


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "001750.pt"


def decode_audio(input_path, model_path=DEFAULT_MODEL):
    """Decode a WAV audio file and return the predicted Morse text."""
    rate, data = scipy.io.wavfile.read(input_path)

    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) == 0:
        raise ValueError("The audio file contains no samples.")

    # Resample and rescale.
    length = len(data) / rate
    new_length = int(length * SAMPLE_FREQ)
    data = signal.resample(data, new_length)
    data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak == 0:
        raise ValueError("The audio file is silent and cannot be decoded.")
    data /= peak

    # Create spectrogram.
    spec = get_spectrogram(data)
    spectrogram_size = spec.shape[0]

    # Load model.
    device = torch.device("cpu")
    model = Net(num_tags, spectrogram_size)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    # Run model on audio.
    spec = torch.from_numpy(spec)
    spec = spec.permute(1, 0)
    spec = spec.unsqueeze(0)
    with torch.no_grad():
        y_pred = model(spec)

    # TODO: proper beam search.
    return prediction_to_str(torch.argmax(y_pred[0], 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode Morse-code audio to text.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="path to the trained model (default: bundled model)",
    )
    parser.add_argument("input", help="input WAV audio file")
    parser.add_argument(
        "--output", help="write the decoded text to this file instead of stdout"
    )
    args = parser.parse_args()

    decoded_text = decode_audio(args.input, args.model)
    if args.output:
        Path(args.output).write_text(decoded_text + "\n", encoding="utf-8")
    else:
        print(decoded_text)
