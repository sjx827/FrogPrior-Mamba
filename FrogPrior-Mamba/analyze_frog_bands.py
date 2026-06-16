import argparse
import csv
from pathlib import Path

import librosa
import numpy as np


FREQ_BANDS = [
    ("low_noise", 0.0, 1000.0, 3.0),
    ("frog_main", 1529.2, 2500.0, 3.5),
    ("frog_core", 1668.3, 2050.0, 4.5),
    ("mid_transition", 2500.0, 4000.0, 1.5),
    ("high_harmonic", 4000.0, 4800.0, 1.5),
]


def band_energy_ratios(path, sr, n_fft, hop_length, win_length, compress_factor):
    wav, _ = librosa.load(path, sr=sr)
    spec = librosa.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    mag = np.abs(spec) ** compress_factor
    energy = mag ** 2
    total_energy = float(np.sum(energy) + 1e-12)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    rows = []
    for name, lo, hi, weight in FREQ_BANDS:
        mask = (freqs >= lo) & (freqs <= hi)
        ratio = float(np.sum(energy[mask]) / total_energy)
        rows.append(
            {
                "file": Path(path).name,
                "band": name,
                "low_hz": lo,
                "high_hz": hi,
                "weight": weight,
                "energy_ratio": ratio,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Analyze frog-call energy ratios in predefined frequency bands.")
    parser.add_argument("--input", required=True, help="A wav file or a directory containing wav files.")
    parser.add_argument("--output_csv", default="frog_band_energy.csv")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=100)
    parser.add_argument("--win_length", type=int, default=400)
    parser.add_argument("--compress_factor", type=float, default=0.3)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        wavs = sorted(input_path.glob("*.wav"))
    else:
        wavs = [input_path]

    if not wavs:
        raise FileNotFoundError(f"No wav files found: {input_path}")

    all_rows = []
    for wav in wavs:
        all_rows.extend(
            band_energy_ratios(
                wav,
                args.sr,
                args.n_fft,
                args.hop_length,
                args.win_length,
                args.compress_factor,
            )
        )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "band", "low_hz", "high_hz", "weight", "energy_ratio"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
