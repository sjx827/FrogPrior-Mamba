import argparse
import json
import os
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def calculate_rms(signal):
    return np.sqrt(np.mean(signal**2))


def list_wavs(path):
    return sorted([p for p in Path(path).iterdir() if p.suffix.lower() == ".wav"])


def load_wavs(paths, sr):
    wavs = []
    for path in paths:
        wav, _ = librosa.load(path, sr=sr)
        wavs.append(wav)
    return wavs


def mix_at_snr(clean, noise, snr):
    clean_rms = calculate_rms(clean) or 1e-8
    noise_rms = calculate_rms(noise) or 1e-8
    target_noise_rms = clean_rms / (10 ** (snr / 20))
    return clean + noise * (target_noise_rms / noise_rms)


def build_clean_track(frog_wavs, target_length, calls_per_chunk):
    clean = np.zeros(target_length, dtype=np.float32)
    grid_size = target_length // calls_per_chunk

    for grid_idx in range(calls_per_chunk):
        frog = random.choice(frog_wavs)
        max_start = grid_size - len(frog)
        if max_start > 0:
            start = grid_idx * grid_size + random.randint(0, max_start)
            clean[start:start + len(frog)] += frog
        else:
            start = grid_idx * grid_size
            clean[start:start + grid_size] += frog[:grid_size]

    return clean


def make_noise_chunks(noise_files, sr, target_length, chunks_per_noise):
    chunks = []
    for noise_file in noise_files:
        wav, _ = librosa.load(noise_file, sr=sr)
        stem = Path(noise_file).stem
        for idx in range(chunks_per_noise):
            start = idx * target_length
            end = start + target_length
            chunk = np.zeros(target_length, dtype=np.float32)
            if start < len(wav):
                part = wav[start:min(end, len(wav))]
                chunk[:len(part)] = part
            chunks.append((f"{stem}_part{idx + 1}", chunk))
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize frog-call enhancement data from clean calls and noise recordings."
    )
    parser.add_argument("--clean_dir", required=True, help="Directory containing clean frog-call wav files.")
    parser.add_argument("--noise_dir", required=True, help="Directory containing airport-noise wav files.")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory.")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=10.0, help="Segment duration in seconds.")
    parser.add_argument("--snrs", type=int, nargs="+", default=[-15, -10, -5, 0, 5])
    parser.add_argument("--chunks_per_noise", type=int, default=6)
    parser.add_argument("--calls_per_chunk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    clean_files = list_wavs(args.clean_dir)
    noise_files = list_wavs(args.noise_dir)
    if not clean_files:
        raise FileNotFoundError(f"No wav files found in clean_dir: {args.clean_dir}")
    if not noise_files:
        raise FileNotFoundError(f"No wav files found in noise_dir: {args.noise_dir}")

    output_dir = Path(args.output_dir)
    out_clean_dir = output_dir / "clean"
    out_noisy_dir = output_dir / "noisy"
    out_clean_dir.mkdir(parents=True, exist_ok=True)
    out_noisy_dir.mkdir(parents=True, exist_ok=True)

    target_length = int(args.duration * args.sr)
    frog_wavs = load_wavs(clean_files, args.sr)
    noise_chunks = make_noise_chunks(noise_files, args.sr, target_length, args.chunks_per_noise)

    clean_json = {}
    noisy_json = {}
    file_idx = 0
    total = len(args.snrs) * len(noise_chunks)

    for snr in tqdm(args.snrs, desc="SNR"):
        for noise_name, noise in tqdm(noise_chunks, desc=f"snr={snr}", leave=False):
            clean = build_clean_track(frog_wavs, target_length, args.calls_per_chunk)
            mixed = mix_at_snr(clean, noise, snr)

            max_amp = np.max(np.abs(mixed))
            if max_amp > 0.95:
                scale = 0.95 / max_amp
                mixed = mixed * scale
                clean = clean * scale

            filename = f"mix_{noise_name}_snr{snr}_{file_idx:04d}.wav"
            clean_path = out_clean_dir / filename
            noisy_path = out_noisy_dir / filename

            sf.write(clean_path, clean, args.sr)
            sf.write(noisy_path, mixed, args.sr)
            clean_json[filename] = str(clean_path.resolve())
            noisy_json[filename] = str(noisy_path.resolve())
            file_idx += 1

    with open(output_dir / "train_clean.json", "w", encoding="utf-8") as f:
        json.dump(clean_json, f, indent=2, ensure_ascii=False)
    with open(output_dir / "train_noisy.json", "w", encoding="utf-8") as f:
        json.dump(noisy_json, f, indent=2, ensure_ascii=False)

    print(f"Generated {file_idx}/{total} mixtures under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
