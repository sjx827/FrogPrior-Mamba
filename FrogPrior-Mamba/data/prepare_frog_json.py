import argparse
import json
from pathlib import Path


def wav_files(directory):
    directory = Path(directory)
    return sorted(
        str(p.resolve())
        for p in directory.rglob("*.wav")
        if p.is_file() and "Zone.Identifier" not in p.name
    )


def infer_dir(base, split, kind):
    candidates = [
        Path(base) / split / kind,
        Path(base) / "processed" / split / kind,
        Path(base) / kind,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def write_json(paths, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)


def make_split(source_base, out_dir, split):
    clean_dir = infer_dir(source_base, split, "clean")
    noisy_dir = infer_dir(source_base, split, "noisy")
    if clean_dir is None or noisy_dir is None:
        print(f"[{split}] skip: clean_dir={clean_dir}, noisy_dir={noisy_dir}")
        return

    clean_paths = wav_files(clean_dir)
    noisy_paths = wav_files(noisy_dir)
    clean_names = {Path(p).name for p in clean_paths}
    noisy_names = {Path(p).name for p in noisy_paths}

    paired_clean = [p for p in clean_paths if Path(p).name in noisy_names]
    paired_noisy = [p for p in noisy_paths if Path(p).name in clean_names]

    write_json(paired_clean, Path(out_dir) / f"{split}_clean.json")
    write_json(paired_noisy, Path(out_dir) / f"{split}_noisy.json")
    print(f"[{split}] clean={len(paired_clean)}, noisy={len(paired_noisy)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare clean/noisy JSON files for FrogPrior-Mamba.")
    parser.add_argument("--source_base", required=True, help="Dataset root containing split/clean and split/noisy folders.")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], choices=["train", "valid", "test"])
    args = parser.parse_args()

    for split in args.splits:
        make_split(args.source_base, args.out_dir, split)


if __name__ == "__main__":
    main()
