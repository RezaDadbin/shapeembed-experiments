#!/usr/bin/env python3
"""Build publication tables from archived JSON records without retraining models."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "archived"


def tables():
    two = []
    for path in sorted((ARCHIVE / "2d").glob("*.json")):
        data = json.loads(path.read_text())
        if "official_classification" in data:
            row = data["official_classification"]["LogisticRegression"]
            dataset = "mnist"
            model = "vae (legacy)" if "shapeembed" in path.name else "ae (legacy)"
        else:
            row = data
            dataset = path.name.split("_")[0]
            model = {"original": "vae", "ae": "ae", "mlp_decoder": "mlp_decoder"}.get(data.get("model_type"), "mlp_decoder")
        two.append(dict(dataset=dataset, model=model, macro_f1_mean=row["macro_f1_mean"],
                        macro_f1_std=row["macro_f1_std"], accuracy_mean=row["accuracy_mean"],
                        source=path.relative_to(ROOT).as_posix()))
    three = []
    for path in sorted((ARCHIVE / "3d").glob("*_evaluation.json")):
        data = json.loads(path.read_text())
        keys = ("run", "cv_accuracy_mean", "cv_accuracy_std", "latent_drift_under_permutation",
                "accuracy_with_permuted_inputs", "test_recon_error", "test_edm_residual")
        three.append({key: data[key] for key in keys})
    return two, three


def render_csv(rows):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def check_sources():
    for entry in json.loads((ARCHIVE / "manifest.json").read_text()):
        path = ROOT / entry["file"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["published_sha256"]:
            raise SystemExit(f"Archived asset changed: {entry['file']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify source hashes and committed tables")
    args = parser.parse_args()
    check_sources()
    two, three = tables()
    outputs = {
        "results/2d_summary.csv": render_csv(two),
        "results/3d_summary.csv": render_csv(three),
    }
    for name, content in outputs.items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Outdated table: {name}; run python scripts/summarize_results.py")
        else:
            path.write_text(content)
    print(f"Verified {len(two)} 2D runs and {len(three)} 3D runs; " + ("tables match." if args.check else "tables written."))


if __name__ == "__main__":
    main()
