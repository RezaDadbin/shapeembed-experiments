import re
import shutil
import zipfile
import urllib.request
from pathlib import Path

from config import DATA_RAW

RAW = DATA_RAW
ZIP_PATH = RAW / "BBBC010_v1_foreground_eachworm.zip"
MASK_DIR = RAW / "BBBC010_v1_foreground_eachworm"

OUT_ROOT = RAW / "bbbc010"
LIVE_DIR = OUT_ROOT / "live"
DEAD_DIR = OUT_ROOT / "dead"

URL = "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip"

VALID_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def download_if_needed():
    RAW.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"Zip already exists: {ZIP_PATH}")
        return

    print("Downloading BBBC010 each-worm masks...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print(f"Downloaded: {ZIP_PATH}")


def unzip_if_needed():
    if MASK_DIR.exists() and any(MASK_DIR.rglob("*")):
        print(f"Unzipped folder already exists: {MASK_DIR}")
        return

    print("Unzipping...")
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(MASK_DIR)

    print(f"Unzipped to: {MASK_DIR}")


def prepare_output_dirs():
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    DEAD_DIR.mkdir(parents=True, exist_ok=True)


def extract_column_number(path: Path):
    """
    Extract the well column number from BBBC010 filenames.

    Official rule:
        columns 01-12 -> live
        columns 13-24 -> dead

    This accepts common well patterns such as:
        A01, B13, P24
        A_01, B-13
        rowA_col01
        C01, D24
    """

    text = str(path).lower()

    patterns = [
        r"(?<![a-z0-9])([a-p])[_\-]?([0-9]{1,2})(?![a-z0-9])",
        r"row[_\-]?([a-p]).*col(?:umn)?[_\-]?([0-9]{1,2})",
        r"col(?:umn)?[_\-]?([0-9]{1,2})",
    ]

    columns = []

    for pat in patterns:
        for m in re.finditer(pat, text):
            if len(m.groups()) == 2:
                col = int(m.group(2))
            else:
                col = int(m.group(1))

            if 1 <= col <= 24:
                columns.append(col)

    columns = sorted(set(columns))

    if len(columns) == 1:
        return columns[0]

    return None


def organize():
    files = [
        p for p in MASK_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]

    print(f"Found mask files: {len(files)}")

    if len(files) == 0:
        raise RuntimeError(
            f"No image files found inside {MASK_DIR}. "
            f"The zip may have extracted somewhere unexpected."
        )

    prepare_output_dirs()

    copied_live = 0
    copied_dead = 0
    skipped = []

    for p in sorted(files):
        col = extract_column_number(p)

        if col is None:
            skipped.append(p)
            continue

        if 1 <= col <= 12:
            dst_dir = LIVE_DIR
            copied_live += 1
        else:
            dst_dir = DEAD_DIR
            copied_dead += 1

        dst = dst_dir / p.name

        # Stable names make reruns idempotent, including duplicate basenames.
        import hashlib
        digest = hashlib.sha256(p.relative_to(MASK_DIR).as_posix().encode()).hexdigest()[:12]
        dst = dst_dir / f"{p.stem}_{digest}{p.suffix}"

        shutil.copy2(p, dst)

    print()
    print("Organization finished.")
    print(f"Live copied: {copied_live}")
    print(f"Dead copied: {copied_dead}")
    print(f"Skipped: {len(skipped)}")
    print()
    print("Expected ShapeEmbed paper counts approximately:")
    print("Live: 768")
    print("Dead: 639")
    print("Total: 1407")

    if skipped:
        print()
        print("First skipped files:")
        for p in skipped[:20]:
            print("  ", p)


if __name__ == "__main__":
    download_if_needed()
    unzip_if_needed()
    organize()
