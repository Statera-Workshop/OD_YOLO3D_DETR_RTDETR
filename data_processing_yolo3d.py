"""
Step 1 - data processing for the LiDAR BEV tasks (KITTI_5k).

    python data_processing_kitti5k.py --zip-dir ~/Downloads     # official KITTI object zips in DIR
    python data_processing_kitti5k.py --download                # try the official download links

KITTI_5k = the 7,481 labelled frames of the KITTI Object training set, split with seed 777 into
trainval 5,011 / test 2,470 (the lists are shipped in data_splits/KITTI_5k/, so the split is
reproduced exactly). Required official archives (http://www.cvlibs.net/datasets/kitti/eval_object.php,
registration and acceptance of the KITTI terms, CC BY-NC-SA 3.0):

    data_object_velodyne.zip   data_object_image_2.zip   data_object_calib.zip   data_object_label_2.zip

Only training/{velodyne, image_2, calib, label_2} of the listed frames is extracted to
$PHYSICAL_AI_ROOT/datasets/KITTI_5k/training/, plus ImageSets/{trainval,test}.txt and split_info.json.
image_2 is used by the camera 3D demo only.
"""

import argparse
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config as C

HERE = Path(__file__).resolve().parent
SPLITS = HERE / "data_splits" / "KITTI_5k"
PARTS = {  # archive -> (folder, extension)
    "data_object_velodyne.zip": ("velodyne", ".bin"),
    "data_object_image_2.zip": ("image_2", ".png"),
    "data_object_calib.zip": ("calib", ".txt"),
    "data_object_label_2.zip": ("label_2", ".txt"),
}
# Official links as published on the KITTI page (not verified by this script; if they fail,
# download the archives in a browser after logging in and use --zip-dir).
KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
EXPECTED = {"trainval": 5011, "test": 2470}


def read_ids(split):
    return [x.strip() for x in (SPLITS / "ImageSets" / f"{split}.txt").read_text().split() if x.strip()]


def complete(root, ids):
    return all((root / "training" / d / f"{i}{ext}").is_file() for d, ext in PARTS.values() for i in ids)


def download(url, dst):
    if dst.is_file():
        print(f"Found        : {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    print(f"Download     : {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r               {done / 1e9:6.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip-dir", default=str(C.DATASETS_ROOT / "downloads"), help="folder with the KITTI zips")
    ap.add_argument("--download", action="store_true", help="download missing zips from the official links")
    a = ap.parse_args()
    root, zip_dir = C.KITTI_ROOT, Path(a.zip_dir).expanduser()
    splits = {s: read_ids(s) for s in EXPECTED}
    wanted = set(splits["trainval"]) | set(splits["test"])
    print(f"KITTI_5k     : {root}")
    for s, ids in splits.items():
        note = "" if len(ids) == EXPECTED[s] else f"  (official KITTI_5k: {EXPECTED[s]})"
        print(f"split {s:8s}: {len(ids)} frames{note}")

    if complete(root, sorted(wanted)):
        print("Frames       : complete, nothing to extract")
    else:
        pat = re.compile(r"(?:^|/)training/(velodyne|image_2|calib|label_2)/(\d{6})\.(bin|png|txt)$")
        for name, (folder, ext) in PARTS.items():
            z = zip_dir / name
            if not z.is_file():
                if not a.download:
                    raise SystemExit(f"missing {z} (download it from the KITTI page, or use --download)")
                download(KITTI_URL + name, z)
            n = 0
            print(f"Extract      : {name} ({folder})")
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    g = pat.search(m)
                    if not g or g.group(1) != folder or g.group(2) not in wanted:
                        continue
                    out = root / "training" / folder / f"{g.group(2)}{ext}"
                    if not out.is_file():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(m) as src, open(out, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    n += 1
            print(f"               {n} files")

    (root / "ImageSets").mkdir(parents=True, exist_ok=True)
    for s in EXPECTED:
        shutil.copy2(SPLITS / "ImageSets" / f"{s}.txt", root / "ImageSets" / f"{s}.txt")
    if (SPLITS / "split_info.json").is_file():
        shutil.copy2(SPLITS / "split_info.json", root / "split_info.json")
    ok = complete(root, sorted(wanted))
    print("Data processing:", "done - next: python train_<task>_<version>.py" if ok else "INCOMPLETE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
