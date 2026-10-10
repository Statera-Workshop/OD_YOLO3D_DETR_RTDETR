"""Evaluate yolo3d rtdetr on the KITTI test split only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.rtdetr.build import build_model

if __name__ == "__main__":
    main("evaluate", task="yolo3d", version="rtdetr", build_model=build_model)
