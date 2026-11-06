#!/usr/bin/env python3
"""Servo testing and calibration script for Raspberry Pi Zero W thin client."""

import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct execution without prior pip install
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from manipulator3.servo_tester import main

if __name__ == "__main__":
    sys.exit(main())
