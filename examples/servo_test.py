#!/usr/bin/env python3
"""Servo testing and calibration script for Raspberry Pi Zero W thin client."""

import sys
from manipulator3.servo_tester import main

if __name__ == "__main__":
    sys.exit(main())
