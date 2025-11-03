#!/usr/bin/env python3
"""
Installation and launcher script for Hand Gesture Mouse Control.
This script will install required dependencies and run the gesture control system.

---
Alfred Agent Integration (gesture_cursor_control)
Version: 1.0.0 | Author: Alfred AI | Date: 2025-09-18
Description: AI agent controlling cursor and window actions via webcam gestures.

Gesture Mappings:
    - One finger move: Move cursor (index tip, smoothing, Mediapipe)
    - Two fingers drag: Thumb & index extended = drag/select (keep debounce/velocity)
    - Pinch zoom: Thumb-index pinch distance mapped to zoom/scroll (min: 0.02, max: 0.15, sensitivity: 2.0)
    - Full hand Alt+Tab: All fingers extended, hold >300ms triggers Alt+Tab (no conflict)
    - Full hand swipe: Swipe left/right with full hand = Alt+Left/Right (threshold: 0.05, history: 5)

Safety:
    - No actions if no hand detected
    - Reset smoothing on hand loss
    - Debounce all gestures

Logging:
    - Log gesture detection and actions
    - Warn on detection failure or conflicts

Instructions:
    - Do not modify existing code structure or functionalities
    - Integrate gestures cleanly with current mouse movement and interaction system
    - Ensure gestures are responsive and precise, minimizing false positives
    - All gesture actions should supplement, not override, current behavior
---
"""

import subprocess
import sys
import os


def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False


def main():
    print("🚀 Hand Gesture Mouse Control - Installation & Launcher")
    print("=" * 60)
    # Only show minimal diagnostics
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Required packages
    packages = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("pyautogui", "pyautogui"),
        ("numpy", "numpy"),
        ("pygetwindow", "pygetwindow")
    ]

    print("\n📦 Checking and installing required packages...")
    print("-" * 40)

    missing_packages = []
    for package, import_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)

    if missing_packages:
        print(
            f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        print("-" * 40)
        for package in missing_packages:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package])
                print(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                print("Please install manually using: pip install " + package)
                return False

    print("\n✅ All packages are installed!")

    # Check if main script exists
    if not os.path.exists("gesture_control_main.py"):
        print("❌ gesture_control_main.py not found!")
        return False

    print("\n🎮 Starting Hand Gesture Mouse Control...")
    print("=" * 60)
    print("Controls:")
    print("• Index finger pointing: Move mouse cursor")
    print("• Index finger shake: Mouse click")
    print("• Open hand → Fist: Move active window")
    print("• Four fingers swipe up/down: Scroll")
    print("• Four fingers swipe left/right: Switch desktop")
    print("• Press 'Q' to quit")
    print("=" * 60)

    try:
        # Run the main program as a subprocess for isolation
        result = subprocess.run(
            [sys.executable, "gesture_control_main.py"], check=False)
        if result.returncode != 0:
            print(
                f"❌ gesture_control_main.py exited with code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Program completed successfully!")
