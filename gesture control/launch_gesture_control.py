#!/usr/bin/env python3
"""
Universal launcher for Hand Gesture Mouse Control.
Tries to run the full version first, falls back to simple version if needed.
"""

import sys
import subprocess
import os


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def run_simple_version():
    """Run the simple gesture control version."""
    print("🎮 Launching Simple Gesture Control...")
    print("(Basic motion detection - no MediaPipe required)")
    print("-" * 50)

    try:
        import simple_gesture_control
        simple_gesture_control.main()
        return True
    except Exception as e:
        print(f"❌ Simple version failed: {e}")
        return False


def run_full_version():
    """Run the full gesture control version."""
    print("🎮 Launching Full Gesture Control...")
    print("(Advanced MediaPipe-based gesture recognition)")
    print("-" * 50)

    try:
        import gesture_control_main
        gesture_control_main.main()
        return True
    except Exception as e:
        print(f"❌ Full version failed: {e}")
        return False


def main():
    """Main launcher function."""
    print("🚀 Hand Gesture Mouse Control - Universal Launcher")
    print("=" * 60)

    # Check what's available
    has_opencv = check_package("cv2")
    has_pyautogui = check_package("pyautogui")
    has_mediapipe = check_package("mediapipe")
    has_numpy = check_package("numpy")
    has_pygetwindow = check_package("pygetwindow")

    print("📦 Package Status:")
    print(f"  OpenCV: {'✅' if has_opencv else '❌'}")
    print(f"  PyAutoGUI: {'✅' if has_pyautogui else '❌'}")
    print(f"  MediaPipe: {'✅' if has_mediapipe else '❌'}")
    print(f"  NumPy: {'✅' if has_numpy else '❌'}")
    print(f"  PyGetWindow: {'✅' if has_pygetwindow else '❌'}")
    print()

    # Determine which version to run
    if has_opencv and has_pyautogui and has_mediapipe and has_numpy and has_pygetwindow:
        print("🎉 All packages available! Running full version...")
        if run_full_version():
            return
        else:
            print("\n⚠️ Full version failed, trying simple version...")

    if has_opencv and has_pyautogui:
        print("🎯 Running simple version (MediaPipe not available)...")
        if run_simple_version():
            return
        else:
            print("\n❌ Simple version also failed!")
    else:
        print("❌ Missing basic dependencies!")
        print("\n🔧 Please install required packages:")
        print("pip install opencv-python pyautogui")
        print("\nFor full features, also install:")
        print("pip install mediapipe numpy pygetwindow")
        return False

    print("\n❌ All versions failed to run!")
    return False


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n🆘 Need help? Check README_INSTALLATION.md")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
