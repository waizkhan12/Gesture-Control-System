#!/usr/bin/env python3
"""
Alternative installation script for dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def main():
    """Install dependencies using alternative methods."""
    print("🚀 Installing Hand Gesture Mouse Control Dependencies")
    print("=" * 60)
    
    # Try different installation methods
    methods = [
        # Method 1: Direct pip install
        ("pip install opencv-python", "Installing OpenCV"),
        ("pip install mediapipe", "Installing MediaPipe"),
        ("pip install pyautogui", "Installing PyAutoGUI"),
        ("pip install numpy", "Installing NumPy"),
        ("pip install pygetwindow", "Installing PyGetWindow"),
        
        # Method 2: Try with --user flag
        ("pip install --user opencv-python", "Installing OpenCV (user)"),
        ("pip install --user mediapipe", "Installing MediaPipe (user)"),
        ("pip install --user pyautogui", "Installing PyAutoGUI (user)"),
        ("pip install --user numpy", "Installing NumPy (user)"),
        ("pip install --user pygetwindow", "Installing PyGetWindow (user)"),
    ]
    
    success_count = 0
    for command, description in methods:
        if run_command(command, description):
            success_count += 1
    
    print(f"\n📊 Installation Summary: {success_count}/{len(methods)} commands succeeded")
    
    # Test imports
    print("\n🧪 Testing imports...")
    packages = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("pyautogui", "PyAutoGUI"),
        ("numpy", "NumPy"),
        ("pygetwindow", "PyGetWindow")
    ]
    
    working_packages = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}: Working")
            working_packages.append(name)
        except ImportError:
            print(f"❌ {name}: Not working")
    
    print(f"\n📦 Working packages: {len(working_packages)}/{len(packages)}")
    
    if len(working_packages) >= 3:  # At least basic packages working
        print("\n🎉 Basic installation successful!")
        print("You can now try running: python gesture_control_main.py")
        return True
    else:
        print("\n❌ Installation incomplete. Please install manually:")
        print("pip install opencv-python mediapipe pyautogui numpy pygetwindow")
        return False

if __name__ == "__main__":
    main()
