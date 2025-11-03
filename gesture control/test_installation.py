#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly.
"""

def test_imports():
    """Test importing all required packages."""
    print("🧪 Testing package imports...")
    print("-" * 30)
    
    packages = [
        ("OpenCV", "cv2"),
        ("MediaPipe", "mediapipe"),
        ("PyAutoGUI", "pyautogui"),
        ("NumPy", "numpy"),
        ("PyGetWindow", "pygetwindow")
    ]
    
    all_good = True
    
    for name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            all_good = False
    
    return all_good

def test_camera():
    """Test camera access."""
    print("\n📷 Testing camera access...")
    print("-" * 30)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera: OK")
            cap.release()
            return True
        else:
            print("❌ Camera: FAILED - Cannot open camera")
            return False
    except Exception as e:
        print(f"❌ Camera: FAILED - {e}")
        return False

def test_mediapipe():
    """Test MediaPipe hand detection."""
    print("\n🤚 Testing MediaPipe hand detection...")
    print("-" * 30)
    
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        print("✅ MediaPipe: OK")
        return True
    except Exception as e:
        print(f"❌ MediaPipe: FAILED - {e}")
        return False

def test_pyautogui():
    """Test PyAutoGUI functionality."""
    print("\n🖱️ Testing PyAutoGUI...")
    print("-" * 30)
    
    try:
        import pyautogui
        screen_size = pyautogui.size()
        print(f"✅ PyAutoGUI: OK - Screen size: {screen_size}")
        return True
    except Exception as e:
        print(f"❌ PyAutoGUI: FAILED - {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Hand Gesture Mouse Control - Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_camera,
        test_mediapipe,
        test_pyautogui
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! The system is ready to run.")
        print("\nTo start the gesture control system, run:")
        print("python gesture_control_main.py")
        return True
    else:
        print("❌ Some tests failed. Please install missing dependencies.")
        print("\nTo install dependencies, run:")
        print("pip install opencv-python mediapipe pyautogui numpy pygetwindow")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
