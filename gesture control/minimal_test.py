#!/usr/bin/env python3
"""
Minimal test to isolate camera and PyAutoGUI issues.
"""

import cv2
import pyautogui
import time

def test_basic_functionality():
    """Test basic camera and mouse functionality."""
    print("🧪 Minimal Functionality Test")
    print("=" * 40)
    
    # Test 1: Camera
    print("1. Testing camera...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✅ Frame read successfully: {frame.shape}")
            else:
                print("❌ Cannot read frames from camera")
                cap.release()
                return False
        else:
            print("❌ Cannot open camera")
            return False
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False
    
    # Test 2: PyAutoGUI
    print("\n2. Testing PyAutoGUI...")
    try:
        screen_size = pyautogui.size()
        print(f"✅ Screen size: {screen_size}")
        
        current_pos = pyautogui.position()
        print(f"✅ Current mouse position: {current_pos}")
        
        # Test mouse movement (small movement)
        pyautogui.moveRel(10, 10, duration=0.1)
        time.sleep(0.2)
        pyautogui.moveRel(-10, -10, duration=0.1)
        print("✅ Mouse movement test successful")
        
    except Exception as e:
        print(f"❌ PyAutoGUI error: {e}")
        return False
    
    # Test 3: OpenCV display
    print("\n3. Testing OpenCV display...")
    try:
        # Show a simple window
        cv2.imshow("Test Window", frame)
        print("✅ OpenCV window created")
        
        print("Press any key to continue...")
        cv2.waitKey(3000)  # Wait 3 seconds
        cv2.destroyAllWindows()
        print("✅ OpenCV display test successful")
        
    except Exception as e:
        print(f"❌ OpenCV display error: {e}")
        return False
    
    # Test 4: Camera loop
    print("\n4. Testing camera loop...")
    try:
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 30:  # Test for 30 frames
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame in loop")
                break
            
            frame_count += 1
            
            # Show frame
            cv2.imshow("Camera Test", frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit key pressed")
                break
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"✅ Camera loop successful: {frame_count} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
        
    except Exception as e:
        print(f"❌ Camera loop error: {e}")
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n🎉 All tests passed!")
    return True

def main():
    """Run the minimal test."""
    try:
        success = test_basic_functionality()
        if success:
            print("\n✅ Your system is ready for gesture control!")
            print("You can now run: python simple_gesture_control.py")
        else:
            print("\n❌ Some tests failed. Please check the errors above.")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
