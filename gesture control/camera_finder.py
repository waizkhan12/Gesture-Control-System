#!/usr/bin/env python3
"""
Simple camera finder and working gesture control creator.
"""

import cv2
import pyautogui
import numpy as np
import time

def find_working_camera():
    """Find a working camera."""
    print("🔍 Looking for working cameras...")
    
    for camera_index in range(5):
        print(f"Testing camera {camera_index}...")
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Camera {camera_index} works! Frame size: {frame.shape}")
                    cap.release()
                    return camera_index
                else:
                    print(f"❌ Camera {camera_index} opened but no frames")
            cap.release()
        except Exception as e:
            print(f"❌ Camera {camera_index} error: {e}")
    
    return None

def create_working_gesture_control(camera_index):
    """Create a working gesture control file."""
    code = f'''#!/usr/bin/env python3
"""
Working Hand Gesture Mouse Control
Camera Index: {camera_index}
"""

import cv2
import pyautogui
import numpy as np
import time

def main():
    print("🚀 Working Hand Gesture Mouse Control")
    print("=" * 40)
    
    # Initialize camera
    cap = cv2.VideoCapture({camera_index})
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_index}")
        return
    
    print(f"✅ Camera {camera_index} initialized")
    
    # Get screen size
    screen_width, screen_height = pyautogui.size()
    print(f"📺 Screen size: {{screen_width}}x{{screen_height}}")
    
    # Motion detection variables
    prev_frame = None
    prev_center = None
    last_click_time = 0
    click_debounce = 0.5
    click_threshold = 50
    
    print("\\n🎮 Controls:")
    print("- Move your hand to control mouse")
    print("- Quick hand movement = click")
    print("- Press 'Q' to quit")
    print("-" * 40)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Motion detection
            if prev_frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_frame, gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 1000:
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                            center = (center_x, center_y)
                            
                            # Map to screen coordinates
                            norm_x = center_x / frame.shape[1]
                            norm_y = center_y / frame.shape[0]
                            screen_x = int(norm_x * screen_width)
                            screen_y = int(norm_y * screen_height)
                            
                            # Move mouse
                            try:
                                pyautogui.moveTo(screen_x, screen_y, duration=0)
                            except:
                                pass
                            
                            # Check for click
                            current_time = time.time()
                            if (prev_center is not None and 
                                current_time - last_click_time > click_debounce):
                                
                                dx = center[0] - prev_center[0]
                                dy = center[1] - prev_center[1]
                                distance = np.sqrt(dx*dx + dy*dy)
                                
                                if distance > click_threshold:
                                    try:
                                        pyautogui.click()
                                        print("🖱️ Click!")
                                        last_click_time = current_time
                                    except:
                                        pass
                            
                            prev_center = center
                            
                            # Draw on frame
                            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                            cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Draw instructions
            cv2.putText(frame, "Move hand to control mouse", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Quick movement = click", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'Q' to quit", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Working Gesture Control", frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
                
    except KeyboardInterrupt:
        print("\\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {{e}}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
'''
    
    with open("working_gesture_control.py", "w") as f:
        f.write(code)
    
    print(f"✅ Created working_gesture_control.py with camera {camera_index}")

def main():
    """Main function."""
    print("🔧 Camera Finder & Gesture Control Creator")
    print("=" * 50)
    
    # Find working camera
    working_camera = find_working_camera()
    
    if working_camera is not None:
        # Create working version
        create_working_gesture_control(working_camera)
        
        print(f"\\n🎉 Success! Found working camera: {working_camera}")
        print("\\n🚀 To run the gesture control:")
        print("python working_gesture_control.py")
        
    else:
        print("\\n❌ No working cameras found!")
        print("\\n🔧 Try these solutions:")
        print("1. Close other apps using the camera")
        print("2. Check camera permissions in Windows Settings")
        print("3. Restart your computer")
        print("4. Try a different camera")

if __name__ == "__main__":
    main()
