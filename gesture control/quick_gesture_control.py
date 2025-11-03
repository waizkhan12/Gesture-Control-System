#!/usr/bin/env python3
"""
Quick Working Hand Gesture Mouse Control
This version is designed to work even with camera issues.
"""

import cv2
import pyautogui
import numpy as np
import datetime as time
import sys 

def find_working_camera():
    """Quickly find a working camera."""
    print("🔍 Finding working camera...")
    
    for i in range(3):  # Try cameras 0, 1, 2
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Quick test - try to read one frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Camera {i} works!")
                    return i, cap
                else:
                    cap.release()
            else:
                cap.release()
        except:
            pass
    
    print("❌ No working cameras found")
    return None, None

def main():
    """Main gesture control function."""
    print("🚀 Quick Hand Gesture Mouse Control")
    print("=" * 40)
    
    # Find working camera
    camera_index, cap = find_working_camera()
    if cap is None:
        print("\n❌ Cannot access camera!")
        print("\n🔧 Try these fixes:")
        print("1. Close other apps using camera (Zoom, Teams, etc.)")
        print("2. Run as Administrator")
        print("3. Check camera permissions in Windows Settings")
        print("4. Try: python -c \"import cv2; cap = cv2.VideoCapture(1); print(cap.isOpened())\"")
        return
    
    print(f"📷 Using camera {camera_index}")
    
    # Get screen size
    try:
        screen_width, screen_height = pyautogui.size()
        print(f"📺 Screen size: {screen_width}x{screen_height}")
    except Exception as e:
        print(f"❌ Cannot get screen size: {e}")
        return
    
    # Motion detection variables
    prev_frame = None
    prev_center = None
    last_click_time = 0
    click_debounce = 0.5
    click_threshold = 50
    
    print("\n🎮 Controls:")
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
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Motion detection
            if prev_frame is not None:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate difference
                diff = cv2.absdiff(prev_frame, gray)
                
                # Apply threshold
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    
                    if area > 1000:  # Minimum area threshold
                        # Calculate center
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
                            
                            # Clamp to screen bounds
                            screen_x = max(0, min(screen_x, screen_width - 1))
                            screen_y = max(0, min(screen_y, screen_height - 1))
                            
                            # Move mouse
                            try:
                                pyautogui.moveTo(screen_x, screen_y, duration=0)
                            except Exception as e:
                                print(f"Mouse move error: {e}")
                            
                            # Check for click gesture
                            current_time = time.time()
                            if (prev_center is not None and 
                                current_time - last_click_time > click_debounce):
                                
                                # Calculate distance moved
                                dx = center[0] - prev_center[0]
                                dy = center[1] - prev_center[1]
                                distance = np.sqrt(dx*dx + dy*dy)
                                
                                if distance > click_threshold:
                                    try:
                                        pyautogui.click()
                                        print("🖱️ Click!")
                                        last_click_time = current_time
                                    except Exception as e:
                                        print(f"Click error: {e}")
                            
                            # Update previous center
                            prev_center = center
                            
                            # Draw on frame
                            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            # Update previous frame
            prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Draw instructions
            cv2.putText(frame, "Move hand to control mouse", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Quick movement = click", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'Q' to quit", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Quick Gesture Control", frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
                
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
