# Hand Gesture Mouse Control - Installation Guide

## 🚀 Quick Start Options

### Option 1: Simple Version (Recommended for Quick Testing)
```bash
# Install basic dependencies
pip install opencv-python pyautogui

# Run simplified version
python simple_gesture_control.py
```

### Option 2: Full Version (With MediaPipe)
```bash
# Install all dependencies
pip install opencv-python mediapipe pyautogui numpy pygetwindow

# Run full version
python gesture_control_main.py
```

### Option 3: Automatic Installation
```bash
# Run the installer
python install_and_run.py
```

## 📋 System Requirements

- **Python**: 3.9 or higher
- **Camera**: Webcam with minimum 640x480 resolution
- **OS**: Windows 10+, macOS 12+, or Linux Ubuntu 20.04+

## 🔧 Installation Methods

### Method 1: Using pip (Standard)
```bash
pip install opencv-python mediapipe pyautogui numpy pygetwindow
```

### Method 2: Using conda (Alternative)
```bash
conda install opencv mediapipe pyautogui numpy
pip install pygetwindow
```

### Method 3: Manual Installation
If pip fails, try installing packages individually:
```bash
pip install opencv-python
pip install pyautogui
pip install numpy
pip install pygetwindow
# MediaPipe might need special handling - see troubleshooting below
```

## 🎮 Available Programs

### 1. `simple_gesture_control.py` - Basic Version
- **Features**: Mouse control using motion detection
- **Dependencies**: OpenCV, PyAutoGUI only
- **Best for**: Quick testing, systems with limited resources

**Controls:**
- Move hand to control mouse cursor
- Quick hand movement = mouse click
- Press 'Q' to quit

### 2. `gesture_control_main.py` - Full Version
- **Features**: Advanced gesture recognition with MediaPipe
- **Dependencies**: All packages required
- **Best for**: Full functionality with all gesture types

**Controls:**
- Index finger pointing: Move mouse cursor
- Index finger shake: Mouse click
- Open hand → Fist: Move active window
- Four fingers swipe up/down: Scroll
- Four fingers swipe left/right: Switch desktop
- Press 'Q' to quit

## 🛠️ Troubleshooting

### Problem: MediaPipe Installation Fails
**Solution**: MediaPipe can be tricky on some systems. Try:
```bash
# Option 1: Use conda
conda install -c conda-forge mediapipe

# Option 2: Install specific version
pip install mediapipe==0.10.7

# Option 3: Use the simple version instead
python simple_gesture_control.py
```

### Problem: Camera Not Detected
**Solutions**:
1. Check camera permissions in system settings
2. Try different camera index: Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`
3. Close other applications using the camera

### Problem: PyAutoGUI Permission Errors
**Solutions**:
- **Windows**: Run as Administrator
- **macOS**: Grant accessibility permissions in System Preferences
- **Linux**: Install additional packages: `sudo apt-get install python3-tk python3-dev`

### Problem: Import Errors
**Solution**: Check Python version and install missing packages:
```bash
python --version  # Should be 3.9+
pip list  # Check installed packages
```

## 🎯 Quick Test

Run this to test if everything is working:
```bash
python test_installation.py
```

## 📁 File Structure

```
camera tracker/
├── gesture_control_main.py      # Full version with MediaPipe
├── simple_gesture_control.py    # Basic version without MediaPipe
├── install_and_run.py           # Automatic installer
├── test_installation.py         # Installation tester
├── requirements.txt             # Dependencies list
├── run_gesture_control.bat      # Windows batch file
└── README_INSTALLATION.md       # This file
```

## 🚀 Getting Started

1. **Choose your version**:
   - For quick testing: Use `simple_gesture_control.py`
   - For full features: Use `gesture_control_main.py`

2. **Install dependencies** (see methods above)

3. **Run the program**:
   ```bash
   python simple_gesture_control.py
   # OR
   python gesture_control_main.py
   ```

4. **Follow on-screen instructions**

## 💡 Tips for Best Results

- **Lighting**: Ensure good lighting on your hands
- **Background**: Use a plain background
- **Distance**: Stay 2-3 feet from camera
- **Hand Position**: Keep palm facing camera
- **Calibration**: Test each gesture type before using

## 🆘 Need Help?

If you're still having issues:

1. Try the simple version first: `python simple_gesture_control.py`
2. Check the troubleshooting section above
3. Make sure your camera is working with other applications
4. Verify Python version is 3.9 or higher

## 🎉 Success!

Once everything is working, you'll have a fully functional hand gesture mouse control system!

- **Simple version**: Basic mouse control
- **Full version**: Advanced gestures with window management and desktop switching
