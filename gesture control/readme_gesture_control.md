# Webcam Gesture Mouse & Window Control

A production-grade prototype that maps hand gestures from webcam input to OS-level mouse and window actions using MediaPipe and OpenCV.

## Features

- **Gesture Recognition**: Open hand, fist, and index pointing detection
- **Mouse Control**: Drag operations and shake-to-click functionality  
- **Window Management**: Move active windows using hand gestures
- **Cross-Platform**: Windows, macOS, and Linux support
- **Performance Optimized**: <50ms latency target, efficient processing
- **Configurable**: Adjustable thresholds, sensitivity, and smoothing
- **Production Ready**: Error handling, logging, and unit tests

## System Requirements

- **Python**: 3.9 or higher
- **Camera**: Webcam with minimum 640x480 resolution
- **OS**: Windows 10+, macOS 10.14+, or Linux with X11/Wayland
- **RAM**: Minimum 4GB recommended
- **CPU**: Modern multi-core processor recommended

## Installation

### Quick Install

```bash
# Install required dependencies
pip install opencv-python mediapipe pyautogui numpy pygetwindow

# Optional platform-specific packages
# Windows:
pip install pywin32 pywinauto

# macOS:
pip install pyobjc-core pyobjc-framework-Quartz
```

### Development Install

```bash
# Clone or download the gesture control files
# Install dependencies
pip install -r requirements.txt

# Install test dependencies
pip install pytest pytest-cov
```

## Permissions Setup

### Windows

1. **Camera Access**: 
   - Go to Settings > Privacy & Security > Camera
   - Enable "Let apps access your camera"
   - Enable for Python/your IDE

2. **Mouse Control**:
   - Run Command Prompt as Administrator
   - The application will request permission on first mouse action

3. **Window Control**:
   - Usually works by default
   - Some antivirus software may require exceptions

### macOS

1. **Camera Access**:
   - System Preferences > Security & Privacy > Camera
   - Add Terminal/Python/your IDE to allowed applications

2. **Accessibility/Input Monitoring**:
   - System Preferences > Security & Privacy > Privacy tab
   - Select "Accessibility" and add your application
   - Select "Input Monitoring" and add your application
   - **Important**: You may need to restart your application after granting permissions

3. **Screen Recording** (if using window control):
   - System Preferences > Security & Privacy > Privacy tab
   - Select "Screen Recording" and add your application

### Linux

1. **Camera Access**:
   - Ensure user is in `video` group: `sudo usermod -a -G video $USER`
   - Log out and back in

2. **Display Server Compatibility**:
   - X11: Should work out of the box
   - Wayland: May have limitations with window control

## Usage

### Basic Operation

```bash
# Run the gesture control system
python gesture_control_main.py
```

**Controls**:
- Press `Q` to quit
- Use `Ctrl+C` for emergency stop

### Gesture Commands

#### Mouse Control

1. **Drag Mode**:
   - Extend index finger while curling other fingers
   - Move index finger to control cursor
   - Cursor follows index fingertip with smoothing
   - Mouse button is held down during pointing gesture

2. **Click Action**:
   - While in any mode, make a quick horizontal shake motion
   - System detects rapid velocity changes (>1200 px/s)
   - Debounced to prevent accidental multiple clicks

#### Window Management

1. **Window Move Mode**:
   - Show open hand (all fingers extended) for ~300ms
   - Quickly make a fist within 1 second
   - Active window will be brought to front
   - Hand center position controls window movement
   - Show open hand again to exit mode

### Configuration

The system uses `gesture_config.json` for customization:

```json
{
  "cam_width": 640,
  "cam_height": 480,
  "fps_limit": 30,
  "open_threshold": 0.15,
  "fist_threshold": 0.06,
  "index_extension_threshold": 0.02,
  "smooth_alpha": 0.22,
  "shake_velocity_threshold": 1200,
  "shake_window_ms": 220,
  "click_debounce_ms": 350,
  "no_hand_timeout_ms": 2000,
  "log_level": "INFO"
}
```

**Key Parameters**:
- `smooth_alpha`: Lower values = more smoothing (0.1-0.3)
- `shake_velocity_threshold`: Higher = requires more vigorous shake
- `open_threshold/fist_threshold`: Adjust gesture sensitivity

## Calibration

### First Run

1. Launch the application
2. Position yourself 2-3 feet from camera
3. Ensure good lighting on your hands
4. Test each gesture type:
   - Open hand: All fingers clearly extended
   - Fist: All fingers curled tightly
   - Pointing: Only index finger extended

### Fine-Tuning

If gestures aren't detected reliably:

1. **Lighting**: Ensure even lighting, avoid backlighting
2. **Distance**: Stay within 2-4 feet of camera
3. **Background**: Plain backgrounds work better
4. **Hand Position**: Keep palm generally facing camera
5. **Thresholds**: Adjust in config file if needed

### Performance Tuning

- **High CPU Usage**: Reduce `fps_limit` or `cam_width/cam_height`
- **Laggy Response**: Increase `smooth_alpha` or reduce smoothing
- **False Clicks**: Increase `shake_velocity_threshold`
- **Missed Gestures**: Decrease threshold values

## Testing

### Unit Tests

```bash
# Run all tests
python -m pytest test_gesture_control.py -v

# Run with coverage
python -m pytest test_gesture_control.py --cov=gesture_control_main

# Run specific test class
python -m pytest test_gesture_control.py::TestGestureDetector -v
```

### Manual Test Scenarios

1. **Drag Test**:
   - Point index finger, move around screen
   - Verify cursor follows smoothly
   - Check latency is acceptable (<100ms perceived)

2. **Click Test**:
   - Make small, quick horizontal shakes
   - Verify clicks are registered
   - Ensure no false positives during normal movement

3. **Window Move Test**:
   - Open hand → fist sequence
   - Verify window comes to front and moves with hand
   - Test exit with open hand

### Performance Metrics

**Target Performance**:
- Click false positive rate: <5%
- Drag stability: <10px RMS jitter
- Processing latency: <50ms
- Frame rate: 30fps sustained

**Benchmark Command**:
```bash
python test_gesture_control.py  # Includes performance benchmarks
```

## Troubleshooting

### Common Issues

1. **"No camera found"**:
   - Check camera permissions
   - Verify camera isn't used by another app
   - Try different camera index (change cv2.VideoCapture(0) to cv2.VideoCapture(1))

2. **"Permission denied" for mouse control**:
   - Run as administrator (Windows)
   - Enable accessibility permissions (macOS)
   - Check user groups (Linux)

3. **Gestures not detected**:
   - Improve lighting conditions
   - Adjust camera angle
   - Check configuration thresholds
   - Ensure MediaPipe installation is complete

4. **High CPU usage**:
   - Reduce camera resolution in config
   - Lower frame rate limit
   - Close other applications

5. **Window control not working**:
   - Verify pygetwindow compatibility with your system
   - Check window manager compatibility (Linux)
   - Some applications may block external window manipulation

### Debug Mode

Enable detailed logging:

```python
# In gesture_control_main.py, modify config:
config.log_level = "DEBUG"
config.enable_file_logging = True
```

This creates `gesture_control.log` with detailed information.

### Platform-Specific Notes

**Windows**:
- Works best with Windows 10/11
- Some antivirus software may interfere
- UAC prompts may appear for mouse control

**macOS**:
- Requires explicit permission grants
- May need application restart after permission changes
- Window control limited by macOS security model

**Linux**:
- X11 generally works better than Wayland
- Window management varies by desktop environment
- Some distributions require additional packages

## Architecture

### Core Components

- `GestureController`: Main application coordinator
- `GestureDetector`: MediaPipe-based gesture recognition
- `CoordinateMapper`: Screen coordinate transformation and smoothing
- `MotionAnalyzer`: Velocity calculation and shake detection
- `WindowController`: OS window manipulation

### State Machine

```
IDLE → DRAG (index pointing)
IDLE → WINDOW_MOVE (open → fist)
DRAG → IDLE (stop pointing)
WINDOW_MOVE → IDLE (open hand)
```

### Performance Architecture

- Single-threaded design for simplicity
- Efficient NumPy operations for coordinate processing
- Minimal memory allocation in main loop
- Configurable frame rate limiting

## Development

### Code Style

- PEP 8 compliant
- Type hints throughout
- Comprehensive docstrings
- Error handling and logging

### Contributing

1. Follow existing code patterns
2. Add unit tests for new features
3. Update documentation
4. Test on multiple platforms
5. Performance test changes

### Extending

Common extension points:
- Add new gesture types in `GestureDetector`
- Implement custom smoothing algorithms in `CoordinateMapper`
- Add platform-specific optimizations
- Create gesture macros or sequences

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check troubleshooting section
2. Review logs in debug mode
3. Test with unit test suite
4. Verify platform-specific permissions

---

**Version**: 1.0.0  
**Last Updated**: September 2025  
**Tested Platforms**: Windows 11, macOS 13+, Ubuntu 22.04+