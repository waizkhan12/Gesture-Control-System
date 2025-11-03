#!/usr/bin/env python3
"""
Robust Gesture Control - Cleaned and Fixed Version

Features:
- 1 finger -> move mouse only
- 2 fingers -> click only (no movement while clicking)
- Pinch (thumb+index) -> zoom (ctrl+scroll)
- Full hand open -> Alt+Tab (hold & cycle by moving hand)
- 4-finger swipe (index+middle+ring+pinky, thumb folded) -> desktop switch / scroll up/down
- Pointing at camera (forward) -> detected using landmark.z (stable across frames)
- Brightness/contrast preprocessing
- Debounce / gesture lock to avoid spamming
"""

from dataclasses import dataclass
from enum import Enum
import time
import math
import logging
import json
import os
from typing import Optional, Tuple, List, Dict, Any

try:
    import cv2
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'opencv-python'], check=True)
    import cv2
try:
    import mediapipe as mp
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'mediapipe'], check=True)
    import mediapipe as mp
try:
    import numpy as np
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'numpy'], check=True)
    import numpy as np
try:
    import pyautogui
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'pyautogui'], check=True)
    import pyautogui
try:
    import pygetwindow as gw
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'pygetwindow'], check=True)
    import pygetwindow as gw

# PyAutoGUI safety
pyautogui.PAUSE = 0.01
pyautogui.FAILSAFE = True

# ----------------------------
# Config and small utilities
# ----------------------------


class GestureState(Enum):
    IDLE = "idle"
    DRAG = "drag"
    WINDOW_MOVE = "window_move"
    CALIBRATING = "calibrating"


@dataclass
class Config:
    cam_width: int = 640
    cam_height: int = 480
    fps_limit: int = 30
    open_threshold: float = 0.15
    fist_threshold: float = 0.06
    index_extension_threshold: float = 0.02
    smooth_alpha: float = 0.22
    shake_velocity_threshold: int = 1200
    shake_window_ms: int = 220
    click_debounce_ms: int = 350
    no_hand_timeout_ms: int = 2000
    open_to_fist_timeout_ms: int = 1000
    swipe_threshold: float = 0.05
    swipe_history_size: int = 6
    scroll_sensitivity: float = 1.0
    enable_desktop_switching: bool = True
    log_level: str = "INFO"
    enable_file_logging: bool = False

    # pinch params (pixels and thresholds)
    pinch_distance_px: int = 40
    pinch_smooth_alpha: float = 0.35
    pinch_cooldown: float = 0.08

    @classmethod
    def load(cls, config_path: str = "gesture_config.json") -> "Config":
        """
        Load configuration from a JSON file. Returns a Config instance.
        """
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**data)
            except (IOError, ValueError) as e:
                logging.warning("Failed to load config, using defaults: %s", e)
        return cls()

    def save(self, config_path: str = "gesture_config.json") -> None:
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.__dict__, f, indent=2)
        except IOError as e:
            logging.error("Failed to save config: %s", e)


def adjust_brightness_contrast(frame: np.ndarray, alpha: float = 1.3, beta: int = 30) -> np.ndarray:
    """Adjust brightness/contrast quickly. Tune alpha/beta to taste."""
    if hasattr(cv2, 'convertScaleAbs'):
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    else:
        return frame


# ----------------------------
# Coordinate mapping and smoothing
# ----------------------------
class CoordinateMapper:
    def __init__(self, smooth_alpha: float = 0.22):
        self.smooth_alpha = smooth_alpha
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoothed_pos: Optional[np.ndarray] = None

    def normalize_to_screen(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        x = int(norm_x * self.screen_width)
        y = int(norm_y * self.screen_height)
        return max(0, min(x, self.screen_width - 1)), max(0, min(y, self.screen_height - 1))

    def smooth_position(self, x: int, y: int) -> Tuple[int, int]:
        new_pos = np.array([x, y], dtype=float)
        if self.smoothed_pos is None:
            self.smoothed_pos = new_pos
        else:
            self.smoothed_pos = (
                self.smooth_alpha * new_pos + (1 - self.smooth_alpha) * self.smoothed_pos)
        return int(self.smoothed_pos[0]), int(self.smoothed_pos[1])

    def reset_smoothing(self) -> None:
        self.smoothed_pos = None


# ----------------------------
# Motion analyzer for velocity / shake detection
# ----------------------------
class MotionAnalyzer:
    def __init__(self, window_size_ms: int = 220):
        self.window_size_ms = window_size_ms
        self.position_history: List[Tuple[float, int, int]] = []

    def add_position(self, x: int, y: int) -> None:
        timestamp = time.time() * 1000
        self.position_history.append((timestamp, x, y))
        self._prune_history(timestamp)

    def _prune_history(self, current_time_ms: float) -> None:
        cutoff_time = current_time_ms - self.window_size_ms
        self.position_history = [
            sample for sample in self.position_history if sample[0] > cutoff_time]

    def get_velocity(self) -> float:
        if len(self.position_history) < 2:
            return 0.0
        start_time, start_x, start_y = self.position_history[0]
        end_time, end_x, end_y = self.position_history[-1]
        dt = (end_time - start_time) / 1000.0
        if dt <= 0:
            return 0.0
        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.hypot(dx, dy)
        return distance / dt

    def clear_history(self) -> None:
        self.position_history.clear()


# ----------------------------
# Gesture detection helpers
# ----------------------------
class GestureDetector:
    # Duplicate is_fist removed

    def is_three_fingers(self, landmarks) -> bool:
        # Index, middle, ring extended, thumb and pinky folded
        finger_tips = [8, 12, 16]
        mcp_joints = [5, 9, 13]
        extended_count = 0
        for tip_id, mcp_id in zip(finger_tips, mcp_joints):
            tip = landmarks.landmark[tip_id]
            mcp = landmarks.landmark[mcp_id]
            if math.hypot(tip.x - mcp.x, tip.y - mcp.y) > self.config.open_threshold:
                extended_count += 1
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        pinky_tip = landmarks.landmark[20]
        pinky_mcp = landmarks.landmark[17]
        thumb_folded = (thumb_ip.x - thumb_tip.x) > 0.02
        pinky_folded = math.hypot(
            pinky_tip.x - pinky_mcp.x, pinky_tip.y - pinky_mcp.y) < self.config.fist_threshold * 2
        return extended_count == 3 and thumb_folded and pinky_folded

    def __init__(self, config: Config):
        self.config = config
        # stability buffers
        self.forward_point_frames = 0
        self.forward_point_required = 3

    def count_extended_fingers(self, landmarks) -> int:
        # Tip vs MCP distance heuristic
        finger_tips = [4, 8, 12, 16, 20]
        mcp_joints = [2, 5, 9, 13, 17]
        count = 0
        for tip_id, mcp_id in zip(finger_tips, mcp_joints):
            tip = landmarks.landmark[tip_id]
            mcp = landmarks.landmark[mcp_id]
            distance = math.hypot(tip.x - mcp.x, tip.y - mcp.y)
            if distance > self.config.open_threshold:
                count += 1
        return count

    def is_open_hand(self, landmarks) -> bool:
        return self.count_extended_fingers(landmarks) == 5

    def is_fist(self, landmarks) -> bool:
        return self.count_extended_fingers(landmarks) == 0

    def is_index_pointing(self, landmarks) -> bool:
        # index extended and other fingers folded (thumb ignored)
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        index_up = (
            index_pip.y - index_tip.y) > self.config.index_extension_threshold
        other_tips = [12, 16, 20]
        other_mcps = [9, 13, 17]
        others_curled = True
        for tip_id, mcp_id in zip(other_tips, other_mcps):
            tip = landmarks.landmark[tip_id]
            mcp = landmarks.landmark[mcp_id]
            if math.hypot(tip.x - mcp.x, tip.y - mcp.y) > self.config.fist_threshold * 2:
                others_curled = False
                break
        return index_up and others_curled

    def is_one_finger(self, landmarks) -> bool:
        # require index extended and other fingers folded for robustness
        return self.is_index_pointing(landmarks)

    def is_two_fingers_click(self, landmarks) -> bool:
        # Strict: index + middle extended, ring + pinky + thumb folded
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        ring_tip = landmarks.landmark[16]
        ring_mcp = landmarks.landmark[13]
        pinky_tip = landmarks.landmark[20]
        pinky_mcp = landmarks.landmark[17]
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        index_up = (
            index_pip.y - index_tip.y) > self.config.index_extension_threshold
        middle_up = (middle_pip.y -
                     middle_tip.y) > self.config.index_extension_threshold
        ring_folded = math.hypot(
            ring_tip.x - ring_mcp.x, ring_tip.y - ring_mcp.y) < self.config.fist_threshold * 2
        pinky_folded = math.hypot(
            pinky_tip.x - pinky_mcp.x, pinky_tip.y - pinky_mcp.y) < self.config.fist_threshold * 2
        thumb_folded = (thumb_ip.x - thumb_tip.x) > 0.02
        return index_up and middle_up and ring_folded and pinky_folded and thumb_folded

    def is_pinch(self, landmarks) -> bool:
        # pixel distance between thumb tip and index tip
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        # virtual pixel distance using camera resolution for thresholding
        dx = (thumb_tip.x - index_tip.x) * self.config.cam_width
        dy = (thumb_tip.y - index_tip.y) * self.config.cam_height
        dist_px = math.hypot(dx, dy)
        return dist_px < self.config.pinch_distance_px

    def is_full_hand(self, landmarks) -> bool:
        # strict: all five fingers extended
        return self.count_extended_fingers(landmarks) == 5

    def is_four_fingers(self, landmarks) -> bool:
        # Robust: index+middle+ring+pinky extended, thumb folded or angled inward
        finger_tips = [8, 12, 16, 20]
        mcp_joints = [5, 9, 13, 17]
        extended_count = 0
        for tip_id, mcp_id in zip(finger_tips, mcp_joints):
            tip = landmarks.landmark[tip_id]
            mcp = landmarks.landmark[mcp_id]
            if math.hypot(tip.x - mcp.x, tip.y - mcp.y) > self.config.open_threshold:
                extended_count += 1
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        thumb_folded = (
            thumb_ip.x - thumb_tip.x) > 0.01 or abs(thumb_tip.y - thumb_ip.y) < 0.04
        return extended_count == 4 and thumb_folded

    def get_hand_center(self, landmarks) -> Tuple[float, float]:
        x_coords = [l.x for l in landmarks.landmark]
        y_coords = [l.y for l in landmarks.landmark]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return center_x, center_y

    def is_pointing_forward(self, landmarks) -> bool:
        # Use z coordinates: index_tip.z significantly closer than index_pip.z and palm
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        wrist = landmarks.landmark[0]
        mcp_mid = landmarks.landmark[9]  # approximate mid palm
        avg_palm_z = (wrist.z + mcp_mid.z) / 2.0
        # require tip closer by threshold
        if (index_tip.z + 0.0) < (index_pip.z - 0.06) and (index_tip.z < avg_palm_z - 0.06):
            self.forward_point_frames += 1
        else:
            self.forward_point_frames = 0
        return self.forward_point_frames >= self.forward_point_required


# ----------------------------
# Window controller
# ----------------------------
class WindowController:
    def __init__(self, config: Config):
        self.config = config

    def get_active_window(self):
        try:
            window = gw.getActiveWindow()
            if window and window.isActive:
                return window
        except (AttributeError, TypeError) as e:
            logging.debug("Failed to get active window: %s", e)
        return None

    def move_window(self, window, x: int, y: int) -> bool:
        try:
            new_x = max(0, x - window.width // 2)
            new_y = max(0, y - window.height // 2)
            screen_w, screen_h = pyautogui.size()
            new_x = min(new_x, screen_w - window.width)
            new_y = min(new_y, screen_h - window.height)
            window.moveTo(new_x, new_y)
            return True
        except (AttributeError, TypeError) as e:
            logging.debug("Failed to move window: %s", e)
            return False

    def handle_scroll(self, direction: str, sensitivity: float = 1.0) -> bool:
        try:
            # amount is tuned to be noticeable but not huge
            amount = int(200 * sensitivity)
            if direction == "swipe_up":
                # positive -> scroll up
                pyautogui.scroll(amount)
            elif direction == "swipe_down":
                # negative -> scroll down
                pyautogui.scroll(-amount)
            else:
                return False
            return True
        except (AttributeError, TypeError, Exception) as e:
            logging.debug("Failed to scroll: %s", e)
            return False

    def switch_desktop(self, direction: str) -> bool:
        try:
            import sys
            if not self.config.enable_desktop_switching:
                return False
            if sys.platform == "win32":
                if direction == "swipe_left":
                    pyautogui.hotkey("ctrl", "win", "left")
                elif direction == "swipe_right":
                    pyautogui.hotkey("ctrl", "win", "right")
                else:
                    return False
            elif sys.platform == "darwin":
                if direction == "swipe_left":
                    pyautogui.hotkey("ctrl", "left")
                elif direction == "swipe_right":
                    pyautogui.hotkey("ctrl", "right")
                else:
                    return False
            elif sys.platform == "linux":
                if direction == "swipe_left":
                    pyautogui.hotkey("ctrl", "alt", "left")
                elif direction == "swipe_right":
                    pyautogui.hotkey("ctrl", "alt", "right")
                else:
                    return False
            else:
                logging.warning(
                    "Desktop switching not supported on %s", sys.platform)
                return False
            return True
        except (AttributeError, TypeError, Exception) as e:
            logging.debug("Failed to switch desktop: %s", e)
            return False


# ----------------------------
# Main controller
# ----------------------------
class GestureController:
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()

        # state & cooldowns
        self.last_full_hand_time = 0.0
        self.alt_tab_active = False
        self.last_full_hand_pos: Optional[Tuple[int, int]] = None
        self.last_tab_press_time = 0.0
        self.last_pinch_distance: Optional[float] = None
        self.pinch_smooth: Optional[float] = None

        self.coord_mapper = CoordinateMapper(config.smooth_alpha)
        self.motion_analyzer = MotionAnalyzer(config.shake_window_ms)
        self.gesture_detector = GestureDetector(config)
        self.window_controller = WindowController(config)

        self.current_state = GestureState.IDLE
        self.last_click_time = 0.0
        self.last_hand_seen_time = time.time()
        self.open_hand_start_time = 0.0
        self.is_dragging = False
        self.target_window = None

        # swipe history now stores tuples of (avg_x, avg_y, timestamp)
        self.swipe_positions: List[Tuple[float, float, float]] = []
        self.last_swipe_time = 0.0
        self.last_swipe_direction: Optional[str] = None

        self.gesture_locked = False
        self.last_action_time = 0.0
        self.action_cooldown = 0.5

        # mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.55, min_tracking_confidence=0.55)
        self.mp_drawing = mp.solutions.drawing_utils

        self.cap = None
        self.running = False

    def setup_logging(self) -> None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=getattr(
            logging, self.config.log_level.upper()), format=log_format)
        if self.config.enable_file_logging:
            fh = logging.FileHandler("gesture_control.log")
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)

    def initialize_camera(self) -> bool:
        try:
            # try common device indices (0..3) and pick the first valid camera
            for i in range(0, 4):
                cap = cv2.VideoCapture(i)
                if cap is None or not cap.isOpened():
                    if cap:
                        cap.release()
                    continue
                # found camera
                self.cap = cap
                logging.info("Using camera index %d", i)
                break
            if not self.cap:
                logging.error("No camera found")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.cam_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.cam_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps_limit)
            return True
        except (AttributeError, TypeError, ValueError) as e:
            logging.error("Camera initialization failed: %s", e)
            return False

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        # Preprocess frame: flip & adjust brightness/contrast
        frame = cv2.flip(frame, 1)
        frame = adjust_brightness_contrast(frame, alpha=1.2, beta=30)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            # Gesture confidence check
            confidence = 0.0
            if hasattr(results, 'multi_handedness') and results.multi_handedness:
                confidence = results.multi_handedness[0].classification[0].score
            if confidence < 0.8:
                self.motion_analyzer.clear_history()
                self.coord_mapper.reset_smoothing()
                return None

            landmarks = results.multi_hand_landmarks[0]
            self.last_hand_seen_time = time.time()

            # index for pointer
            index_tip = landmarks.landmark[8]
            screen_x, screen_y = self.coord_mapper.normalize_to_screen(
                index_tip.x, index_tip.y)
            # keep both raw and smoothed positions (use raw for drag to reduce lag)
            raw_x, raw_y = screen_x, screen_y
            smooth_x, smooth_y = self.coord_mapper.smooth_position(
                screen_x, screen_y)

            self.motion_analyzer.add_position(smooth_x, smooth_y)
            velocity = self.motion_analyzer.get_velocity()

            # gesture detections
            is_one = self.gesture_detector.is_one_finger(landmarks)
            is_two_click = self.gesture_detector.is_two_fingers_click(
                landmarks)
            is_pinch = self.gesture_detector.is_pinch(landmarks)
            is_full = self.gesture_detector.is_full_hand(landmarks)
            is_four = self.gesture_detector.is_four_fingers(landmarks)
            is_point_forward = self.gesture_detector.is_pointing_forward(
                landmarks)
            is_fist = self.gesture_detector.is_fist(landmarks)
            is_three = self.gesture_detector.is_three_fingers(landmarks)

            # swipe detection: when four fingers (exclude thumb). Track both axes.
            swipe_direction = None
            if is_four:
                finger_tips = [8, 12, 16, 20]
                avg_x = sum(landmarks.landmark[t].x for t in finger_tips) / 4.0
                avg_y = sum(landmarks.landmark[t].y for t in finger_tips) / 4.0
                self.swipe_positions.append((avg_x, avg_y, time.time()))
                if len(self.swipe_positions) > self.config.swipe_history_size:
                    self.swipe_positions.pop(0)
                # detect meaningful movement across stored history
                if len(self.swipe_positions) >= 2:
                    start_x, start_y, _ = self.swipe_positions[0]
                    end_x, end_y, _ = self.swipe_positions[-1]
                    dx = end_x - start_x
                    dy = end_y - start_y
                    abs_dx = abs(dx)
                    abs_dy = abs(dy)
                    # choose dominant axis and threshold
                    if abs_dx > abs_dy and abs_dx > self.config.swipe_threshold:
                        swipe_direction = "swipe_left" if dx < 0 else "swipe_right"
                    elif abs_dy >= abs_dx and abs_dy > self.config.swipe_threshold:
                        swipe_direction = "swipe_down" if dy > 0 else "swipe_up"
                    if swipe_direction:
                        self.swipe_positions.clear()
            else:
                self.swipe_positions.clear()

            # pinch distance (pixels) for zoom mapping
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            pinch_dx = (thumb_tip.x - index_tip.x) * self.config.cam_width
            pinch_dy = (thumb_tip.y - index_tip.y) * self.config.cam_height
            pinch_distance = math.hypot(pinch_dx, pinch_dy)

            center_x, center_y = self.gesture_detector.get_hand_center(
                landmarks)
            center_screen_x, center_screen_y = self.coord_mapper.normalize_to_screen(
                center_x, center_y)

            return {
                "hand_landmarks": landmarks,
                "index_pos": (smooth_x, smooth_y),
                "index_pos_raw": (raw_x, raw_y),
                "hand_center": (center_screen_x, center_screen_y),
                "velocity": velocity,
                "gestures": {
                    "one_finger": is_one,
                    "two_fingers_click": is_two_click,
                    "pinch": is_pinch,
                    "full_hand": is_full,
                    "four_fingers": is_four,
                    "point_forward": is_point_forward,
                    "fist": is_fist,
                    "three_fingers": is_three,
                },
                "swipe_direction": swipe_direction,
                "pinch_distance": pinch_distance,
            }
        else:
            self.motion_analyzer.clear_history()
            self.coord_mapper.reset_smoothing()
            return None

    def _lock_gesture(self, current_time: float) -> None:
        self.gesture_locked = True
        self.last_action_time = current_time

    def handle_gesture_logic(self, detection_data: Optional[Dict[str, Any]]) -> None:
        current_time = time.time()

        # unlock after cooldown
        if self.gesture_locked and (current_time - self.last_action_time >= self.action_cooldown):
            self.gesture_locked = False

        if detection_data is None:
            if current_time - self.last_hand_seen_time > self.config.no_hand_timeout_ms / 1000.0:
                self._ensure_mouse_released()
                self._exit_window_move_mode()
            return

        gestures = detection_data["gestures"]
        index_pos = detection_data["index_pos"]
        hand_center = detection_data["hand_center"]
        swipe_direction = detection_data.get("swipe_direction")
        pinch_distance = detection_data.get("pinch_distance")
        velocity = detection_data.get("velocity", 0.0)

        # Exclusivity: Only one gesture triggers at a time
        # Only allow one gesture to act per frame
        if gestures.get("full_hand"):
            # Alt+Tab
            if not self.alt_tab_active:
                pyautogui.keyDown("alt")
                self.alt_tab_active = True
                self.last_full_hand_pos = hand_center
                self.last_tab_press_time = current_time
                self._lock_gesture(current_time)
            else:
                cur_x = hand_center[0]
                prev_x = self.last_full_hand_pos[0] if self.last_full_hand_pos else cur_x
                move_threshold = 80
                if abs(cur_x - prev_x) > move_threshold and (current_time - self.last_tab_press_time > self.action_cooldown):
                    pyautogui.press("tab")
                    self.last_tab_press_time = current_time
                    self.last_full_hand_pos = (cur_x, hand_center[1])
            return
        else:
            if self.alt_tab_active:
                pyautogui.keyUp("alt")
                self.alt_tab_active = False
                self.last_full_hand_pos = None

        # Fist and one finger are strictly exclusive
        # Simplified fist-drag: Hold fist to start/continue drag anywhere; release to drop
        if gestures.get("fist") and not gestures.get("one_finger"):
            # Use raw coordinates to minimize perceived lag during drag
            raw_pos = detection_data.get("index_pos_raw", index_pos)
            cursor_x, cursor_y = raw_pos
            if not self.is_dragging:
                self._enter_drag_mode(index_pos)
            else:
                try:
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0)
                except (pyautogui.FailSafeException, TypeError, ValueError) as e:
                    logging.debug("moveTo during drag failed: %s", e)
            return
        else:
            # ensure we release if fist no longer detected
            self._exit_drag_mode()

        if gestures.get("one_finger") and not gestures.get("fist"):
            # Mouse move
            try:
                pyautogui.moveTo(index_pos[0], index_pos[1], duration=0)
            except (pyautogui.FailSafeException, TypeError, ValueError) as e:
                logging.debug("moveTo failed: %s", e)
            return

        if gestures.get("two_fingers_click"):
            # Mouse click
            if current_time - self.last_click_time > self.config.click_debounce_ms / 1000.0:
                try:
                    pyautogui.click()
                    self.last_click_time = current_time
                    self._lock_gesture(current_time)
                except (pyautogui.FailSafeException, TypeError, ValueError) as e:
                    logging.debug("click failed: %s", e)
            return

        if gestures.get("pinch"):
            # Zoom/scroll
            if self.pinch_smooth is None:
                self.pinch_smooth = pinch_distance
            else:
                self.pinch_smooth = (self.config.pinch_smooth_alpha * pinch_distance +
                                     (1 - self.config.pinch_smooth_alpha) * self.pinch_smooth)
            if self.last_pinch_distance is not None:
                delta = self.pinch_smooth - self.last_pinch_distance
                now = time.time()
                if abs(delta) > 0.5 and (now - self.last_action_time) > self.config.pinch_cooldown:
                    scroll_amount = int(delta * 2.0)
                    scroll_amount = max(-600, min(600, scroll_amount))
                    try:
                        pyautogui.keyDown("ctrl")
                        pyautogui.scroll(scroll_amount)
                        pyautogui.keyUp("ctrl")
                        self._lock_gesture(now)
                    except (pyautogui.FailSafeException, TypeError, ValueError) as e:
                        logging.debug("pinch-scroll failed: %s", e)
            self.last_pinch_distance = self.pinch_smooth
            return
        else:
            self.pinch_smooth = None
            self.last_pinch_distance = None

        # ---- Touch-like continuous swipe/scroll ----
        if gestures.get("four_fingers"):
            # Touchscreen-style: vertical swipes scroll continuously, horizontal swipes switch desktops repeatedly
            if swipe_direction:
                # For desktop switch, allow repeated switching if swipe continues
                if swipe_direction in ("swipe_left", "swipe_right"):
                    if (current_time - self.last_swipe_time > 0.25):
                        success = self.window_controller.switch_desktop(
                            swipe_direction)
                        if not success:
                            logging.debug(
                                "Desktop switch failed or unsupported.")
                        self.last_swipe_time = current_time
                        self.last_swipe_direction = swipe_direction
                        self._lock_gesture(current_time)
                # For scrolling, use the distance of the swipe for amount
                elif swipe_direction in ("swipe_up", "swipe_down"):
                    # Calculate scroll amount based on swipe distance
                    if len(self.swipe_positions) >= 2 and (current_time - self.last_swipe_time > 0.25):
                        start_x, start_y, _ = self.swipe_positions[0]
                        end_x, end_y, _ = self.swipe_positions[-1]
                        dy = end_y - start_y
                        scroll_amount = int(
                            dy * self.config.cam_height * self.config.scroll_sensitivity)
                        if swipe_direction == "swipe_up":
                            scroll_amount = abs(scroll_amount)
                        else:
                            scroll_amount = -abs(scroll_amount)
                        pyautogui.scroll(scroll_amount)
                        self.last_swipe_time = current_time
                        self.last_swipe_direction = swipe_direction
                        self._lock_gesture(current_time)
            return

        # Ignore three fingers (reserved)
        if gestures.get("three_fingers"):
            return

        # Only allow defined gestures to move/click/scroll. Ignore all other hand shapes.
        if not any([gestures.get("one_finger"), gestures.get("two_fingers_click"), gestures.get("pinch"), gestures.get("full_hand"), gestures.get("four_fingers"), gestures.get("fist")]):
            # No defined gesture detected: do nothing (natural movement filter)
            return

        # Shake detection for click (fallback)
        if velocity > self.config.shake_velocity_threshold and (current_time - self.last_click_time > self.config.click_debounce_ms / 1000.0):
            try:
                pyautogui.click()
                self.last_click_time = current_time
                self._lock_gesture(current_time)
            except Exception as e:
                logging.debug("shake-click failed: %s", e)

    def _ensure_mouse_released(self) -> None:
        if self.is_dragging:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
            self.is_dragging = False

    def _enter_drag_mode(self, start_pos: Tuple[int, int]) -> None:
        logging.debug("Entering drag mode")
        self.current_state = GestureState.DRAG
        self.is_dragging = True
        try:
            pyautogui.mouseDown()
            pyautogui.moveTo(start_pos[0], start_pos[1], duration=0)
        except Exception as e:
            logging.warning("enter_drag_mode error: %s", e)

    def _exit_drag_mode(self) -> None:
        if self.is_dragging:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
            self.is_dragging = False
        self.current_state = GestureState.IDLE

    def _enter_window_move_mode(self) -> None:
        logging.debug("Entering window move mode")
        self.target_window = self.window_controller.get_active_window()
        if self.target_window:
            self.current_state = GestureState.WINDOW_MOVE
        else:
            logging.debug("No active window for window move")

    def _exit_window_move_mode(self) -> None:
        if self.current_state == GestureState.WINDOW_MOVE:
            self.target_window = None
            self.current_state = GestureState.IDLE

    def _perform_click(self) -> None:
        try:
            pyautogui.click()
        except Exception as e:
            logging.warning("click error: %s", e)

    def run(self) -> None:
        if not self.initialize_camera():
            logging.error("Camera initialization failed - exiting")
            return
        logging.info("Starting gesture control - press 'q' to quit")
        self.running = True
        try:
            frame_time_target = 1.0 / self.config.fps_limit
            while self.running:
                loop_start = time.time()
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logging.error("Failed to read camera frame, retrying...")
                    time.sleep(0.1)
                    continue
                try:
                    detection = self.process_frame(frame)
                except Exception as e:
                    logging.error("process_frame error: %s", e)
                    detection = None
                try:
                    self.handle_gesture_logic(detection)
                except Exception as e:
                    logging.error("handle_gesture_logic error: %s", e)
                try:
                    self._draw_debug_info(frame, detection)
                except Exception as e:
                    logging.debug("draw_debug_info error: %s", e)
                try:
                    cv2.imshow("Gesture Control", frame)
                except Exception as e:
                    logging.debug("imshow error: %s", e)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                # During drag, run at higher refresh and avoid artificial delay
                elapsed = time.time() - loop_start
                dynamic_target = 1.0 / 60.0 if self.is_dragging else frame_time_target
                sleep_time = max(0, dynamic_target - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        except Exception as e:
            logging.error("Runtime error: %s", e)
        finally:
            self.cleanup()

    def _draw_debug_info(self, frame: Any, detection_data: Optional[Dict[str, Any]]) -> None:
        state_color = (
            0, 255, 0) if self.current_state == GestureState.IDLE else (0, 0, 255)
        cv2.putText(frame, f"State: {self.current_state.value}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        if detection_data:
            try:
                self.mp_drawing.draw_landmarks(
                    frame, detection_data["hand_landmarks"], self.mp_hands.HAND_CONNECTIONS)
            except Exception:
                pass
            vel = detection_data.get("velocity", 0.0)
            vel_color = (0, 255, 255) if vel > self.config.shake_velocity_threshold else (
                255, 255, 255)
            cv2.putText(frame, f"Vel: {int(vel)} px/s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, vel_color, 2)
            gestures = detection_data.get("gestures", {})
            y = 90
            for g, active in gestures.items():
                color = (0, 255, 0) if active else (128, 128, 128)
                cv2.putText(frame, f"{g}: {active}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 22
            swipe = detection_data.get("swipe_direction")
            if swipe:
                cv2.putText(frame, f"Swipe: {swipe}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def cleanup(self) -> None:
        logging.info("Cleaning up...")
        self.running = False
        self._ensure_mouse_released()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        logging.info("Cleanup complete")


# ----------------------------
# Entrypoint
# ----------------------------
def main():
    cfg = Config.load()
    controller = GestureController(cfg)
    controller.run()


if __name__ == "__main__":
    main()
