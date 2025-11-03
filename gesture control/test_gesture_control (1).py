#!/usr/bin/env python3
"""
Unit tests for gesture control system.

Run with: python -m pytest test_gesture_control.py -v
"""

import unittest
import math
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import the main modules (assuming they're in the same directory)
from gesture_control_main import (
    Config, CoordinateMapper, MotionAnalyzer, GestureDetector,
    WindowController, GestureState
)


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        self.config = Config()
    
    def test_default_values(self):
        """Test default configuration values."""
        self.assertEqual(self.config.cam_width, 640)
        self.assertEqual(self.config.cam_height, 480)
        self.assertEqual(self.config.fps_limit, 30)
        self.assertGreater(self.config.open_threshold, 0)
        self.assertGreater(self.config.fist_threshold, 0)
    
    def test_threshold_relationships(self):
        """Test that thresholds have logical relationships."""
        self.assertLess(self.config.fist_threshold, self.config.open_threshold)
        self.assertGreater(self.config.shake_velocity_threshold, 0)


class TestCoordinateMapper(unittest.TestCase):
    """Test coordinate mapping and smoothing."""
    
    def setUp(self):
        # Mock pyautogui.size() to return consistent screen size
        self.screen_size_patcher = patch('pyautogui.size', return_value=(1920, 1080))
        self.screen_size_patcher.start()
        self.mapper = CoordinateMapper(smooth_alpha=0.5)
    
    def tearDown(self):
        self.screen_size_patcher.stop()
    
    def test_normalize_to_screen(self):
        """Test coordinate normalization."""
        # Test corners and center
        x, y = self.mapper.normalize_to_screen(0.0, 0.0)
        self.assertEqual((x, y), (0, 0))
        
        x, y = self.mapper.normalize_to_screen(1.0, 1.0)
        self.assertEqual((x, y), (1919, 1079))  # Max screen coordinates
        
        x, y = self.mapper.normalize_to_screen(0.5, 0.5)
        self.assertEqual((x, y), (960, 540))  # Center
    
    def test_coordinate_bounds(self):
        """Test coordinate boundary conditions."""
        # Test out-of-bounds coordinates
        x, y = self.mapper.normalize_to_screen(-0.1, -0.1)
        self.assertEqual((x, y), (0, 0))
        
        x, y = self.mapper.normalize_to_screen(1.1, 1.1)
        self.assertEqual((x, y), (1919, 1079))
    
    def test_smoothing_initialization(self):
        """Test smoothing filter initialization."""
        # First call should not smooth
        x1, y1 = self.mapper.smooth_position(100, 100)
        self.assertEqual((x1, y1), (100, 100))
        
        # Second call should apply smoothing
        x2, y2 = self.mapper.smooth_position(200, 200)
        self.assertEqual((x2, y2), (150, 150))  # With alpha=0.5
    
    def test_smoothing_reset(self):
        """Test smoothing filter reset."""
        self.mapper.smooth_position(100, 100)
        self.mapper.reset_smoothing()
        
        x, y = self.mapper.smooth_position(200, 200)
        self.assertEqual((x, y), (200, 200))  # Should not smooth after reset


class TestMotionAnalyzer(unittest.TestCase):
    """Test motion analysis and velocity calculation."""
    
    def setUp(self):
        self.analyzer = MotionAnalyzer(window_size_ms=100)
    
    def test_empty_history_velocity(self):
        """Test velocity calculation with no history."""
        self.assertEqual(self.analyzer.get_velocity(), 0.0)
    
    def test_single_point_velocity(self):
        """Test velocity calculation with single point."""
        self.analyzer.add_position(100, 100)
        self.assertEqual(self.analyzer.get_velocity(), 0.0)
    
    def test_velocity_calculation(self):
        """Test basic velocity calculation."""
        # Mock time to control timestamps
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 0.1]  # 100ms apart
            
            self.analyzer.add_position(0, 0)
            self.analyzer.add_position(100, 0)  # 100 pixels in 100ms = 1000 px/s
            
            velocity = self.analyzer.get_velocity()
            self.assertAlmostEqual(velocity, 1000.0, places=1)
    
    def test_diagonal_velocity(self):
        """Test velocity calculation for diagonal movement."""
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 0.1]
            
            self.analyzer.add_position(0, 0)
            self.analyzer.add_position(30, 40)  # 3-4-5 triangle, 50 pixels
            
            velocity = self.analyzer.get_velocity()
            self.assertAlmostEqual(velocity, 500.0, places=1)
    
    def test_history_pruning(self):
        """Test that old position samples are pruned."""
        with patch('time.time') as mock_time:
            # Add positions with increasing timestamps
            mock_time.side_effect = [0.0, 0.05, 0.15]  # 0ms, 50ms, 150ms
            
            self.analyzer.add_position(0, 0)    # t=0, should be pruned
            self.analyzer.add_position(50, 0)   # t=50ms, should remain
            self.analyzer.add_position(100, 0)  # t=150ms, current
            
            # History should only contain last 2 points (within 100ms window)
            self.assertEqual(len(self.analyzer.position_history), 2)
    
    def test_clear_history(self):
        """Test clearing position history."""
        self.analyzer.add_position(100, 100)
        self.analyzer.add_position(200, 200)
        self.analyzer.clear_history()
        
        self.assertEqual(len(self.analyzer.position_history), 0)
        self.assertEqual(self.analyzer.get_velocity(), 0.0)


class MockLandmark:
    """Mock MediaPipe landmark for testing."""
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z


class MockHandLandmarks:
    """Mock MediaPipe hand landmarks for testing."""
    def __init__(self, landmarks_dict):
        self.landmark = [MockLandmark(0, 0)] * 21  # Initialize 21 landmarks
        for idx, (x, y) in landmarks_dict.items():
            self.landmark[idx] = MockLandmark(x, y)


class TestGestureDetector(unittest.TestCase):
    """Test gesture detection logic."""
    
    def setUp(self):
        self.config = Config()
        self.detector = GestureDetector(self.config)
    
    def create_open_hand(self):
        """Create mock landmarks for open hand gesture."""
        return MockHandLandmarks({
            # Fingertips far from MCP joints (open hand)
            8: (0.8, 0.2),   # Index tip
            12: (0.9, 0.1),  # Middle tip
            16: (0.85, 0.15), # Ring tip
            20: (0.75, 0.25), # Pinky tip
            # MCP joints
            5: (0.5, 0.5),   # Index MCP
            9: (0.6, 0.5),   # Middle MCP
            13: (0.55, 0.55), # Ring MCP
            17: (0.45, 0.55)  # Pinky MCP
        })
    
    def create_fist(self):
        """Create mock landmarks for fist gesture."""
        return MockHandLandmarks({
            # Fingertips close to MCP joints (closed fist)
            8: (0.52, 0.48),  # Index tip
            12: (0.58, 0.48), # Middle tip
            16: (0.57, 0.52), # Ring tip
            20: (0.47, 0.52), # Pinky tip
            # MCP joints
            5: (0.5, 0.5),    # Index MCP
            9: (0.6, 0.5),    # Middle MCP
            13: (0.55, 0.55), # Ring MCP
            17: (0.45, 0.55)  # Pinky MCP
        })
    
    def create_pointing_hand(self):
        """Create mock landmarks for index pointing gesture."""
        return MockHandLandmarks({
            # Index extended, others curled
            8: (0.7, 0.2),   # Index tip (extended)
            6: (0.6, 0.4),   # Index PIP
            12: (0.58, 0.48), # Middle tip (curled)
            16: (0.57, 0.52), # Ring tip (curled)
            20: (0.47, 0.52), # Pinky tip (curled)
            # MCP joints
            5: (0.5, 0.5),    # Index MCP
            9: (0.6, 0.5),    # Middle MCP
            13: (0.55, 0.55), # Ring MCP
            17: (0.45, 0.55)  # Pinky MCP
        })
    
    def test_open_hand_detection(self):
        """Test open hand gesture detection."""
        open_hand = self.create_open_hand()
        self.assertTrue(self.detector.is_open_hand(open_hand))
        
        fist = self.create_fist()
        self.assertFalse(self.detector.is_open_hand(fist))
    
    def test_fist_detection(self):
        """Test fist gesture detection."""
        fist = self.create_fist()
        self.assertTrue(self.detector.is_fist(fist))
        
        open_hand = self.create_open_hand()
        self.assertFalse(self.detector.is_fist(open_hand))
    
    def test_pointing_detection(self):
        """Test index pointing gesture detection."""
        pointing_hand = self.create_pointing_hand()
        self.assertTrue(self.detector.is_index_pointing(pointing_hand))
        
        open_hand = self.create_open_hand()
        self.assertFalse(self.detector.is_index_pointing(open_hand))
        
        fist = self.create_fist()
        self.assertFalse(self.detector.is_index_pointing(fist))
    
    def test_hand_center_calculation(self):
        """Test hand center calculation."""
        landmarks_dict = {i: (0.5, 0.5) for i in range(21)}  # All at center
        hand = MockHandLandmarks(landmarks_dict)
        
        center_x, center_y = self.detector.get_hand_center(hand)
        self.assertAlmostEqual(center_x, 0.5, places=2)
        self.assertAlmostEqual(center_y, 0.5, places=2)


class TestWindowController(unittest.TestCase):
    """Test window manipulation functionality."""
    
    def setUp(self):
        self.controller = WindowController()
    
    @patch('pygetwindow.getActiveWindow')
    def test_get_active_window(self, mock_get_active):
        """Test getting active window."""
        # Mock active window
        mock_window = Mock()
        mock_window.isActive = True
        mock_get_active.return_value = mock_window
        
        result = self.controller.get_active_window()
        self.assertEqual(result, mock_window)
        
        # Test no active window
        mock_get_active.return_value = None
        result = self.controller.get_active_window()
        self.assertIsNone(result)
    
    @patch('pyautogui.size', return_value=(1920, 1080))
    def test_move_window(self, mock_size):
        """Test window moving functionality."""
        # Mock window object
        mock_window = Mock()
        mock_window.width = 800
        mock_window.height = 600
        
        # Test successful move
        success = self.controller.move_window(mock_window, 960, 540)
        self.assertTrue(success)
        mock_window.moveTo.assert_called_once_with(560, 240)  # Centered position
        
        # Test move with window exception
        mock_window.moveTo.side_effect = Exception("Move failed")
        success = self.controller.move_window(mock_window, 960, 540)
        self.assertFalse(success)
    
    def test_bring_to_front(self):
        """Test bringing window to front."""
        mock_window = Mock()
        
        # Test successful activation
        success = self.controller.bring_to_front(mock_window)
        self.assertTrue(success)
        mock_window.activate.assert_called_once()
        
        # Test activation failure
        mock_window.activate.side_effect = Exception("Activation failed")
        success = self.controller.bring_to_front(mock_window)
        self.assertFalse(success)


class TestGestureIntegration(unittest.TestCase):
    """Integration tests for gesture system."""
    
    def setUp(self):
        self.config = Config()
        self.config.shake_velocity_threshold = 1000
        self.config.click_debounce_ms = 100
        
    @patch('pyautogui.size', return_value=(1920, 1080))
    def test_coordinate_mapping_integration(self, mock_size):
        """Test integration of coordinate mapping components."""
        mapper = CoordinateMapper(smooth_alpha=0.5)
        analyzer = MotionAnalyzer(window_size_ms=200)
        
        # Simulate hand movement sequence
        positions = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
        
        for norm_x, norm_y in positions:
            screen_x, screen_y = mapper.normalize_to_screen(norm_x, norm_y)
            smooth_x, smooth_y = mapper.smooth_position(screen_x, screen_y)
            analyzer.add_position(smooth_x, smooth_y)
        
        # Verify final position is reasonable
        final_velocity = analyzer.get_velocity()
        self.assertGreater(final_velocity, 0)
    
    def test_gesture_state_transitions(self):
        """Test logical gesture state transitions."""
        # Test valid state transitions
        valid_transitions = {
            GestureState.IDLE: [GestureState.DRAG, GestureState.WINDOW_MOVE],
            GestureState.DRAG: [GestureState.IDLE],
            GestureState.WINDOW_MOVE: [GestureState.IDLE]
        }
        
        for state, allowed in valid_transitions.items():
            self.assertIsInstance(state, GestureState)
            for target in allowed:
                self.assertIsInstance(target, GestureState)


class TestThresholds(unittest.TestCase):
    """Test threshold configurations and edge cases."""
    
    def setUp(self):
        self.config = Config()
        self.detector = GestureDetector(self.config)
    
    def test_threshold_edge_cases(self):
        """Test behavior at threshold boundaries."""
        # Create hand landmarks right at threshold boundaries
        edge_open = MockHandLandmarks({
            8: (0.5 + self.config.open_threshold, 0.5),
            12: (0.5 + self.config.open_threshold, 0.5),
            16: (0.5 + self.config.open_threshold, 0.5),
            20: (0.5 + self.config.open_threshold, 0.5),
            5: (0.5, 0.5), 9: (0.5, 0.5), 13: (0.5, 0.5), 17: (0.5, 0.5)
        })
        
        edge_fist = MockHandLandmarks({
            8: (0.5 + self.config.fist_threshold, 0.5),
            12: (0.5 + self.config.fist_threshold, 0.5),
            16: (0.5 + self.config.fist_threshold, 0.5),
            20: (0.5 + self.config.fist_threshold, 0.5),
            5: (0.5, 0.5), 9: (0.5, 0.5), 13: (0.5, 0.5), 17: (0.5, 0.5)
        })
        
        # Test threshold boundaries
        self.assertTrue(self.detector.is_open_hand(edge_open))
        self.assertFalse(self.detector.is_fist(edge_fist))


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance and stability metrics."""
    
    def test_smoothing_stability(self):
        """Test smoothing reduces jitter."""
        mapper = CoordinateMapper(smooth_alpha=0.2)
        
        # Simulate noisy input
        noisy_positions = [(100 + i % 3, 100 + i % 2) for i in range(10)]
        smoothed_positions = []
        
        for x, y in noisy_positions:
            smooth_x, smooth_y = mapper.smooth_position(x, y)
            smoothed_positions.append((smooth_x, smooth_y))
        
        # Calculate variance of smoothed vs original
        orig_var_x = np.var([pos[0] for pos in noisy_positions])
        smooth_var_x = np.var([pos[0] for pos in smoothed_positions])
        
        # Smoothed positions should have lower variance
        self.assertLess(smooth_var_x, orig_var_x)
    
    def test_velocity_calculation_precision(self):
        """Test velocity calculation precision and stability."""
        analyzer = MotionAnalyzer(window_size_ms=100)
        
        # Test with known velocity
        with patch('time.time') as mock_time:
            timestamps = [i * 0.01 for i in range(10)]  # 10ms intervals
            mock_time.side_effect = timestamps
            
            # Move 10 pixels every 10ms = 1000 px/s
            for i, t in enumerate(timestamps):
                analyzer.add_position(i * 10, 0)
            
            velocity = analyzer.get_velocity()
            # Should be close to 1000 px/s
            self.assertAlmostEqual(velocity, 1000.0, delta=50.0)


def create_test_suite():
    """Create comprehensive test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestConfig,
        TestCoordinateMapper,
        TestMotionAnalyzer,
        TestGestureDetector,
        TestWindowController,
        TestGestureIntegration,
        TestThresholds,
        TestPerformanceMetrics
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_performance_benchmarks():
    """Run performance benchmarks for key components."""
    print("\n=== Performance Benchmarks ===")
    
    # Benchmark coordinate mapping
    mapper = CoordinateMapper()
    start_time = time.time()
    
    for _ in range(10000):
        x, y = mapper.normalize_to_screen(0.5, 0.5)
        mapper.smooth_position(x, y)
    
    coord_time = time.time() - start_time
    print(f"Coordinate mapping: {coord_time:.4f}s for 10k operations")
    
    # Benchmark gesture detection
    config = Config()
    detector = GestureDetector(config)
    hand = MockHandLandmarks({i: (0.5, 0.5) for i in range(21)})
    
    start_time = time.time()
    
    for _ in range(1000):
        detector.is_open_hand(hand)
        detector.is_fist(hand)
        detector.is_index_pointing(hand)
    
    gesture_time = time.time() - start_time
    print(f"Gesture detection: {gesture_time:.4f}s for 1k operations")
    
    # Benchmark motion analysis
    analyzer = MotionAnalyzer()
    start_time = time.time()
    
    for i in range(1000):
        analyzer.add_position(i, i)
        analyzer.get_velocity()
    
    motion_time = time.time() - start_time
    print(f"Motion analysis: {motion_time:.4f}s for 1k operations")
    
    print(f"\nTarget: <50ms per frame at 30fps")
    total_per_frame = (coord_time/10000 + gesture_time/1000 + motion_time/1000) * 1000
    print(f"Estimated processing time per frame: {total_per_frame:.2f}ms")


if __name__ == "__main__":
    # Run unit tests
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance benchmarks
    if result.wasSuccessful():
        run_performance_benchmarks()
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
    