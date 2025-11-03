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
    
    