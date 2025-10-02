#!/usr/bin/env python3
"""
Test script for the multiprocessing timeout mechanism in kolam_performance_optimizer.py
"""

import multiprocessing
import time
from kolam_performance_optimizer import PerformanceOptimizer
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))


def test_timeout_mechanism():
    """Test the multiprocessing timeout mechanism."""
    print("Testing multiprocessing timeout mechanism...")

    optimizer = PerformanceOptimizer()

    # Test 1: Function that completes normally
    def quick_function():
        time.sleep(0.1)
        return "Success!"

    try:
        result = optimizer._execute_with_timeout(quick_function, (), 1.0)
        print(f"✓ Quick function test passed: {result}")
    except Exception as e:
        print(f"✗ Quick function test failed: {e}")

    # Test 2: Function that times out
    def slow_function():
        time.sleep(2.0)  # Sleep longer than timeout
        return "Should not reach here"

    try:
        result = optimizer._execute_with_timeout(slow_function, (), 0.5)
        print(f"✗ Slow function test failed - should have timed out: {result}")
    except TimeoutError as e:
        print(f"✓ Slow function test passed - correctly timed out: {e}")
    except Exception as e:
        print(f"✗ Slow function test failed with unexpected error: {e}")

    # Test 3: Function that raises an exception
    def error_function():
        raise ValueError("Test exception")

    try:
        result = optimizer._execute_with_timeout(error_function, (), 1.0)
        print(
            f"✗ Error function test failed - should have raised exception: {result}")
    except ValueError as e:
        print(
            f"✓ Error function test passed - correctly raised exception: {e}")
    except Exception as e:
        print(f"✗ Error function test failed with unexpected error: {e}")

    print("Multiprocessing timeout mechanism test completed!")


if __name__ == "__main__":
    # Ensure multiprocessing works on Windows
    multiprocessing.freeze_support()
    test_timeout_mechanism()
