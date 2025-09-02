#!/usr/bin/env python3
"""
Improved NvBot3 Feedback Bridge with Connection Pooling
======================================================

Thread-safe integration bridge with proper database connection management,
operation queuing, and improved error handling for high-concurrency scenarios.
"""

import sys
import os
import threading
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dashboard_path = os.path.join(current_dir, '..', 'web_dashboard')
sys.path.append(web_dashboard_path)

project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Global variables with thread safety
TRACKER = None
TRACKING_ENABLED = False
_tracker_lock = threading.RLock()
_initialization_lock = threading.Lock()

def init_tracker():
    """Initialize the improved tracker with thread safety"""
    global TRACKER, TRACKING_ENABLED
    
    with _initialization_lock:
        if TRACKER is not None:
            logger.debug("TRACKER already initialized")
            return TRACKER
        
        try:
            logger.debug("Initializing ImprovedSignalTracker...")
            
            # Import the improved tracker
            try:
                from web_dashboard.database.signal_tracker_improved import ImprovedSignalTracker
                TRACKER = ImprovedSignalTracker()
            except ImportError:
                # Fallback to original tracker if improved version not available
                logger.warning("ImprovedSignalTracker not available, using original")
                from web_dashboard.database.signal_tracker import SignalTracker
                TRACKER = SignalTracker()
            
            if TRACKER:
                TRACKING_ENABLED = True
                logger.info("SUCCESS: NvBot3 tracking system initialized with improved database handling")
            else:
                TRACKING_ENABLED = False
                logger.warning("WARNING: TRACKER initialization failed")
            
            return TRACKER
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            TRACKING_ENABLED = False
            return None
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            TRACKING_ENABLED = False
            return None

def ensure_tracker_initialized():
    """Ensure tracker is initialized with thread safety"""
    global TRACKER, TRACKING_ENABLED
    
    if TRACKER is None or not TRACKING_ENABLED:
        with _tracker_lock:
            if TRACKER is None:
                init_tracker()

def track_signal(symbol: str, prediction_data: Dict, current_price: float) -> Optional[str]:
    """
    Thread-safe signal tracking function with improved error handling
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTCUSDT')
        prediction_data (dict): Prediction data with 'type', 'predicted_change', 'confidence'
        current_price (float): Current asset price
    
    Returns:
        str: Signal ID if saved successfully, None if error
    """
    
    # Ensure tracker is initialized
    ensure_tracker_initialized()
    if not TRACKING_ENABLED:
        logger.warning("Tracking not enabled, signal not saved")
        return None
    
    # Input validation
    if not symbol or not isinstance(symbol, str):
        logger.error("Invalid symbol provided")
        return None
    
    if not prediction_data or not isinstance(prediction_data, dict):
        logger.error("Invalid prediction_data provided")
        return None
    
    if not isinstance(current_price, (int, float)) or current_price <= 0:
        logger.error("Invalid current_price provided")
        return None
    
    try:
        with _tracker_lock:
            # Prepare signal data with validation
            signal_data = {
                'type': prediction_data.get('type', 'unknown'),
                'predicted_change': float(prediction_data.get('predicted_change', 0)),
                'confidence': float(prediction_data.get('confidence', 0)),
                'entry_price': float(current_price),
                'timestamp': datetime.now().isoformat(),
                'expected_timeframe': prediction_data.get('expected_timeframe', 60)
            }
            
            # Validate signal data
            if signal_data['confidence'] < 0 or signal_data['confidence'] > 1:
                logger.warning(f"Invalid confidence score: {signal_data['confidence']}")
                signal_data['confidence'] = max(0, min(1, signal_data['confidence']))
            
            # Save signal using tracker
            if TRACKER is not None:
                signal_id = TRACKER.save_new_signal(symbol, signal_data)
                
                if signal_id:
                    logger.info(f"SUCCESS: Signal saved: {symbol} - {signal_data['type']} - "
                              f"Confidence: {signal_data['confidence']:.3f} - ID: {signal_id}")
                    return signal_id
                else:
                    logger.warning(f"WARNING: Failed to save signal for {symbol}")
                    return None
            else:
                logger.error("TRACKER not initialized")
                return None
                
    except ValueError as e:
        logger.error(f"Data validation error for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error saving signal for {symbol}: {e}")
        return None

def update_price(symbol: str, price: float) -> bool:
    """
    Thread-safe price update function
    
    Args:
        symbol (str): Asset symbol
        price (float): Current price
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    ensure_tracker_initialized()
    if not TRACKING_ENABLED:
        return False
    
    # Input validation
    if not symbol or not isinstance(symbol, str):
        logger.error("Invalid symbol for price update")
        return False
    
    if not isinstance(price, (int, float)) or price <= 0:
        logger.error("Invalid price for update")
        return False
    
    try:
        with _tracker_lock:
            if TRACKER is not None:
                TRACKER.update_price_tracking(symbol, float(price))
                logger.debug(f"Price updated for {symbol}: {price}")
                return True
            else:
                logger.error("TRACKER not initialized for price update")
                return False
                
    except Exception as e:
        logger.error(f"Error updating price for {symbol}: {e}")
        return False

def get_tracking_stats() -> Dict[str, Any]:
    """
    Get tracking statistics with improved error handling
    
    Returns:
        dict: Statistics or error information
    """
    
    ensure_tracker_initialized()
    if not TRACKING_ENABLED:
        return {"error": "Tracking system not available", "total_signals": 0}
    
    try:
        with _tracker_lock:
            if TRACKER is not None:
                stats = TRACKER.get_performance_stats(days=30)
                return stats
            else:
                logger.warning("TRACKER not initialized for stats")
                return {"error": "TRACKER not initialized", "total_signals": 0}
                
    except Exception as e:
        logger.error(f"Error getting tracking stats: {e}")
        return {"error": str(e), "total_signals": 0}

def get_active_signals() -> List[Dict]:
    """
    Get active signals with thread safety
    
    Returns:
        list: List of active signals
    """
    
    ensure_tracker_initialized()
    if not TRACKING_ENABLED:
        return []
    
    try:
        with _tracker_lock:
            if TRACKER is not None:
                signals = TRACKER.get_active_signals()
                return signals if signals else []
            else:
                logger.warning("TRACKER not initialized for active signals")
                return []
                
    except Exception as e:
        logger.error(f"Error getting active signals: {e}")
        return []

def save_user_feedback(signal_id: str, feedback_data: Dict) -> bool:
    """
    Save user feedback with validation
    
    Args:
        signal_id (str): Signal identifier
        feedback_data (dict): Feedback information
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    ensure_tracker_initialized()
    if not TRACKING_ENABLED:
        return False
    
    # Input validation
    if not signal_id or not isinstance(signal_id, str):
        logger.error("Invalid signal_id for feedback")
        return False
    
    if not feedback_data or not isinstance(feedback_data, dict):
        logger.error("Invalid feedback_data provided")
        return False
    
    try:
        with _tracker_lock:
            if TRACKER is not None:
                TRACKER.save_user_feedback(signal_id, feedback_data)
                logger.info(f"Feedback saved for signal: {signal_id}")
                return True
            else:
                logger.error("TRACKER not initialized for feedback")
                return False
                
    except Exception as e:
        logger.error(f"Error saving feedback for {signal_id}: {e}")
        return False

def cleanup_old_signals(days: int = 30) -> int:
    """
    Clean up old signals to maintain performance
    
    Args:
        days (int): Number of days to keep
    
    Returns:
        int: Number of signals cleaned up
    """
    
    ensure_tracker_initialized()
    if not TRACKING_ENABLED:
        return 0
    
    try:
        with _tracker_lock:
            if TRACKER is not None and hasattr(TRACKER, 'cleanup_old_signals'):
                count = TRACKER.cleanup_old_signals(days)
                logger.info(f"Cleaned up {count} old signals")
                return count
            else:
                logger.warning("Cleanup not available in current tracker")
                return 0
                
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return 0

def get_system_health() -> Dict[str, Any]:
    """
    Get system health information
    
    Returns:
        dict: System health metrics
    """
    
    health = {
        'tracker_initialized': TRACKER is not None,
        'tracking_enabled': TRACKING_ENABLED,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        if TRACKING_ENABLED and TRACKER is not None:
            # Try to get basic stats to test database connectivity
            stats = get_tracking_stats()
            health['database_accessible'] = 'error' not in stats
            health['total_signals'] = stats.get('total_signals', 0)
        else:
            health['database_accessible'] = False
            health['total_signals'] = 0
            
    except Exception as e:
        health['database_accessible'] = False
        health['error'] = str(e)
    
    return health

def shutdown_tracking():
    """Properly shutdown the tracking system"""
    global TRACKER, TRACKING_ENABLED
    
    logger.info("Shutting down tracking system...")
    
    with _tracker_lock:
        if TRACKER is not None:
            try:
                if hasattr(TRACKER, 'close'):
                    TRACKER.close()
                TRACKER = None
                TRACKING_ENABLED = False
                logger.info("Tracking system shutdown complete")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

# Compatibility aliases
manual_price_update = update_price

# Improved integration example
def example_integration():
    """
    Enhanced integration example with error handling
    """
    
    print("Enhanced NvBot3 Integration Example")
    print("=" * 50)
    
    # Check system health first
    health = get_system_health()
    print(f"System Health: {health}")
    
    if not health['tracking_enabled']:
        print("ERROR: Tracking system not available")
        return
    
    # Example signal data
    symbol = "BTCUSDT"
    current_price = 67250.0
    
    prediction = {
        'type': 'momentum_high',
        'predicted_change': 5.2,
        'confidence': 0.85,
        'expected_timeframe': 60
    }
    
    # Track the signal
    signal_id = track_signal(symbol, prediction, current_price)
    
    if signal_id:
        print(f"SUCCESS: Signal tracked with ID: {signal_id}")
        
        # Simulate price updates
        time.sleep(1)  # Wait a bit
        
        new_price = 68500.0
        if update_price(symbol, new_price):
            print(f"SUCCESS: Price updated to {new_price}")
        
        # Get statistics
        stats = get_tracking_stats()
        print(f"Statistics: Total signals: {stats.get('total_signals', 0)}")
        
        # Get active signals
        active = get_active_signals()
        print(f"Active signals: {len(active)}")
        
    else:
        print("ERROR: Failed to track signal")

if __name__ == "__main__":
    try:
        print("Testing improved NvBot3 feedback bridge...")
        example_integration()
        
        # Test cleanup
        cleaned = cleanup_old_signals(30)
        print(f"Cleaned up {cleaned} old signals")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        shutdown_tracking()