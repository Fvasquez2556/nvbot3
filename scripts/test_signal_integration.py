#!/usr/bin/env python3
"""
🧪 Test Signal Integration - NvBot3
==================================

Test script to verify that the signal generator correctly integrates
with the web dashboard tracking system.

Tests:
1. Integration bridge functionality
2. Signal generation and tracking
3. Dashboard data flow
4. Model loading and prediction
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
import time
import json
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import components to test
try:
    from integration.nvbot3_feedback_bridge import track_signal, update_price, init_tracker
    bridge_available = True
except ImportError as e:
    print(f"❌ Could not import feedback bridge: {e}")
    bridge_available = False

try:
    from scripts.signal_generator import NvBot3SignalGenerator
    generator_available = True
except ImportError as e:
    print(f"❌ Could not import signal generator: {e}")
    generator_available = False

try:
    from web_dashboard.database.signal_tracker import SignalTracker
    tracker_available = True
except ImportError as e:
    print(f"❌ Could not import signal tracker: {e}")
    tracker_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class SignalIntegrationTester:
    """Comprehensive tester for signal integration"""
    
    def __init__(self):
        self.test_results = {}
        self.test_count = 0
        self.passed_tests = 0
        
        logger.info("🧪 Signal Integration Tester initialized")
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results"""
        self.test_count += 1
        logger.info(f"🔬 Test {self.test_count}: {test_name}")
        
        try:
            result = test_func()
            if result:
                self.passed_tests += 1
                self.test_results[test_name] = {"status": "PASSED", "error": None}
                logger.info(f"   ✅ PASSED")
                return True
            else:
                self.test_results[test_name] = {"status": "FAILED", "error": "Test returned False"}
                logger.error(f"   ❌ FAILED")
                return False
                
        except Exception as e:
            self.test_results[test_name] = {"status": "ERROR", "error": str(e)}
            logger.error(f"   💥 ERROR: {e}")
            return False
    
    def test_imports(self) -> bool:
        """Test that all required modules can be imported"""
        logger.info("   Checking component imports...")
        
        if not bridge_available:
            logger.error("   ❌ Integration bridge not available")
            return False
        
        if not generator_available:
            logger.error("   ❌ Signal generator not available")
            return False
        
        if not tracker_available:
            logger.error("   ❌ Signal tracker not available")
            return False
        
        logger.info("   ✅ All components imported successfully")
        return True
    
    def test_bridge_initialization(self) -> bool:
        """Test that the integration bridge can be initialized"""
        logger.info("   Testing bridge initialization...")
        
        try:
            init_tracker()
            logger.info("   ✅ Bridge initialized successfully")
            return True
        except Exception as e:
            logger.error(f"   ❌ Bridge initialization failed: {e}")
            return False
    
    def test_signal_tracking(self) -> bool:
        """Test signal tracking functionality"""
        logger.info("   Testing signal tracking...")
        
        # Create test prediction data
        test_prediction = {
            'type': 'momentum',
            'prediction': 1,
            'confidence': 0.85,
            'predicted_change': 5.2,
            'timestamp': '2025-08-31T12:00:00',
            'model_description': '🔥 Test Momentum Model'
        }
        
        test_symbol = 'BTCUSDT'
        test_price = 45000.0
        
        try:
            # Test signal tracking
            signal_id = track_signal(test_symbol, test_prediction, test_price)
            
            if signal_id:
                logger.info(f"   ✅ Signal tracked successfully (ID: {signal_id})")
                return True
            else:
                logger.error("   ❌ Signal tracking returned None")
                return False
                
        except Exception as e:
            logger.error(f"   ❌ Signal tracking failed: {e}")
            return False
    
    def test_price_update(self) -> bool:
        """Test price update functionality"""
        logger.info("   Testing price updates...")
        
        test_symbol = 'BTCUSDT'
        test_price = 45500.0
        
        try:
            update_price(test_symbol, test_price)
            logger.info("   ✅ Price updated successfully")
            return True
        except Exception as e:
            logger.error(f"   ❌ Price update failed: {e}")
            return False
    
    def test_signal_generator_init(self) -> bool:
        """Test signal generator initialization"""
        logger.info("   Testing signal generator initialization...")
        
        try:
            generator = NvBot3SignalGenerator()
            logger.info("   ✅ Signal generator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"   ❌ Signal generator initialization failed: {e}")
            return False
    
    def test_feature_calculation(self) -> bool:
        """Test feature calculation with dummy data"""
        logger.info("   Testing feature calculation...")
        
        try:
            # Create dummy market data
            dates = pd.date_range('2025-08-01', periods=100, freq='5T')
            dummy_data = pd.DataFrame({
                'open': np.random.uniform(40000, 45000, 100),
                'high': np.random.uniform(45000, 46000, 100),
                'low': np.random.uniform(39000, 40000, 100),
                'close': np.random.uniform(40000, 45000, 100),
                'volume': np.random.uniform(100, 1000, 100),
                'symbol': 'BTCUSDT',
                'timeframe': '5m'
            }, index=dates)
            
            generator = NvBot3SignalGenerator()
            features_df = generator.calculate_features(dummy_data)
            
            if len(features_df.columns) > len(dummy_data.columns):
                logger.info(f"   ✅ Features calculated: {len(features_df.columns)} features")
                return True
            else:
                logger.error("   ❌ No additional features were calculated")
                return False
                
        except Exception as e:
            logger.error(f"   ❌ Feature calculation failed: {e}")
            return False
    
    def test_database_connection(self) -> bool:
        """Test database connection and basic operations"""
        logger.info("   Testing database connection...")
        
        try:
            tracker = SignalTracker()
            
            # Test getting active signals
            active_signals = tracker.get_active_signals()
            logger.info(f"   ✅ Database connected. Active signals: {len(active_signals)}")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Database connection failed: {e}")
            return False
    
    def test_end_to_end_signal_flow(self) -> bool:
        """Test complete signal flow from generation to tracking"""
        logger.info("   Testing end-to-end signal flow...")
        
        try:
            # Create a mock prediction that would trigger tracking
            test_prediction = {
                'type': 'rebound',
                'prediction': 1,
                'confidence': 0.78,  # Above threshold
                'predicted_change': 2.5,
                'timestamp': '2025-08-31T12:00:00',
                'model_description': '⚡ Test Rebound Model'
            }
            
            test_symbol = 'ETHUSDT'
            test_price = 2800.0
            
            # Step 1: Update price
            update_price(test_symbol, test_price)
            
            # Step 2: Track signal
            signal_id = track_signal(test_symbol, test_prediction, test_price)
            
            # Step 3: Verify signal was stored
            if signal_id:
                tracker = SignalTracker()
                active_signals = tracker.get_active_signals()
                
                # Check if our signal is in the active signals
                found_signal = any(
                    signal.get('symbol') == test_symbol and 
                    signal.get('type') == test_prediction['type']
                    for signal in active_signals
                )
                
                if found_signal:
                    logger.info(f"   ✅ End-to-end flow successful (Signal ID: {signal_id})")
                    return True
                else:
                    logger.error("   ❌ Signal not found in active signals")
                    return False
            else:
                logger.error("   ❌ Signal tracking failed")
                return False
                
        except Exception as e:
            logger.error(f"   ❌ End-to-end test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict:
        """Run all integration tests"""
        logger.info("🚀 Starting Signal Integration Tests")
        logger.info("=" * 50)
        
        # Define test suite
        tests = [
            ("Component Imports", self.test_imports),
            ("Bridge Initialization", self.test_bridge_initialization),
            ("Database Connection", self.test_database_connection),
            ("Signal Tracking", self.test_signal_tracking),
            ("Price Updates", self.test_price_update),
            ("Signal Generator Init", self.test_signal_generator_init),
            ("Feature Calculation", self.test_feature_calculation),
            ("End-to-End Signal Flow", self.test_end_to_end_signal_flow)
        ]
        
        # Run tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            time.sleep(0.5)  # Brief pause between tests
        
        # Generate report
        success_rate = (self.passed_tests / self.test_count) * 100
        
        results = {
            'total_tests': self.test_count,
            'passed_tests': self.passed_tests,
            'failed_tests': self.test_count - self.passed_tests,
            'success_rate': success_rate,
            'test_details': self.test_results,
            'overall_status': 'PASSED' if success_rate == 100 else 'PARTIAL' if success_rate >= 50 else 'FAILED'
        }
        
        # Print summary
        logger.info("=" * 50)
        logger.info("🧪 TEST RESULTS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Tests: {results['total_tests']}")
        logger.info(f"Passed: {results['passed_tests']}")
        logger.info(f"Failed: {results['failed_tests']}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Status: {results['overall_status']}")
        
        # Detailed results
        logger.info("\\n📋 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
            logger.info(f"   {status_emoji} {test_name}: {result['status']}")
            if result['error']:
                logger.info(f"      Error: {result['error']}")
        
        # Recommendations
        if success_rate < 100:
            logger.info("\\n💡 RECOMMENDATIONS:")
            if not self.test_results.get("Component Imports", {}).get("status") == "PASSED":
                logger.info("   - Ensure all required modules are properly installed")
            if not self.test_results.get("Database Connection", {}).get("status") == "PASSED":
                logger.info("   - Check that the web dashboard database is accessible")
            if not self.test_results.get("Bridge Initialization", {}).get("status") == "PASSED":
                logger.info("   - Verify integration bridge configuration")
        else:
            logger.info("\\n🎉 ALL TESTS PASSED! Integration is working correctly.")
            logger.info("\\n🚀 Next steps:")
            logger.info("   1. Run: python scripts/signal_generator.py --mode scan --symbol BTCUSDT")
            logger.info("   2. Check web dashboard for signals: python scripts/start_dashboard.py")
            logger.info("   3. Start continuous monitoring: python scripts/signal_generator.py --mode monitor")
        
        return results

def main():
    """Main function to run integration tests"""
    print("=== NVBOT3 SIGNAL INTEGRATION TESTER ===")
    print("Testing signal generation and dashboard integration")
    print("=" * 60)
    
    try:
        tester = SignalIntegrationTester()
        results = tester.run_all_tests()
        
        # Return appropriate exit code
        if results['overall_status'] == 'PASSED':
            return 0
        elif results['overall_status'] == 'PARTIAL':
            return 1
        else:
            return 2
            
    except KeyboardInterrupt:
        logger.warning("⚠️ Tests interrupted by user")
        return 3
        
    except Exception as e:
        logger.error(f"💥 Critical error during testing: {e}")
        return 4

if __name__ == "__main__":
    exit(main())