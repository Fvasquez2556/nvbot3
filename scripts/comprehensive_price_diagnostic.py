#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NvBot3 - Comprehensive Price Diagnostic Tool
Analyzes price data issues across all 76 monitored cryptocurrencies
"""

import requests
import json
import pandas as pd
import os
from datetime import datetime, timedelta
import sys
import yaml
from typing import Dict, List, Optional
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time

class ComprehensivePriceDiagnostic:
    def __init__(self, config_path='config/training_config.yaml'):
        self.binance_api_url = "https://api.binance.com/api/v3"
        self.config_path = config_path
        self.setup_symbol_lists()
        self.load_config()
        
    def setup_symbol_lists(self):
        """Setup comprehensive symbol lists for NvBot3"""
        
        # Original 30 training symbols (3 tiers)
        self.training_symbols = {
            'tier_1': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 
                      'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT'],
            
            'tier_2': ['UNIUSDT', 'AAVEUSDT', 'MKRUSDT', 'SUSHIUSDT', 'COMPUSDT',
                      'YFIUSDT', 'SNXUSDT', 'CRVUSDT', '1INCHUSDT', 'ALPHAUSDT'],
            
            'tier_3': ['SANDUSDT', 'MANAUSDT', 'ENJUSDT', 'CHZUSDT', 'BATUSDT',
                      'ZRXUSDT', 'STORJUSDT', 'OCEANUSDT', 'FETUSDT', 'IOTAUSDT']
        }
        
        # Additional 46 monitoring symbols (3 monitoring tiers)
        self.monitoring_symbols = {
            'monitoring_tier_m1': [
                'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'TRXUSDT', 'ATOMUSDT',
                'VETUSDT', 'NEOUSDT', 'DASHUSDT', 'ETCUSDT', 'ZECUSDT',
                'QTUMUSDT', 'LSKUSDT', 'ICXUSDT', 'ZILUSDT', 'ONTUSDT', 'RVNUSDT'
            ],
            
            'monitoring_tier_m2': [
                'CAKEUSDT', 'BANDUSDT', 'KNCUSDT', 'LRCUSDT', 'CHRUSDT',
                'INJUSDT', 'DYDXUSDT', 'ENSUSDT', 'MASKUSDT', 'PERPUSDT',
                'SUPERUSDT', 'ONEUSDT', 'NEARUSDT', 'ALGOUSDT', 'EGLDUSDT'
            ],
            
            'monitoring_tier_m3': [
                'AXSUSDT', 'SLPUSDT', 'ALICEUSDT', 'TLMUSDT', 'GALAUSDT',
                'FLMUSDT', 'CITYUSDT', 'BICOUSDT', 'GMTUSDT', 'MOVEUSDT',
                'FLOWUSDT', 'THETAUSDT', 'TFUELUSDT', 'WOOUSDT', 'JASMYUSDT'
            ]
        }
        
        # Flatten all symbols
        self.all_training_symbols = []
        for tier_symbols in self.training_symbols.values():
            self.all_training_symbols.extend(tier_symbols)
            
        self.all_monitoring_symbols = []
        for tier_symbols in self.monitoring_symbols.values():
            self.all_monitoring_symbols.extend(tier_symbols)
            
        self.all_symbols = self.all_training_symbols + self.all_monitoring_symbols
        
        print(f"Configured for {len(self.all_symbols)} total symbols:")
        print(f"   - {len(self.all_training_symbols)} training symbols")
        print(f"   - {len(self.all_monitoring_symbols)} monitoring symbols")
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"WARNING: Config file not found: {self.config_path}")
            self.config = {}
    
    def get_current_binance_prices(self, max_workers: int = 10) -> Optional[Dict]:
        """Get current prices from Binance API using parallel requests"""
        print(f"🔍 Fetching current prices for {len(self.all_symbols)} symbols...")
        
        try:
            # Get all prices at once (more efficient)
            response = requests.get(f"{self.binance_api_url}/ticker/price", timeout=30)
            
            if response.status_code == 200:
                all_binance_prices = {item['symbol']: float(item['price']) 
                                    for item in response.json()}
                
                # Filter only our symbols
                current_prices = {}
                missing_symbols = []
                
                for symbol in self.all_symbols:
                    if symbol in all_binance_prices:
                        current_prices[symbol] = all_binance_prices[symbol]
                    else:
                        missing_symbols.append(symbol)
                
                print(f"✅ Found prices for {len(current_prices)}/{len(self.all_symbols)} symbols")
                if missing_symbols:
                    print(f"⚠️  Missing symbols: {missing_symbols}")
                
                return {
                    'prices': current_prices,
                    'missing_symbols': missing_symbols,
                    'timestamp': datetime.now()
                }
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None
    
    def analyze_dashboard_prices(self, dashboard_data: Dict) -> Dict:
        """Compare dashboard prices with current Binance prices"""
        print("\n🔍 Analyzing dashboard vs real prices...")
        
        current_data = self.get_current_binance_prices()
        if not current_data:
            return {"error": "Could not fetch current prices"}
        
        current_prices = current_data['prices']
        analysis = {
            'timestamp': datetime.now(),
            'total_symbols_checked': len(dashboard_data),
            'price_discrepancies': {},
            'critical_errors': [],
            'warnings': [],
            'summary': {}
        }
        
        for symbol, dashboard_price in dashboard_data.items():
            if symbol in current_prices:
                real_price = current_prices[symbol]
                dashboard_price_float = float(dashboard_price)
                
                # Calculate percentage difference
                if real_price > 0:
                    diff_percent = ((dashboard_price_float - real_price) / real_price) * 100
                    
                    analysis['price_discrepancies'][symbol] = {
                        'dashboard_price': dashboard_price_float,
                        'real_price': real_price,
                        'difference_percent': diff_percent,
                        'difference_absolute': dashboard_price_float - real_price
                    }
                    
                    # Classify severity
                    if abs(diff_percent) > 100:  # More than 100% difference
                        analysis['critical_errors'].append({
                            'symbol': symbol,
                            'issue': f'Price difference: {diff_percent:.1f}%',
                            'dashboard': dashboard_price_float,
                            'real': real_price
                        })
                    elif abs(diff_percent) > 10:  # More than 10% difference
                        analysis['warnings'].append({
                            'symbol': symbol,
                            'issue': f'Price difference: {diff_percent:.1f}%',
                            'dashboard': dashboard_price_float,
                            'real': real_price
                        })
        
        # Generate summary
        analysis['summary'] = {
            'critical_errors_count': len(analysis['critical_errors']),
            'warnings_count': len(analysis['warnings']),
            'symbols_with_issues': len(analysis['critical_errors']) + len(analysis['warnings']),
            'accuracy_percentage': ((len(dashboard_data) - len(analysis['critical_errors']) - len(analysis['warnings'])) / len(dashboard_data)) * 100 if dashboard_data else 0
        }
        
        return analysis
    
    def check_local_data_files(self) -> Dict:
        """Check all local CSV files for data quality issues"""
        print(f"\n📁 Checking local data files for {len(self.all_symbols)} symbols...")
        
        data_dir = 'data/raw'
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        
        if not os.path.exists(data_dir):
            return {"error": f"Data directory not found: {data_dir}"}
        
        file_analysis = {
            'timestamp': datetime.now(),
            'total_expected_files': len(self.all_symbols) * len(timeframes),
            'files_found': 0,
            'files_missing': 0,
            'files_with_issues': {},
            'data_freshness': {},
            'price_anomalies': {},
            'summary_by_symbol': {}
        }
        
        # Define expected price ranges (September 2025)
        expected_ranges = {
            'BTCUSDT': (50000, 150000), 'ETHUSDT': (2000, 8000), 'BNBUSDT': (200, 1000),
            'ADAUSDT': (0.1, 5.0), 'SOLUSDT': (50, 500), 'XRPUSDT': (0.3, 3.0),
            'DOTUSDT': (3, 50), 'LINKUSDT': (5, 100), 'MATICUSDT': (0.3, 5.0),
            'AVAXUSDT': (10, 200), 'UNIUSDT': (3, 50), 'AAVEUSDT': (50, 500),
            'MKRUSDT': (500, 5000), 'SUSHIUSDT': (0.5, 10), 'COMPUSDT': (30, 500),
            'YFIUSDT': (3000, 50000), 'SNXUSDT': (1, 20), 'CRVUSDT': (0.2, 5),
            '1INCHUSDT': (0.1, 3), 'ALPHAUSDT': (0.05, 2), 'ENJUSDT': (0.01, 1.0),
            'SANDUSDT': (0.1, 3), 'MANAUSDT': (0.2, 5), 'CHZUSDT': (0.05, 1),
            'BATUSDT': (0.1, 3), 'ZRXUSDT': (0.2, 5), 'STORJUSDT': (0.2, 5),
            'OCEANUSDT': (0.2, 5), 'FETUSDT': (0.5, 10), 'IOTAUSDT': (0.1, 3),
            # Additional monitoring symbols ranges
            'LTCUSDT': (50, 200), 'BCHUSDT': (200, 800), 'XLMUSDT': (0.05, 0.5),
            'TRXUSDT': (0.05, 0.3), 'ATOMUSDT': (5, 50), 'VETUSDT': (0.01, 0.1),
            'NEOUSDT': (5, 100), 'DASHUSDT': (20, 200), 'ETCUSDT': (15, 100),
            'ZECUSDT': (20, 200), 'QTUMUSDT': (1, 20), 'LSKUSDT': (0.5, 10),
            'ICXUSDT': (0.1, 5), 'ZILUSDT': (0.01, 0.2), 'ONTUSDT': (0.1, 5),
            'RVNUSDT': (0.01, 0.2), 'CAKEUSDT': (1, 20), 'BANDUSDT': (0.5, 20),
            'KNCUSDT': (0.3, 10), 'LRCUSDT': (0.1, 5), 'CHRUSDT': (0.05, 2),
            'INJUSDT': (5, 100), 'DYDXUSDT': (0.5, 20), 'ENSUSDT': (10, 100),
            'MASKUSDT': (1, 50), 'PERPUSDT': (0.3, 10), 'SUPERUSDT': (0.05, 5),
            'ONEUSDT': (0.005, 0.1), 'NEARUSDT': (1, 20), 'ALGOUSDT': (0.1, 3),
            'EGLDUSDT': (20, 300), 'AXSUSDT': (1, 50), 'SLPUSDT': (0.001, 0.1),
            'ALICEUSDT': (0.5, 20), 'TLMUSDT': (0.005, 0.5), 'GALAUSDT': (0.01, 0.5),
            'FLMUSDT': (0.05, 2), 'CITYUSDT': (1, 50), 'BICOUSDT': (0.1, 5),
            'GMTUSDT': (0.05, 2), 'MOVEUSDT': (0.1, 10), 'FLOWUSDT': (0.3, 20),
            'THETAUSDT': (0.5, 15), 'TFUELUSDT': (0.01, 1), 'WOOUSDT': (0.1, 5),
            'JASMYUSDT': (0.001, 0.1)
        }
        
        for symbol in self.all_symbols:
            symbol_issues = []
            symbol_files_found = 0
            
            for timeframe in timeframes:
                filename = f"{symbol}_{timeframe}.csv"
                filepath = os.path.join(data_dir, filename)
                
                if os.path.exists(filepath):
                    symbol_files_found += 1
                    file_analysis['files_found'] += 1
                    
                    try:
                        # Try reading with UTF-8 first, fallback to latin-1
                        try:
                            df = pd.read_csv(filepath, encoding='utf-8')
                        except UnicodeDecodeError:
                            df = pd.read_csv(filepath, encoding='latin-1')
                        
                        # Check data freshness
                        if 'timestamp' in df.columns and len(df) > 0:
                            latest_timestamp = df['timestamp'].max()
                            latest_date = datetime.fromtimestamp(latest_timestamp / 1000)
                            days_old = (datetime.now() - latest_date).days
                            
                            if days_old > 7:  # More than a week old
                                symbol_issues.append(f"{timeframe}: {days_old} days old")
                        
                        # Check for price anomalies
                        if 'close' in df.columns and len(df) > 0:
                            latest_price = df['close'].iloc[-1]
                            
                            if symbol in expected_ranges:
                                min_price, max_price = expected_ranges[symbol]
                                if not (min_price <= latest_price <= max_price):
                                    symbol_issues.append(f"{timeframe}: Price anomaly {latest_price}")
                                    
                                    if symbol not in file_analysis['price_anomalies']:
                                        file_analysis['price_anomalies'][symbol] = []
                                    file_analysis['price_anomalies'][symbol].append({
                                        'timeframe': timeframe,
                                        'price': latest_price,
                                        'expected_range': expected_ranges[symbol]
                                    })
                        
                        # Check data completeness
                        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        if missing_columns:
                            symbol_issues.append(f"{timeframe}: Missing columns {missing_columns}")
                        
                        # Check for data gaps
                        if len(df) < 100:  # Too little data
                            symbol_issues.append(f"{timeframe}: Only {len(df)} records")
                            
                    except Exception as e:
                        symbol_issues.append(f"{timeframe}: Read error - {str(e)[:50]}...")
                        
                else:
                    file_analysis['files_missing'] += 1
                    symbol_issues.append(f"{timeframe}: File missing")
            
            # Summary for this symbol
            file_analysis['summary_by_symbol'][symbol] = {
                'files_found': symbol_files_found,
                'files_expected': len(timeframes),
                'completion_percentage': (symbol_files_found / len(timeframes)) * 100,
                'issues': symbol_issues
            }
            
            if symbol_issues:
                file_analysis['files_with_issues'][symbol] = symbol_issues
        
        return file_analysis
    
    def check_dashboard_data_source(self) -> Dict:
        """Check where the dashboard is getting its price data"""
        print("\n🔍 Analyzing dashboard data sources...")
        
        data_sources = {
            'signal_tracker_db': False,
            'local_csv_files': False,
            'binance_api_direct': False,
            'websocket_feed': False
        }
        
        # Check if signals.db exists and has recent data
        signals_db_path = 'web_dashboard/database/signals.db'
        if os.path.exists(signals_db_path):
            data_sources['signal_tracker_db'] = True
            print(f"✅ Found signals database: {signals_db_path}")
        
        # Check if local CSV files exist
        data_dir = 'data/raw'
        if os.path.exists(data_dir) and os.listdir(data_dir):
            data_sources['local_csv_files'] = True
            print(f"✅ Found local CSV files: {len(os.listdir(data_dir))} files")
        
        # Check for API configuration
        env_file = '.env'
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    env_content = f.read()
                    if 'BINANCE_API_KEY' in env_content:
                        data_sources['binance_api_direct'] = True
                        print("✅ Found Binance API configuration")
            except UnicodeDecodeError:
                print("⚠️ Could not read .env file (encoding issue)")
        
        # Check for WebSocket configuration
        dashboard_files = ['web_dashboard/app.py', 'scripts/signal_generator.py']
        for file_path in dashboard_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'websocket' in content.lower() or 'socketio' in content.lower():
                            data_sources['websocket_feed'] = True
                            print("✅ Found WebSocket/SocketIO configuration")
                            break
                except UnicodeDecodeError:
                    print(f"⚠️ Could not read {file_path} (encoding issue)")
                    continue
        
        return data_sources
    
    def analyze_adausdt_precision_error(self, dashboard_price: float = 2012.254771) -> Dict:
        """Specifically analyze the ADAUSDT precision error"""
        print(f"\n🔍 Analyzing ADAUSDT precision error (Dashboard: {dashboard_price})...")
        
        # Get real ADAUSDT price
        try:
            response = requests.get(f"{self.binance_api_url}/ticker/price?symbol=ADAUSDT")
            real_price = float(response.json()['price'])
            
            # Calculate the multiplication factor
            multiplication_factor = dashboard_price / real_price
            
            # Analyze possible causes
            analysis = {
                'dashboard_price': dashboard_price,
                'real_price': real_price,
                'multiplication_factor': multiplication_factor,
                'possible_causes': [],
                'suggested_fixes': []
            }
            
            # Check for common precision errors
            if abs(multiplication_factor - 1000) < 100:
                analysis['possible_causes'].append("Price stored in wrong units (satoshis vs dollars)")
                analysis['suggested_fixes'].append("Divide price by 1000")
            
            if abs(multiplication_factor - 10000) < 1000:
                analysis['possible_causes'].append("Price multiplied by 10000 (basis points)")
                analysis['suggested_fixes'].append("Divide price by 10000")
            
            # Check if it's a timestamp vs price confusion
            current_timestamp = int(datetime.now().timestamp())
            if abs(dashboard_price - current_timestamp) < 100000:
                analysis['possible_causes'].append("Timestamp stored as price")
                analysis['suggested_fixes'].append("Check data source column mapping")
            
            # Check for data type errors
            if dashboard_price > 1000000:
                analysis['possible_causes'].append("Price stored in microunits")
                analysis['suggested_fixes'].append("Apply correct unit conversion")
            
            return analysis
            
        except Exception as e:
            return {"error": f"Could not fetch ADAUSDT price: {e}"}
    
    def generate_diagnostic_report(self, dashboard_data: Optional[Dict] = None) -> str:
        """Generate comprehensive diagnostic report"""
        print("\n📊 Generating comprehensive diagnostic report...")
        
        report = []
        report.append("=" * 80)
        report.append("NvBot3 - COMPREHENSIVE PRICE DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total symbols configured: {len(self.all_symbols)}")
        report.append("")
        
        # Current prices analysis
        current_data = self.get_current_binance_prices()
        if current_data:
            report.append("🔍 CURRENT BINANCE PRICES STATUS:")
            report.append(f"✅ Successfully fetched: {len(current_data['prices'])} symbols")
            if current_data['missing_symbols']:
                report.append(f"❌ Missing from Binance: {len(current_data['missing_symbols'])} symbols")
                report.append(f"   Missing symbols: {', '.join(current_data['missing_symbols'][:10])}...")
            report.append("")
        
        # Dashboard data sources analysis
        data_sources = self.check_dashboard_data_source()
        report.append("📊 DASHBOARD DATA SOURCES:")
        for source, available in data_sources.items():
            status = "✅" if available else "❌"
            report.append(f"{status} {source.replace('_', ' ').title()}")
        report.append("")
        
        # Dashboard vs reality comparison
        if dashboard_data:
            analysis = self.analyze_dashboard_prices(dashboard_data)
            report.append("🎯 DASHBOARD ACCURACY ANALYSIS:")
            report.append(f"Symbols analyzed: {analysis['total_symbols_checked']}")
            report.append(f"Critical errors: {analysis['summary']['critical_errors_count']}")
            report.append(f"Warnings: {analysis['summary']['warnings_count']}")
            report.append(f"Overall accuracy: {analysis['summary']['accuracy_percentage']:.1f}%")
            
            if analysis['critical_errors']:
                report.append("\n🚨 CRITICAL PRICE ERRORS:")
                for error in analysis['critical_errors'][:5]:  # Show top 5
                    report.append(f"   {error['symbol']}: Dashboard={error['dashboard']}, Real={error['real']}")
            
            # ADAUSDT specific analysis
            if 'ADAUSDT' in dashboard_data:
                adausdt_analysis = self.analyze_adausdt_precision_error(dashboard_data['ADAUSDT'])
                if 'error' not in adausdt_analysis:
                    report.append(f"\n🔬 ADAUSDT PRECISION ERROR ANALYSIS:")
                    report.append(f"Multiplication factor: {adausdt_analysis['multiplication_factor']:.1f}x")
                    report.append("Possible causes:")
                    for cause in adausdt_analysis['possible_causes']:
                        report.append(f"   • {cause}")
                    report.append("Suggested fixes:")
                    for fix in adausdt_analysis['suggested_fixes']:
                        report.append(f"   → {fix}")
            
            report.append("")
        
        # Local data files analysis
        file_analysis = self.check_local_data_files()
        if 'error' not in file_analysis:
            report.append("📁 LOCAL DATA FILES STATUS:")
            report.append(f"Total expected files: {file_analysis['total_expected_files']}")
            report.append(f"Files found: {file_analysis['files_found']}")
            report.append(f"Files missing: {file_analysis['files_missing']}")
            report.append(f"Completion rate: {(file_analysis['files_found']/file_analysis['total_expected_files']*100):.1f}%")
            
            if file_analysis['price_anomalies']:
                report.append(f"\n⚠️  PRICE ANOMALIES DETECTED: {len(file_analysis['price_anomalies'])} symbols")
                for symbol, anomalies in list(file_analysis['price_anomalies'].items())[:5]:
                    report.append(f"   {symbol}: {anomalies[0]['price']} (expected: {anomalies[0]['expected_range']})")
            
            # Show symbols with most issues
            symbols_with_issues = [(symbol, len(data['issues'])) for symbol, data in file_analysis['summary_by_symbol'].items() if data['issues']]
            if symbols_with_issues:
                symbols_with_issues.sort(key=lambda x: x[1], reverse=True)
                report.append(f"\n⚠️  SYMBOLS WITH MOST ISSUES:")
                for symbol, issue_count in symbols_with_issues[:10]:
                    report.append(f"   {symbol}: {issue_count} issues")
        
        report.append("")
        report.append("🔧 RECOMMENDED ACTIONS:")
        report.append("1. Run: python scripts/fix_price_feed.py")
        report.append("2. Restart dashboard: python scripts/start_dashboard.py")
        report.append("3. Verify prices at: http://localhost:5000")
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_full_diagnostic(self, dashboard_data: Optional[Dict] = None):
        """Run complete diagnostic and save report"""
        print("🚀 Starting comprehensive NvBot3 price diagnostic...")
        
        # Generate report
        report = self.generate_diagnostic_report(dashboard_data)
        
        # Save report
        report_filename = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Print summary
        print("\n" + report)
        print(f"\n📄 Full report saved to: {report_filename}")

def main():
    """Main execution function"""
    
    # Example dashboard data showing the issues
    dashboard_data = {
        'ENJUSDT': 0.0679,       # Potentially correct
        'ADAUSDT': 2012.254771,  # CRITICAL ERROR - ~2454x too high
        'BTCUSDT': 67450.0,      # Potentially correct
        'ETHUSDT': 3250.0        # Potentially correct
    }
    
    print("NvBot3 Comprehensive Price Diagnostic Tool")
    print("=" * 50)
    
    diagnostic = ComprehensivePriceDiagnostic()
    diagnostic.run_full_diagnostic(dashboard_data)

if __name__ == "__main__":
    main()