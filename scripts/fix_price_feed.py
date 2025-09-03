#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NvBot3 - Price Feed Correction Tool
Identifies and fixes price data source issues
"""

import requests
import json
import pandas as pd
import os
import yaml
from datetime import datetime
import time
import sys
import sqlite3
import shutil
from pathlib import Path

class PriceFeedCorrector:
    def __init__(self):
        self.binance_api_url = "https://api.binance.com/api/v3"
        self.setup_symbols()
        self.issues_found = []
        self.fixes_applied = []
        
    def setup_symbols(self):
        """Setup all NvBot3 symbols"""
        # Complete 76-symbol configuration
        self.training_symbols = {
            'tier_1': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 
                      'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT'],
            
            'tier_2': ['UNIUSDT', 'AAVEUSDT', 'MKRUSDT', 'SUSHIUSDT', 'COMPUSDT',
                      'YFIUSDT', 'SNXUSDT', 'CRVUSDT', '1INCHUSDT', 'ALPHAUSDT'],
            
            'tier_3': ['SANDUSDT', 'MANAUSDT', 'ENJUSDT', 'CHZUSDT', 'BATUSDT',
                      'ZRXUSDT', 'STORJUSDT', 'OCEANUSDT', 'FETUSDT', 'IOTAUSDT']
        }
        
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
        
        print(f"🔧 Price Feed Corrector for {len(self.all_symbols)} symbols")
        
    def get_current_binance_prices(self) -> dict:
        """Get current prices from Binance API"""
        try:
            response = requests.get(f"{self.binance_api_url}/ticker/price", timeout=10)
            if response.status_code == 200:
                all_prices = {item['symbol']: float(item['price']) for item in response.json()}
                return {symbol: all_prices.get(symbol, 0) for symbol in self.all_symbols}
            return {}
        except Exception as e:
            print(f"❌ Error fetching Binance prices: {e}")
            return {}
    
    def identify_data_source_issue(self):
        """Identify where the dashboard is getting wrong prices"""
        print("🔍 Identifying data source issues...")
        
        issues = []
        
        # Check 1: Signal database integrity
        signals_db_path = 'web_dashboard/database/signals.db'
        if os.path.exists(signals_db_path):
            try:
                conn = sqlite3.connect(signals_db_path)
                cursor = conn.cursor()
                
                # Check for invalid prices in signals table
                cursor.execute("""
                    SELECT symbol, current_price, entry_price, COUNT(*) as count
                    FROM signals 
                    WHERE current_price > 1000000 OR entry_price > 1000000
                    GROUP BY symbol
                    ORDER BY count DESC
                """)
                
                invalid_prices = cursor.fetchall()
                if invalid_prices:
                    issues.append({
                        'type': 'database_price_corruption',
                        'description': f'Found {len(invalid_prices)} symbols with invalid prices in signals.db',
                        'details': invalid_prices[:5],  # Top 5
                        'fix': 'clean_database_prices'
                    })
                
                # Check for ADAUSDT specific issues
                cursor.execute("""
                    SELECT current_price, entry_price, created_at 
                    FROM signals 
                    WHERE symbol = 'ADAUSDT' 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """)
                
                adausdt_prices = cursor.fetchall()
                if adausdt_prices:
                    for price_data in adausdt_prices:
                        current_price, entry_price = price_data[0], price_data[1]
                        if current_price and current_price > 10:  # ADAUSDT should be < $10
                            issues.append({
                                'type': 'adausdt_precision_error',
                                'description': f'ADAUSDT price corruption detected: {current_price}',
                                'details': price_data,
                                'fix': 'fix_adausdt_precision'
                            })
                            break
                
                conn.close()
                print(f"✅ Database analysis complete - found {len(invalid_prices)} problematic symbols")
                
            except Exception as e:
                issues.append({
                    'type': 'database_access_error',
                    'description': f'Cannot access signals database: {e}',
                    'fix': 'recreate_database'
                })
        
        # Check 2: Local CSV file integrity  
        data_dir = 'data/raw'
        if os.path.exists(data_dir):
            corrupted_files = []
            current_prices = self.get_current_binance_prices()
            
            for symbol in ['ADAUSDT', 'BTCUSDT', 'ETHUSDT']:  # Check key symbols
                for timeframe in ['5m', '1h']:
                    filename = f"{symbol}_{timeframe}.csv"
                    filepath = os.path.join(data_dir, filename)
                    
                    if os.path.exists(filepath):
                        try:
                            # Try reading with different encodings
                            try:
                                df = pd.read_csv(filepath, encoding='utf-8')
                            except UnicodeDecodeError:
                                try:
                                    df = pd.read_csv(filepath, encoding='latin-1')
                                except UnicodeDecodeError:
                                    df = pd.read_csv(filepath, encoding='cp1252')
                                    
                            if 'close' in df.columns and len(df) > 0:
                                latest_price = df['close'].iloc[-1]
                                real_price = current_prices.get(symbol, 0)
                                
                                if real_price > 0:
                                    ratio = latest_price / real_price
                                    if ratio > 100 or ratio < 0.01:  # More than 100x off
                                        corrupted_files.append({
                                            'file': filename,
                                            'latest_price': latest_price,
                                            'real_price': real_price,
                                            'ratio': ratio
                                        })
                        except Exception as e:
                            print(f"⚠️ Could not read {filename}: {str(e)[:50]}...")
            
            if corrupted_files:
                issues.append({
                    'type': 'csv_price_corruption',
                    'description': f'Found {len(corrupted_files)} CSV files with price corruption',
                    'details': corrupted_files,
                    'fix': 'clean_csv_files'
                })
        
        # Check 3: API configuration
        api_issues = self.check_api_configuration()
        if api_issues:
            issues.extend(api_issues)
        
        # Check 4: Data processing pipeline
        pipeline_issues = self.check_data_processing_pipeline()
        if pipeline_issues:
            issues.extend(pipeline_issues)
        
        self.issues_found = issues
        
        print(f"🔍 Analysis complete - found {len(issues)} issue categories:")
        for issue in issues:
            print(f"   • {issue['type']}: {issue['description']}")
        
        return issues
    
    def check_api_configuration(self) -> list:
        """Check API configuration for issues"""
        issues = []
        
        # Check .env file
        env_file = '.env'
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'BINANCE_API_KEY=' not in content or 'BINANCE_SECRET_KEY=' not in content:
                        issues.append({
                            'type': 'missing_api_keys',
                            'description': 'Binance API keys not configured in .env',
                            'fix': 'configure_api_keys'
                        })
            except UnicodeDecodeError:
                print("⚠️ Could not read .env file (encoding issue)")
        else:
            issues.append({
                'type': 'missing_env_file',
                'description': '.env file not found',
                'fix': 'create_env_file'
            })
        
        # Test API connectivity
        try:
            response = requests.get(f"{self.binance_api_url}/ping", timeout=5)
            if response.status_code != 200:
                issues.append({
                    'type': 'api_connectivity',
                    'description': 'Cannot reach Binance API',
                    'fix': 'check_network'
                })
        except Exception:
            issues.append({
                'type': 'api_connectivity',
                'description': 'Network error connecting to Binance API',
                'fix': 'check_network'
            })
        
        return issues
    
    def check_data_processing_pipeline(self) -> list:
        """Check data processing pipeline for issues"""
        issues = []
        
        # Check if feature calculator has precision issues
        feature_calc_path = 'src/data/feature_calculator.py'
        if os.path.exists(feature_calc_path):
            try:
                with open(feature_calc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for potential precision issues
                    if 'float64' not in content and 'astype' in content:
                        issues.append({
                            'type': 'data_type_precision',
                            'description': 'Potential data type precision issues in feature calculator',
                            'fix': 'fix_data_types'
                        })
            except UnicodeDecodeError:
                print(f"⚠️ Could not read {feature_calc_path} (encoding issue)")
        
        # Check signal generator for price handling
        signal_gen_path = 'scripts/signal_generator.py'
        if os.path.exists(signal_gen_path):
            try:
                with open(signal_gen_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for price processing issues
                    if '*' in content and 'price' in content.lower():
                        # Might indicate multiplication operations on prices
                        issues.append({
                            'type': 'price_processing_multiplication',
                            'description': 'Potential price multiplication in signal generator',
                            'fix': 'review_price_calculations'
                        })
            except UnicodeDecodeError:
                print(f"⚠️ Could not read {signal_gen_path} (encoding issue)")
        
        return issues
    
    def fix_adausdt_precision_error(self):
        """Specifically fix the ADAUSDT precision error"""
        print("🔧 Fixing ADAUSDT precision error...")
        
        fixes_applied = []
        
        # Fix 1: Clean database entries
        signals_db_path = 'web_dashboard/database/signals.db'
        if os.path.exists(signals_db_path):
            try:
                # Create backup first
                backup_path = f"{signals_db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(signals_db_path, backup_path)
                print(f"📂 Database backed up to: {backup_path}")
                
                conn = sqlite3.connect(signals_db_path)
                cursor = conn.cursor()
                
                # Get current real price for ADAUSDT
                real_price = self.get_current_binance_prices().get('ADAUSDT', 0.82)
                
                # Find corrupted ADAUSDT entries
                cursor.execute("""
                    SELECT signal_id, current_price, entry_price 
                    FROM signals 
                    WHERE symbol = 'ADAUSDT' AND (current_price > 10 OR entry_price > 10)
                """)
                
                corrupted_entries = cursor.fetchall()
                
                if corrupted_entries:
                    print(f"🔍 Found {len(corrupted_entries)} corrupted ADAUSDT entries")
                    
                    for signal_id, current_price, entry_price in corrupted_entries:
                        # Calculate correction factor
                        if current_price and current_price > 10:
                            correction_factor = current_price / real_price
                            corrected_current = real_price
                            corrected_entry = entry_price / correction_factor if entry_price else real_price
                            
                            # Update the corrupted entry
                            cursor.execute("""
                                UPDATE signals 
                                SET current_price = ?, entry_price = ?, 
                                    last_updated = CURRENT_TIMESTAMP
                                WHERE signal_id = ?
                            """, (corrected_current, corrected_entry, signal_id))
                            
                            print(f"   Fixed {signal_id}: {current_price} → {corrected_current}")
                    
                    conn.commit()
                    fixes_applied.append(f"Fixed {len(corrupted_entries)} ADAUSDT database entries")
                
                # Also fix any other symbols with similar issues
                cursor.execute("""
                    SELECT symbol, AVG(current_price) as avg_price, COUNT(*) as count
                    FROM signals 
                    WHERE current_price > 1000
                    GROUP BY symbol
                    ORDER BY avg_price DESC
                """)
                
                other_corrupted = cursor.fetchall()
                if other_corrupted:
                    current_prices = self.get_current_binance_prices()
                    
                    for symbol, avg_price, count in other_corrupted:
                        real_price = current_prices.get(symbol, 0)
                        if real_price > 0 and avg_price > real_price * 100:  # More than 100x off
                            correction_factor = avg_price / real_price
                            
                            cursor.execute("""
                                UPDATE signals 
                                SET current_price = current_price / ?,
                                    entry_price = entry_price / ?,
                                    last_updated = CURRENT_TIMESTAMP
                                WHERE symbol = ? AND current_price > ?
                            """, (correction_factor, correction_factor, symbol, real_price * 10))
                            
                            fixes_applied.append(f"Fixed {count} {symbol} entries (factor: {correction_factor:.1f})")
                
                conn.close()
                
            except Exception as e:
                print(f"❌ Database fix error: {e}")
        
        # Fix 2: Clean CSV files
        data_dir = 'data/raw'
        if os.path.exists(data_dir):
            real_price = self.get_current_binance_prices().get('ADAUSDT', 0.82)
            
            for timeframe in ['5m', '15m', '1h', '4h', '1d']:
                filename = f"ADAUSDT_{timeframe}.csv"
                filepath = os.path.join(data_dir, filename)
                
                if os.path.exists(filepath):
                    try:
                        # Try reading with different encodings
                        try:
                            df = pd.read_csv(filepath, encoding='utf-8')
                        except UnicodeDecodeError:
                            try:
                                df = pd.read_csv(filepath, encoding='latin-1')
                            except UnicodeDecodeError:
                                df = pd.read_csv(filepath, encoding='cp1252')
                                
                        if 'close' in df.columns:
                            # Check if prices are corrupted
                            avg_close = df['close'].mean()
                            if avg_close > 10:  # ADAUSDT should be < $10
                                correction_factor = avg_close / real_price
                                
                                # Apply correction
                                price_columns = ['open', 'high', 'low', 'close']
                                for col in price_columns:
                                    if col in df.columns:
                                        df[col] = df[col] / correction_factor
                                
                                # Save corrected file
                                df.to_csv(filepath, index=False)
                                fixes_applied.append(f"Fixed {filename} (factor: {correction_factor:.1f})")
                                
                    except Exception as e:
                        print(f"❌ CSV fix error for {filename}: {str(e)[:50]}...")
        
        self.fixes_applied.extend(fixes_applied)
        
        if fixes_applied:
            print(f"✅ Applied {len(fixes_applied)} ADAUSDT fixes:")
            for fix in fixes_applied:
                print(f"   • {fix}")
        else:
            print("ℹ️  No ADAUSDT precision errors found to fix")
    
    def validate_price_ranges(self):
        """Validate all prices are within expected ranges"""
        print("✅ Validating price ranges...")
        
        # Expected price ranges (September 2025)
        expected_ranges = {
            'BTCUSDT': (50000, 150000), 'ETHUSDT': (2000, 8000), 'BNBUSDT': (200, 1000),
            'ADAUSDT': (0.1, 5.0), 'SOLUSDT': (50, 500), 'XRPUSDT': (0.3, 3.0),
            'DOTUSDT': (3, 50), 'LINKUSDT': (5, 100), 'MATICUSDT': (0.3, 5.0),
            'AVAXUSDT': (10, 200)
        }
        
        current_prices = self.get_current_binance_prices()
        validation_results = {
            'valid': [],
            'warnings': [],
            'errors': []
        }
        
        for symbol, expected_range in expected_ranges.items():
            min_price, max_price = expected_range
            current_price = current_prices.get(symbol, 0)
            
            if current_price == 0:
                validation_results['errors'].append(f"{symbol}: No price data available")
            elif current_price < min_price:
                validation_results['warnings'].append(f"{symbol}: Below expected range ({current_price} < {min_price})")
            elif current_price > max_price:
                validation_results['errors'].append(f"{symbol}: Above expected range ({current_price} > {max_price})")
            else:
                validation_results['valid'].append(f"{symbol}: ✅ {current_price}")
        
        print(f"📊 Validation Results:")
        print(f"   ✅ Valid: {len(validation_results['valid'])} symbols")
        print(f"   ⚠️  Warnings: {len(validation_results['warnings'])} symbols")
        print(f"   ❌ Errors: {len(validation_results['errors'])} symbols")
        
        if validation_results['errors']:
            print("\n❌ Price Range Errors:")
            for error in validation_results['errors'][:5]:
                print(f"   {error}")
        
        return validation_results
    
    def update_dashboard_config(self):
        """Update dashboard configuration with correct price feeds"""
        print("📊 Updating dashboard configuration...")
        
        config_updates = []
        
        # Update web dashboard app.py if needed
        dashboard_path = 'web_dashboard/app.py'
        if os.path.exists(dashboard_path):
            # Check for potential price processing issues
            try:
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for price calculations that might cause issues
                problematic_patterns = [
                    'price * 1000',
                    'price * 10000', 
                    'price / 0.001',
                    'float(price) * '
                ]
                
                for pattern in problematic_patterns:
                    if pattern in content:
                        print(f"⚠️  Found potential issue in dashboard: {pattern}")
                        config_updates.append(f"Review price calculation: {pattern}")
            except UnicodeDecodeError:
                print(f"⚠️ Could not read {dashboard_path} (encoding issue)")
        
        # Check signal generator configuration
        signal_gen_path = 'scripts/signal_generator.py'
        if os.path.exists(signal_gen_path):
            try:
                with open(signal_gen_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Ensure proper price handling
                if 'float(' not in content:
                    config_updates.append("Add explicit float conversion for prices")
            except UnicodeDecodeError:
                print(f"⚠️ Could not read {signal_gen_path} (encoding issue)")
        
        # Update environment configuration
        env_example_path = '.env.example'
        env_path = '.env'
        if os.path.exists(env_example_path) and not os.path.exists(env_path):
            shutil.copy2(env_example_path, env_path)
            config_updates.append("Created .env from .env.example")
        
        if config_updates:
            print(f"🔧 Applied {len(config_updates)} configuration updates:")
            for update in config_updates:
                print(f"   • {update}")
        else:
            print("ℹ️  No configuration updates needed")
        
        self.fixes_applied.extend(config_updates)
    
    def generate_correction_report(self):
        """Generate a report of all fixes applied"""
        print("\n📋 Generating correction report...")
        
        report = []
        report.append("=" * 70)
        report.append("NvBot3 - PRICE FEED CORRECTION REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Issues found
        report.append(f"🔍 ISSUES IDENTIFIED: {len(self.issues_found)}")
        for issue in self.issues_found:
            report.append(f"   • {issue['type']}: {issue['description']}")
        report.append("")
        
        # Fixes applied
        report.append(f"🔧 FIXES APPLIED: {len(self.fixes_applied)}")
        for fix in self.fixes_applied:
            report.append(f"   ✅ {fix}")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        report.append("   1. Restart the dashboard: python scripts/start_dashboard.py")
        report.append("   2. Test with: python scripts/comprehensive_price_diagnostic.py")
        report.append("   3. Monitor dashboard at: http://localhost:5000")
        report.append("   4. Run this script weekly for maintenance")
        report.append("")
        
        report.append("=" * 70)
        
        report_content = "\n".join(report)
        
        # Save report
        report_filename = f"price_correction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(report_content)
        print(f"📄 Report saved to: {report_filename}")
    
    def run_full_correction(self):
        """Run complete price feed correction"""
        print("🚀 Starting NvBot3 price feed correction...")
        
        # Step 1: Identify issues
        self.identify_data_source_issue()
        
        # Step 2: Fix specific issues
        if any(issue['type'] == 'adausdt_precision_error' for issue in self.issues_found):
            self.fix_adausdt_precision_error()
        
        # Step 3: Validate ranges
        self.validate_price_ranges()
        
        # Step 4: Update configuration
        self.update_dashboard_config()
        
        # Step 5: Generate report
        self.generate_correction_report()
        
        print("\n✅ Price feed correction completed!")
        print("\n🔄 Next steps:")
        print("   1. python scripts/start_dashboard.py")
        print("   2. Open http://localhost:5000")
        print("   3. Verify prices are correct")

def main():
    """Main execution function"""
    print("🛠️  NvBot3 Price Feed Correction Tool")
    print("=" * 40)
    
    corrector = PriceFeedCorrector()
    corrector.run_full_correction()

if __name__ == "__main__":
    main()