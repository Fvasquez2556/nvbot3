#!/usr/bin/env python3
"""
Manual feature counting from FeatureCalculator based on the actual method code.
"""

def count_momentum_features():
    """Count features from calculate_momentum_features method."""
    features = []
    
    # ROC features - periods [5, 10, 14, 20]
    for period in [5, 10, 14, 20]:
        features.append(f'roc_{period}')
    
    # ADX features - periods [14, 20, 30] 
    for period in [14, 20, 30]:
        features.extend([
            f'adx_{period}',
            f'plus_di_{period}',
            f'minus_di_{period}'
        ])
    
    # MACD features (fixed)
    features.extend([
        'macd', 'macd_signal', 'macd_histogram',
        'macd_fast', 'macd_fast_signal', 'macd_fast_histogram'
    ])
    
    # RSI features - periods [7, 14, 21, 30]
    for period in [7, 14, 21, 30]:
        features.append(f'rsi_{period}')
    
    # OBV features
    features.extend(['obv', 'obv_sma_20', 'obv_ratio'])
    
    # Bollinger Bands - periods [20, 30]
    for period in [20, 30]:
        features.extend([
            f'bb_upper_{period}',
            f'bb_middle_{period}', 
            f'bb_lower_{period}',
            f'bb_width_{period}',
            f'bb_position_{period}'
        ])
    
    # Composite features
    features.extend(['momentum_score', 'momentum_strength'])
    
    return features

def count_rebound_features():
    """Count features from calculate_rebound_features method."""
    features = []
    
    # RSI oversold features
    features.extend([
        'rsi_oversold', 'rsi_oversold_extreme',
        'price_change_5', 'rsi_change_5', 'rsi_divergence'
    ])
    
    # MACD histogram features
    features.extend(['macd_hist_increasing', 'macd_hist_divergence'])
    
    # Volume features
    features.extend([
        'volume_sma_20', 'volume_ratio', 'volume_spike', 
        'volume_spike_extreme', 'volume_on_decline'
    ])
    
    # Support/Resistance features
    features.extend(['local_min_5', 'local_min_10'])
    
    # Distance to support - periods [10, 20, 50]
    for period in [10, 20, 50]:
        features.extend([
            f'min_low_{period}',
            f'distance_to_support_{period}'
        ])
    
    # Williams %R - periods [14, 20]
    for period in [14, 20]:
        features.extend([
            f'williams_r_{period}',
            f'williams_oversold_{period}'
        ])
    
    # Composite features
    features.extend(['oversold_score', 'rebound_potential'])
    
    return features

def count_regime_features():
    """Count features from calculate_regime_features method."""
    features = []
    
    # ATR features - periods [14, 20, 30]
    for period in [14, 20, 30]:
        features.extend([
            f'atr_{period}',
            f'atr_ratio_{period}'
        ])
    
    # Moving Averages - periods [10, 20, 50, 100, 200]
    for period in [10, 20, 50, 100, 200]:
        features.extend([
            f'sma_{period}',
            f'ema_{period}',
            f'sma_{period}_slope',
            f'ema_{period}_slope'
        ])
    
    # Price relationships - periods [20, 50, 200]
    for period in [20, 50, 200]:
        features.extend([
            f'price_above_sma_{period}',
            f'price_distance_sma_{period}'
        ])
    
    # MA alignment features
    features.extend(['ma_alignment_bull', 'ma_alignment_bear'])
    
    # Additional Bollinger Bands - periods [10, 20, 50] (note: 20 might overlap)
    for period in [10, 50]:  # Skip 20 as it's already in momentum
        features.append(f'bb_width_{period}')
    
    # True Range features
    features.extend(['true_range', 'true_range_norm'])
    
    # Regime classification features
    features.extend([
        'regime_trending', 'regime_strong_trend', 'regime_consolidation',
        'regime_low_vol', 'regime_high_vol'
    ])
    
    # Trend direction features  
    features.extend(['trend_bullish', 'trend_bearish'])
    
    # Composite feature
    features.append('regime_score')
    
    return features

def count_additional_features():
    """Count features from calculate_additional_features method."""
    features = []
    
    # Candlestick patterns
    features.extend(['doji', 'hammer', 'engulfing_bull'])
    
    # Stochastic - periods [14, 20]
    for period in [14, 20]:
        features.extend([
            f'stoch_k_{period}',
            f'stoch_d_{period}',
            f'stoch_oversold_{period}',
            f'stoch_overbought_{period}'
        ])
    
    # CCI - periods [14, 20] 
    for period in [14, 20]:
        features.extend([
            f'cci_{period}',
            f'cci_oversold_{period}',
            f'cci_overbought_{period}'
        ])
    
    # Money Flow Index
    features.extend(['mfi_14', 'mfi_oversold', 'mfi_overbought'])
    
    # Price Action features
    features.extend([
        'candle_range', 'candle_body', 'upper_shadow', 'lower_shadow'
    ])
    
    # Volume analysis
    features.extend(['vpt', 'ad'])
    
    # Time features (if datetime index exists)
    features.extend([
        'hour', 'day_of_week',
        'asian_session', 'european_session', 'american_session'
    ])
    
    return features

def main():
    """Main analysis function."""
    print("MANUAL FEATURECALCULATOR FEATURE COUNT")
    print("=" * 60)
    
    momentum_features = count_momentum_features()
    rebound_features = count_rebound_features() 
    regime_features = count_regime_features()
    additional_features = count_additional_features()
    
    print(f"MOMENTUM FEATURES ({len(momentum_features)}):")
    for i, feat in enumerate(momentum_features, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\nREBOUND FEATURES ({len(rebound_features)}):")
    for i, feat in enumerate(rebound_features, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\nREGIME FEATURES ({len(regime_features)}):")
    for i, feat in enumerate(regime_features, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\nADDITIONAL FEATURES ({len(additional_features)}):")
    for i, feat in enumerate(additional_features, 1):
        print(f"  {i:2d}. {feat}")
    
    total_features = len(momentum_features) + len(rebound_features) + len(regime_features) + len(additional_features)
    
    print(f"\nSUMMARY:")
    print("-" * 30)
    print(f"Momentum features: {len(momentum_features)}")
    print(f"Rebound features:  {len(rebound_features)}")
    print(f"Regime features:   {len(regime_features)}")  
    print(f"Additional features: {len(additional_features)}")
    print(f"TOTAL CALCULATED: {total_features}")
    
    # Check for potential overlaps
    all_features = momentum_features + rebound_features + regime_features + additional_features
    unique_features = list(set(all_features))
    
    print(f"\nOVERLAP CHECK:")
    print(f"Total features (with duplicates): {len(all_features)}")
    print(f"Unique features: {len(unique_features)}")
    print(f"Duplicate features: {len(all_features) - len(unique_features)}")
    
    if len(all_features) != len(unique_features):
        # Find duplicates
        seen = set()
        duplicates = set()
        for feat in all_features:
            if feat in seen:
                duplicates.add(feat)
            else:
                seen.add(feat)
        
        print(f"Duplicate features found: {sorted(duplicates)}")

if __name__ == "__main__":
    main()