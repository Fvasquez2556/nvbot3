# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NvBot3 is an AI-powered cryptocurrency trading system that combines historical data collection, machine learning models for pattern recognition, and a web dashboard for monitoring trading signals. The project includes anti-overfitting measures and supports 30 strategic cryptocurrency symbols across different market tiers.

## Development Environment Setup

### Virtual Environment
```bash
# Activate virtual environment (always required before running any commands)
nvbot3_env\Scripts\activate  # Windows
# source nvbot3_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_web.txt  # For web dashboard
```

### Project Validation
```bash
python scripts/validate_setup.py  # Validate complete setup
python scripts/verify_installation.py  # Verify installation
```

## Common Development Commands

### Data Management
```bash
# Download historical data for all configured symbols
python scripts/download_historical_data.py

# Process and validate data
python scripts/process_all_data.py
python scripts/validate_csv_format.py
```

### Model Training and Testing
```bash
# Train all models with anti-overfitting validation
python scripts/train_all_models.py

# Test individual models
python scripts/test_single_model.py

# Check training status
python scripts/check_training_status.py
```

### Web Dashboard
```bash
# Start the web dashboard (runs on Flask)
python scripts/start_dashboard.py
# or
python web_dashboard/app.py
```

### System Integration
```bash
# Full setup and run pipeline
python scripts/full_setup_and_run.py

# Test feedback system integration
python scripts/test_feedback_system.py
```

## Architecture Overview

### Core Components

**Data Pipeline** (`src/data/`):
- `data_validator.py` - Validates downloaded market data quality
- `feature_calculator.py` - Computes technical indicators and features
- `target_creator.py` - Creates target variables for model training

**Machine Learning** (`src/models/`):
- `regularized_models.py` - Anti-overfitting model implementations
- `model_trainer.py` - Handles model training with temporal validation

**Validation Framework** (`src/validation/`):
- `temporal_validator.py` - Time-series aware validation splits
- `walk_forward_validator.py` - Walk-forward testing implementation  
- `overfitting_detector.py` - Automated overfitting detection

**Web Dashboard** (`web_dashboard/`):
- `app.py` - Flask web application for signal monitoring
- `database/signal_tracker.py` - SQLite-based signal tracking system
- `templates/dashboard.html` - Web interface for viewing signals

**Integration Layer** (`integration/`):
- `nvbot3_feedback_bridge.py` - Bridges main bot with feedback system

### Configuration

Central configuration in `config/training_config.yaml` defines:
- 30 strategic cryptocurrency symbols organized by market tiers
- Anti-overfitting validation parameters and thresholds
- Model hyperparameters with aggressive regularization
- Feature engineering settings for robust indicators

### Data Flow

1. **Historical Data**: Downloaded via `scripts/download_historical_data.py` using Binance API with rate limiting
2. **Feature Engineering**: Technical indicators computed in `src/data/feature_calculator.py`
3. **Model Training**: Time-series aware training in `src/models/model_trainer.py` with overfitting detection
4. **Signal Generation**: Models generate trading signals tracked in SQLite database
5. **Web Dashboard**: Real-time monitoring of active signals and performance metrics

## Key Development Notes

- Always activate the virtual environment before running any scripts
- The system uses temporal validation (never random splits) to prevent data leakage
- All models include aggressive regularization to combat overfitting
- Rate limiting is implemented for Binance API calls (1200 requests/minute)
- The web dashboard runs independently and can monitor signals in real-time
- Configuration changes should be made in `config/training_config.yaml`
- Python 3.12.10 is the target runtime environment

## Testing

The project uses script-based testing rather than a formal test framework:
- `scripts/test_feedback_system.py` - Tests signal tracking integration
- `scripts/test_single_model.py` - Tests individual model performance
- Data validation is built into the pipeline and runs automatically

## Important Files

- `config/training_config.yaml` - Master configuration file
- `scripts/validate_setup.py` - Environment validation script
- `web_dashboard/database/signals.db` - SQLite database for signal tracking
- `requirements.txt` & `requirements_web.txt` - Python dependencies