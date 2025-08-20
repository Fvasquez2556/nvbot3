#!/usr/bin/env python3
"""
NvBot3 - Historical Data Downloader
==================================

Descarga datos históricos de Binance para entrenamiento de modelos.
Basado en las instrucciones del Training Pipeline NvBot3.

Características:
- Descarga 2+ años de datos históricos
- Múltiples símbolos y timeframes
- Rate limiting respetado
- Progress tracking
- Resume capability
- Data validation
"""

import ccxt
import pandas as pd
import time
import os
import sys
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Configurar logging
def setup_logging():
    """Configurar sistema de logging."""
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_download.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

# Cargar variables de entorno al inicio del programa
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

logger.info(f"Cargando archivo .env desde: {env_path}")
logger.info(f"Variables de entorno cargadas correctamente")

class HistoricalDataDownloader:
    """
    Descargador de datos históricos de Binance optimizado para entrenamiento ML.
    
    Especificaciones según instrucciones:
    - Símbolos: BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT
    - Timeframes: 5m, 15m, 1h, 4h, 1d
    - Período: Desde enero 2022 hasta presente
    - Rate limiting: Máximo 1200 requests/minuto
    """
    
    def __init__(self):
        """Inicializar conexión Binance y configuración."""
        # Las variables de entorno ya están cargadas al inicio del programa
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if api_key and secret_key:
            logger.info(f"SUCCESS: API Key cargada: {api_key[:8]}...")
            logger.info(f"SUCCESS: Secret Key cargada: {secret_key[:8]}...")
        else:
            logger.error("ERROR: No se pudieron cargar las claves API del archivo .env")
            logger.error("ERROR: Claves API de Binance no configuradas!")
            logger.error("Configura BINANCE_API_KEY y BINANCE_SECRET_KEY en el archivo .env")
        
        # Configuración según instrucciones
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        self.timeframes = ['5m', '15m', '1h', '4h', '1d']
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime.now()
        
        # Setup de directorios con paths absolutos
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_data_dir = os.path.join(project_root, 'data', 'raw')
        self.logs_dir = os.path.join(project_root, 'logs')
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Inicializar conexión Binance
        if api_key and secret_key:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
                'timeout': 30000,
                'rateLimit': 50,  # 50ms entre requests para respetar límites
                'enableRateLimit': True,
            })
        else:
            self.exchange = None
        
        # Columnas de datos según especificaciones
        self.columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        
        logger.info("HistoricalDataDownloader inicializado correctamente")
        logger.info(f"Símbolos a descargar: {self.symbols}")
        logger.info(f"Timeframes: {self.timeframes}")
        logger.info(f"Período: {self.start_date.date()} a {self.end_date.date()}")

    def get_file_path(self, symbol: str, timeframe: str) -> str:
        """Generar ruta de archivo para símbolo y timeframe."""
        return os.path.join(self.raw_data_dir, f"{symbol}_{timeframe}.csv")

    def file_exists_and_valid(self, symbol: str, timeframe: str) -> bool:
        """Verificar si archivo existe y tiene datos válidos."""
        file_path = self.get_file_path(symbol, timeframe)
        if not os.path.exists(file_path):
            return False
        
        try:
            df = pd.read_csv(file_path)
            if len(df) < 100:  # Mínimo 100 registros para considerar válido
                return False
            
            # Verificar que tiene datos recientes (últimos 30 días)
            latest_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
            if (datetime.now() - latest_timestamp).days > 30:
                return False
                
            logger.info(f"Archivo existente válido: {file_path} ({len(df)} registros)")
            return True
            
        except Exception as e:
            logger.warning(f"Error al validar archivo {file_path}: {e}")
            return False

    def download_symbol_timeframe(self, symbol: str, timeframe: str, 
                                start_date: datetime, end_date: datetime,
                                resume: bool = True) -> bool:
        """
        Descargar datos para un símbolo y timeframe específico.
        
        Args:
            symbol: Par de trading (ej. BTCUSDT)
            timeframe: Timeframe (ej. 5m, 1h, 1d)
            start_date: Fecha inicio
            end_date: Fecha fin
            resume: Si continuar descarga existente
            
        Returns:
            bool: True si descarga exitosa
        """
        file_path = self.get_file_path(symbol, timeframe)
        
        # Verificar si ya existe y es válido
        if resume and self.file_exists_and_valid(symbol, timeframe):
            logger.info(f"Saltando {symbol} {timeframe} - archivo válido existente")
            return True
        
        logger.info(f"Descargando {symbol} {timeframe}...")
        
        try:
            # Verificar que exchange está configurado
            if not self.exchange:
                logger.error("Exchange no configurado - claves API no válidas")
                return False
            
            # Configurar chunks para evitar saturar RAM
            chunk_size_days = 30  # 30 días por chunk
            current_date = start_date
            all_data = []
            
            with tqdm(desc=f"{symbol} {timeframe}", unit="días") as pbar:
                while current_date < end_date:
                    chunk_end = min(current_date + timedelta(days=chunk_size_days), end_date)
                    
                    # Convertir a timestamps en milisegundos
                    since = int(current_date.timestamp() * 1000)
                    until = int(chunk_end.timestamp() * 1000)
                    
                    # Descargar chunk
                    chunk_data = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=since,
                        limit=1000
                    )
                    
                    if chunk_data:
                        # Filtrar datos dentro del rango temporal
                        filtered_data = [
                            candle for candle in chunk_data 
                            if since <= candle[0] <= until
                        ]
                        all_data.extend(filtered_data)
                        
                        logger.debug(f"Chunk {current_date.date()}: {len(filtered_data)} velas")
                    
                    # Rate limiting
                    time.sleep(0.1)  # 100ms entre requests
                    
                    # Actualizar progreso
                    days_processed = (chunk_end - start_date).days
                    total_days = (end_date - start_date).days
                    pbar.update(chunk_size_days)
                    pbar.set_postfix({
                        'registros': len(all_data),
                        'progreso': f"{days_processed}/{total_days} días"
                    })
                    
                    current_date = chunk_end
            
            # Convertir a DataFrame
            if not all_data:
                logger.error(f"No se obtuvieron datos para {symbol} {timeframe}")
                return False
            
            df = pd.DataFrame(all_data, columns=self.columns[:6])  # OHLCV básico
            
            # Agregar columnas adicionales según especificaciones
            df['close_time'] = df['timestamp'] + self.get_timeframe_ms(timeframe) - 1
            df['quote_asset_volume'] = df['volume'] * df['close']  # Aproximación
            df['number_of_trades'] = 0  # No disponible en API pública
            df['taker_buy_base_asset_volume'] = df['volume'] * 0.5  # Aproximación
            df['taker_buy_quote_asset_volume'] = df['quote_asset_volume'] * 0.5
            
            # Convertir timestamp a datetime legible
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Validar datos antes de guardar
            if not self.validate_downloaded_data(df, symbol, timeframe):
                logger.error(f"Validación falló para {symbol} {timeframe}")
                return False
            
            # Guardar archivo
            df.to_csv(file_path, index=False)
            logger.info(f"SUCCESS: Guardado: {file_path} ({len(df)} registros)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error descargando {symbol} {timeframe}: {e}")
            return False

    def get_timeframe_ms(self, timeframe: str) -> int:
        """Convertir timeframe a milisegundos."""
        timeframe_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return timeframe_ms.get(timeframe, 60 * 1000)

    def validate_downloaded_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """Validar calidad de datos descargados."""
        try:
            # Verificar que no hay gaps mayores a timeframe * 2
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff().dt.total_seconds()
            expected_interval = self.get_timeframe_ms(timeframe) / 1000
            
            large_gaps = time_diffs > (expected_interval * 2)
            if large_gaps.sum() > len(df) * 0.01:  # Máximo 1% de gaps grandes
                logger.warning(f"Demasiados gaps en {symbol} {timeframe}: {large_gaps.sum()}")
                return False
            
            # Verificar que no hay cambios de precio extremos (>50%)
            price_changes = df['close'].pct_change().abs()
            extreme_changes = price_changes > 0.5
            if extreme_changes.sum() > 0:
                logger.warning(f"Cambios de precio extremos en {symbol} {timeframe}: {extreme_changes.sum()}")
                return False
            
            # Verificar que volumen > 0 en 95%+ registros
            zero_volume_pct = (df['volume'] == 0).sum() / len(df)
            if zero_volume_pct > 0.05:
                logger.warning(f"Demasiados registros con volumen cero en {symbol} {timeframe}: {zero_volume_pct:.2%}")
                return False
            
            logger.info(f"SUCCESS: Validacion exitosa: {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Error en validación de {symbol} {timeframe}: {e}")
            return False

    def download_all_data(self, resume: bool = True) -> Dict[str, Dict[str, bool]]:
        """
        Descargar todos los datos según configuración.
        
        Args:
            resume: Si continuar descargas existentes
            
        Returns:
            Dict con resultados de descarga por símbolo/timeframe
        """
        results = {}
        total_combinations = len(self.symbols) * len(self.timeframes)
        completed = 0
        
        logger.info(f"Iniciando descarga de {total_combinations} combinaciones...")
        
        for symbol in self.symbols:
            results[symbol] = {}
            
            for timeframe in self.timeframes:
                logger.info(f"Progreso: {completed + 1}/{total_combinations}")
                
                success = self.download_symbol_timeframe(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    resume=resume
                )
                
                results[symbol][timeframe] = success
                completed += 1
                
                # Pausa entre símbolos para rate limiting
                if success:
                    time.sleep(1)
                else:
                    logger.error(f"ERROR: Fallo descarga: {symbol} {timeframe}")
                    time.sleep(5)  # Pausa más larga en errores
        
        # Reporte final
        self.generate_download_report(results)
        return results

    def generate_download_report(self, results: Dict[str, Dict[str, bool]]) -> None:
        """Generar reporte de descarga."""
        total = 0
        successful = 0
        failed_combinations = []
        
        for symbol, timeframe_results in results.items():
            for timeframe, success in timeframe_results.items():
                total += 1
                if success:
                    successful += 1
                else:
                    failed_combinations.append(f"{symbol} {timeframe}")
        
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        logger.info("="*50)
        logger.info("REPORTE DE DESCARGA")
        logger.info("="*50)
        logger.info(f"Total combinaciones: {total}")
        logger.info(f"Exitosas: {successful}")
        logger.info(f"Fallidas: {total - successful}")
        logger.info(f"Tasa de éxito: {success_rate:.1f}%")
        
        if failed_combinations:
            logger.warning("Combinaciones fallidas:")
            for combo in failed_combinations:
                logger.warning(f"  - {combo}")
        
        logger.info("="*50)

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Descargar datos históricos de Binance')
    parser.add_argument('--symbol', help='Símbolo específico a descargar (ej. BTCUSDT)')
    parser.add_argument('--timeframe', help='Timeframe específico (ej. 5m, 1h)')
    parser.add_argument('--download-all', action='store_true', help='Descargar todos los símbolos y timeframes')
    parser.add_argument('--no-resume', action='store_true', help='No continuar descargas existentes')
    parser.add_argument('--validate-only', action='store_true', help='Solo validar archivos existentes')
    
    args = parser.parse_args()
    
    # Verificar entorno virtual
    if 'nvbot3_env' not in sys.executable:
        logger.error("ERROR: Entorno virtual no esta activo!")
        logger.error("Ejecuta: nvbot3_env\\Scripts\\activate")
        sys.exit(1)
    
    # Verificar variables de entorno
    if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_SECRET_KEY'):
        logger.error("ERROR: Claves API de Binance no configuradas!")
        logger.error("Configura BINANCE_API_KEY y BINANCE_SECRET_KEY en el archivo .env")
        sys.exit(1)
    
    try:
        downloader = HistoricalDataDownloader()
        
        if args.validate_only:
            # Solo validar archivos existentes
            for symbol in downloader.symbols:
                for timeframe in downloader.timeframes:
                    file_path = downloader.get_file_path(symbol, timeframe)
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        is_valid = downloader.validate_downloaded_data(df, symbol, timeframe)
                        status = "SUCCESS" if is_valid else "ERROR"
                        logger.info(f"{status} {symbol} {timeframe}: {len(df)} registros")
            return
        
        if args.symbol and args.timeframe:
            # Descargar símbolo y timeframe específico
            success = downloader.download_symbol_timeframe(
                symbol=args.symbol.upper(),
                timeframe=args.timeframe.lower(),
                start_date=downloader.start_date,
                end_date=downloader.end_date,
                resume=not args.no_resume
            )
            
            if success:
                logger.info("SUCCESS: Descarga completada exitosamente")
            else:
                logger.error("ERROR: Descarga fallo")
                sys.exit(1)
        elif args.download_all:
            # Descargar todos los símbolos y timeframes
            logger.info("Iniciando descarga completa de todos los símbolos y timeframes...")
            results = downloader.download_all_data(resume=not args.no_resume)
            
            # Reporte final
            successful = 0
            total = 0
            for symbol_results in results.values():
                for success in symbol_results.values():
                    total += 1
                    if success:
                        successful += 1
            
            logger.info(f"Descarga completa finalizada: {successful}/{total} exitosas")
            
            if successful < total:
                logger.warning("Algunas descargas fallaron. Revisar logs para detalles.")
                sys.exit(1)
        else:
            # Si no se especifica símbolo/timeframe específico ni --download-all, mostrar ayuda
            logger.error("ERROR: Debes especificar:")
            logger.error("  - Símbolo y timeframe específico: --symbol BTCUSDT --timeframe 5m")
            logger.error("  - Descargar todo: --download-all")
            logger.error("  - Solo validar: --validate-only")
            parser.print_help()
            sys.exit(1)
            
            # Verificar si todas las descargas fueron exitosas
            all_successful = all(
                all(timeframe_results.values()) 
                for timeframe_results in results.values()
            )
            
            if all_successful:
                logger.info("SUCCESS: Todas las descargas completadas exitosamente!")
            else:
                logger.warning("WARNING: Algunas descargas fallaron. Revisa el reporte arriba.")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("Descarga interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
