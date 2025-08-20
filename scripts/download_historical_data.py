#!/usr/bin/env python3
"""
NvBot3 - Historical Data Downloader
===================================

Módulo para descargar datos históricos de Binance para entrenamiento de modelos.
Implementa descarga por chunks, rate limiting, progress tracking y manejo robusto de errores.

Autor: NvBot3 Team
Fecha: Agosto 2025
"""

import ccxt
import pandas as pd
import numpy as np
import time
import os
import sys
import logging
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import asyncio
from dotenv import load_dotenv


class HistoricalDataDownloader:
    """
    Clase para descargar datos históricos de Binance con manejo robusto de errores
    y optimización para recursos limitados de laptop.
    """
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """
        Inicializar el descargador de datos históricos.
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        # Verificar entorno virtual
        self._verify_virtual_environment()
        
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Setup logging
        self._setup_logging()
        
        # Cargar variables de entorno
        load_dotenv()
        
        # Inicializar conexión Binance
        self.exchange = self._initialize_binance_connection()
        
        # Configurar rate limiting
        self.rate_limiter = RateLimiter(
            max_requests=self.config['api']['binance']['rate_limit_per_minute'],
            time_window=60  # 1 minuto
        )
        
        # Setup directorios
        self._ensure_directories()
        
        # Configurar parámetros de descarga
        self.chunk_size_days = self.config['api']['binance']['chunk_size_days']
        self.max_retries = self.config['api']['binance']['max_retries']
        self.retry_delay = self.config['api']['binance']['retry_delay']
        self.backoff_factor = self.config['api']['binance']['backoff_factor']
        
        self.logger.info("HistoricalDataDownloader inicializado correctamente")
    
    def _verify_virtual_environment(self):
        """Verificar que el entorno virtual nvbot3_env está activo."""
        if 'nvbot3_env' not in sys.executable:
            print("❌ ERROR: Entorno virtual nvbot3_env no está activo!")
            print("Por favor ejecuta: nvbot3_env\\Scripts\\activate")
            sys.exit(1)
        print("✅ Entorno virtual nvbot3_env activo")
    
    def _load_config(self, config_path: str) -> Dict:
        """Cargar configuración desde archivo YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error al parsear configuración YAML: {e}")
    
    def _setup_logging(self):
        """Configurar sistema de logging."""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/download_log.txt'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_binance_connection(self) -> ccxt.binance:
        """Inicializar conexión con Binance."""
        try:
            exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY', ''),
                'secret': os.getenv('BINANCE_SECRET_KEY', ''),
                'sandbox': False,  # Usar producción para datos históricos
                'rateLimit': 50,   # Rate limit conservativo
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'  # Spot trading
                }
            })
            
            # Verificar conectividad
            exchange.load_markets()
            self.logger.info("Conexión con Binance establecida correctamente")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Error conectando con Binance: {e}")
            # Para datos históricos, podemos continuar sin API keys
            exchange = ccxt.binance({
                'sandbox': False,
                'rateLimit': 100,
                'enableRateLimit': True
            })
            exchange.load_markets()
            return exchange
    
    def _ensure_directories(self):
        """Crear directorios necesarios si no existen."""
        directories = [
            'data/raw',
            'data/processed', 
            'data/models',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Directorios verificados/creados")
    
    def download_symbol_timeframe(self, symbol: str, timeframe: str, 
                                start_date: str, end_date: Optional[str] = None) -> bool:
        """
        Descargar datos históricos para un símbolo y timeframe específico.
        
        Args:
            symbol: Símbolo de trading (ej: 'BTCUSDT')
            timeframe: Timeframe (ej: '5m', '1h', '1d')
            start_date: Fecha de inicio (formato: 'YYYY-MM-DD')
            end_date: Fecha de fin (opcional, default: hoy)
            
        Returns:
            bool: True si la descarga fue exitosa
        """
        try:
            # Verificar si el símbolo existe en Binance
            if symbol not in self.exchange.markets:
                self.logger.error(f"Símbolo {symbol} no encontrado en Binance")
                return False
            
            # Configurar fechas
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.now().timestamp() * 1000) if end_date is None else \
                     int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Verificar si ya existe archivo y obtener última fecha
            file_path = f"data/raw/{symbol}_{timeframe}.csv"
            start_ts = self._get_resume_timestamp(file_path, start_ts)
            
            if start_ts >= end_ts:
                self.logger.info(f"{symbol} {timeframe}: Ya está actualizado")
                return True
            
            # Calcular chunks para descarga
            chunks = self._calculate_chunks(start_ts, end_ts, timeframe)
            
            self.logger.info(f"Descargando {symbol} {timeframe}: {len(chunks)} chunks")
            
            all_data = []
            
            # Progress bar para chunks
            chunk_pbar = tqdm(chunks, desc=f"{symbol} {timeframe}", 
                            unit="chunk", leave=True)
            
            for chunk_start, chunk_end in chunk_pbar:
                chunk_data = self._download_chunk_with_retry(
                    symbol, timeframe, chunk_start, chunk_end
                )
                
                if chunk_data is not None and len(chunk_data) > 0:
                    all_data.extend(chunk_data)
                    
                    # Actualizar progress bar con info útil
                    chunk_pbar.set_postfix({
                        'records': len(chunk_data),
                        'total': len(all_data)
                    })
                
                # Rate limiting
                self.rate_limiter.wait_if_needed()
            
            chunk_pbar.close()
            
            if len(all_data) == 0:
                self.logger.warning(f"No se obtuvieron datos para {symbol} {timeframe}")
                return False
            
            # Convertir a DataFrame y guardar
            success = self._save_data_to_csv(all_data, file_path, symbol, timeframe)
            
            if success:
                self.logger.info(f"✅ {symbol} {timeframe}: {len(all_data)} registros guardados")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error descargando {symbol} {timeframe}: {e}")
            return False
    
    def _get_resume_timestamp(self, file_path: str, default_start: int) -> int:
        """
        Obtener timestamp para reanudar descarga desde archivo existente.
        
        Args:
            file_path: Ruta del archivo CSV
            default_start: Timestamp de inicio por defecto
            
        Returns:
            int: Timestamp desde donde continuar descarga
        """
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    last_timestamp = df['timestamp'].max()
                    self.logger.info(f"Resumiendo desde: {datetime.fromtimestamp(last_timestamp/1000)}")
                    return int(last_timestamp)
            
            return default_start
            
        except Exception as e:
            self.logger.warning(f"Error leyendo archivo existente {file_path}: {e}")
            return default_start
    
    def _calculate_chunks(self, start_ts: int, end_ts: int, timeframe: str) -> List[Tuple[int, int]]:
        """
        Calcular chunks para descarga eficiente.
        
        Args:
            start_ts: Timestamp de inicio
            end_ts: Timestamp de fin
            timeframe: Timeframe para calcular chunk size apropiado
            
        Returns:
            List[Tuple[int, int]]: Lista de (start, end) timestamps para cada chunk
        """
        chunks = []
        
        # Calcular tamaño de chunk en base al timeframe
        chunk_ms = self.chunk_size_days * 24 * 60 * 60 * 1000  # días a millisegundos
        
        # Ajustar chunk size basado en timeframe para evitar límites de API
        timeframe_limits = {
            '1m': 7,    # 7 días máximo
            '5m': 15,   # 15 días máximo 
            '15m': 30,  # 30 días máximo
            '1h': 60,   # 60 días máximo
            '4h': 120,  # 120 días máximo
            '1d': 365   # 1 año máximo
        }
        
        if timeframe in timeframe_limits:
            max_days = timeframe_limits[timeframe]
            chunk_ms = min(chunk_ms, max_days * 24 * 60 * 60 * 1000)
        
        current_ts = start_ts
        while current_ts < end_ts:
            chunk_end = min(current_ts + chunk_ms, end_ts)
            chunks.append((current_ts, chunk_end))
            current_ts = chunk_end
        
        return chunks
    
    def _download_chunk_with_retry(self, symbol: str, timeframe: str, 
                                 start_ts: int, end_ts: int) -> Optional[List]:
        """
        Descargar chunk con reintentos automáticos.
        
        Args:
            symbol: Símbolo de trading
            timeframe: Timeframe 
            start_ts: Timestamp de inicio del chunk
            end_ts: Timestamp de fin del chunk
            
        Returns:
            Optional[List]: Datos del chunk o None si falla
        """
        for attempt in range(self.max_retries):
            try:
                # Fetch OHLCV data desde Binance
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    since=start_ts,
                    limit=1000  # Límite de Binance
                )
                
                # Filtrar datos que están en el rango correcto
                filtered_data = [
                    candle for candle in ohlcv 
                    if start_ts <= candle[0] <= end_ts
                ]
                
                return filtered_data
                
            except Exception as e:
                wait_time = self.retry_delay * (self.backoff_factor ** attempt)
                
                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"Intento {attempt + 1} falló para {symbol} {timeframe}: {e}. "
                        f"Reintentando en {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Falló descarga después de {self.max_retries} intentos: {e}"
                    )
        
        return None
    
    def _save_data_to_csv(self, data: List, file_path: str, 
                         symbol: str, timeframe: str) -> bool:
        """
        Guardar datos en formato CSV con validación.
        
        Args:
            data: Lista de datos OHLCV
            file_path: Ruta donde guardar el archivo
            symbol: Símbolo para logging
            timeframe: Timeframe para logging
            
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            # Definir columnas según especificación
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
            ]
            
            # Convertir a DataFrame
            df = pd.DataFrame(data, columns=columns[:len(data[0])])
            
            # Completar columnas faltantes con NaN si es necesario
            for col in columns:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Validación de datos
            validation_result = self._validate_data(df, symbol, timeframe)
            
            if not validation_result['valid']:
                self.logger.warning(f"Datos con problemas para {symbol} {timeframe}: {validation_result['issues']}")
            
            # Eliminar duplicados y ordenar por timestamp
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Combinar con datos existentes si el archivo ya existe
            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path)
                df = pd.concat([existing_df, df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Guardar archivo
            df.to_csv(file_path, index=False)
            
            self.logger.info(f"Datos guardados: {file_path} ({len(df)} registros)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando datos en {file_path}: {e}")
            return False
    
    def _validate_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """
        Validar calidad de datos descargados.
        
        Args:
            df: DataFrame con datos
            symbol: Símbolo para contexto
            timeframe: Timeframe para contexto
            
        Returns:
            Dict: Resultado de validación con detalles
        """
        issues = []
        
        # Verificar que no esté vacío
        if len(df) == 0:
            issues.append("DataFrame vacío")
            return {'valid': False, 'issues': issues}
        
        # Verificar duplicados
        duplicates = df.duplicated(subset=['timestamp']).sum()
        if duplicates > 0:
            issues.append(f"{duplicates} timestamps duplicados")
        
        # Verificar gaps en timestamp
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff()
        
        # Calcular timeframe en millisegundos
        timeframe_ms = self._timeframe_to_ms(timeframe)
        max_gap = timeframe_ms * self.config['download']['max_gap_multiplier']
        
        large_gaps = (time_diffs > max_gap).sum()
        if large_gaps > 0:
            issues.append(f"{large_gaps} gaps grandes en timestamps")
        
        # Verificar valores de precio válidos
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                invalid_prices = (df[col] <= 0).sum()
                if invalid_prices > 0:
                    issues.append(f"{invalid_prices} precios inválidos en {col}")
        
        # Verificar volumen
        if 'volume' in df.columns:
            zero_volume = (df['volume'] <= 0).sum()
            volume_ratio = zero_volume / len(df)
            if volume_ratio > 0.05:  # Más del 5% con volumen cero
                issues.append(f"{volume_ratio:.1%} de registros con volumen cero")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_records': len(df),
            'duplicates': duplicates,
            'large_gaps': large_gaps
        }
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convertir timeframe string a millisegundos."""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return timeframe_map.get(timeframe, 60 * 1000)
    
    def download_all_data(self) -> Dict:
        """
        Descargar todos los datos según configuración.
        
        Returns:
            Dict: Resumen de resultados de descarga
        """
        symbols = self.config['data']['symbols']
        timeframes = self.config['data']['timeframes'] 
        start_date = self.config['data']['start_date']
        
        total_combinations = len(symbols) * len(timeframes)
        successful_downloads = 0
        failed_downloads = []
        
        self.logger.info(f"Iniciando descarga de {total_combinations} combinaciones símbolo-timeframe")
        
        # Progress bar principal
        main_pbar = tqdm(total=total_combinations, desc="Descarga General", 
                        unit="combo", position=0)
        
        try:
            for symbol in symbols:
                for timeframe in timeframes:
                    main_pbar.set_description(f"Descargando {symbol} {timeframe}")
                    
                    success = self.download_symbol_timeframe(symbol, timeframe, start_date)
                    
                    if success:
                        successful_downloads += 1
                    else:
                        failed_downloads.append(f"{symbol}_{timeframe}")
                    
                    main_pbar.update(1)
                    main_pbar.set_postfix({
                        'exitosos': successful_downloads,
                        'fallidos': len(failed_downloads)
                    })
                    
                    # Pequeña pausa entre descargas para ser gentil con la API
                    time.sleep(0.5)
            
        finally:
            main_pbar.close()
        
        # Resumen final
        result = {
            'total_combinations': total_combinations,
            'successful_downloads': successful_downloads,
            'failed_downloads': failed_downloads,
            'success_rate': successful_downloads / total_combinations if total_combinations > 0 else 0
        }
        
        self.logger.info(f"Descarga completada: {successful_downloads}/{total_combinations} exitosos")
        if failed_downloads:
            self.logger.warning(f"Descargas fallidas: {failed_downloads}")
        
        return result


class RateLimiter:
    """Clase para manejar rate limiting de API."""
    
    def __init__(self, max_requests: int, time_window: int):
        """
        Inicializar rate limiter.
        
        Args:
            max_requests: Máximo número de requests permitidos
            time_window: Ventana de tiempo en segundos
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def wait_if_needed(self):
        """Esperar si se ha alcanzado el límite de rate."""
        now = time.time()
        
        # Limpiar requests antiguos
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        # Verificar si necesitamos esperar
        if len(self.requests) >= self.max_requests:
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request) + 0.1  # +0.1 por seguridad
            
            if wait_time > 0:
                time.sleep(wait_time)
        
        # Registrar nueva request
        self.requests.append(now)


def main():
    """Función principal para ejecutar el descargador."""
    print("🤖 NvBot3 - Historical Data Downloader")
    print("=" * 50)
    
    try:
        # Verificar entorno virtual antes de comenzar
        if 'nvbot3_env' not in sys.executable:
            print("❌ ERROR: Entorno virtual nvbot3_env no está activo!")
            print("Por favor ejecuta: nvbot3_env\\Scripts\\activate")
            return False
        
        # Inicializar descargador
        downloader = HistoricalDataDownloader()
        
        # Ejecutar descarga completa
        results = downloader.download_all_data()
        
        # Mostrar resumen
        print("\n📊 Resumen de Descarga:")
        print(f"✅ Exitosos: {results['successful_downloads']}")
        print(f"❌ Fallidos: {len(results['failed_downloads'])}")
        print(f"📈 Tasa de éxito: {results['success_rate']:.1%}")
        
        if results['failed_downloads']:
            print(f"\n⚠️  Descargas fallidas: {', '.join(results['failed_downloads'])}")
        
        return results['success_rate'] > 0.8  # Considerar exitoso si >80% funciona
        
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
