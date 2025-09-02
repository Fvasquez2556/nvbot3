#!/usr/bin/env python3
"""
SQLite Connection Pool and Database Manager for NvBot3
=====================================================

Thread-safe connection pooling with proper transaction management,
retry logic, and optimized SQLite configuration for concurrent access.
"""

import sqlite3
import threading
import time
import queue
import logging
from typing import Dict, Any, Optional, Callable, Tuple
from contextlib import contextmanager
from datetime import datetime
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool with retry logic"""
    
    def __init__(self, db_path: str, max_connections: int = 10, timeout: float = 30.0):
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool = queue.Queue(maxsize=max_connections)
        self._pool_lock = threading.RLock()
        self._active_connections = 0
        self._closed = False
        
        # Initialize the pool
        self._create_initial_connections()
        
        # Database operation queue for write operations
        self._operation_queue = queue.Queue()
        self._worker_thread = None
        self._worker_stop_event = threading.Event()
        self._start_worker()
    
    def _create_initial_connections(self):
        """Create initial connections for the pool"""
        for _ in range(min(2, self.max_connections)):  # Start with 2 connections
            try:
                conn = self._create_connection()
                self._pool.put(conn, block=False)
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimal settings"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False  # Allow sharing between threads
        )
        
        # Configure SQLite for concurrent access
        cursor = conn.cursor()
        try:
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            # Set busy timeout
            cursor.execute("PRAGMA busy_timeout=30000")  # 30 seconds
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON")
            # Optimize synchronous mode for better performance
            cursor.execute("PRAGMA synchronous=NORMAL")
            # Set cache size
            cursor.execute("PRAGMA cache_size=10000")
            # Enable automatic vacuum
            cursor.execute("PRAGMA auto_vacuum=INCREMENTAL")
            
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to configure SQLite connection: {e}")
        
        return conn
    
    @contextmanager
    def get_connection(self, readonly: bool = False):
        """Get a connection from the pool with context manager"""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn = None
        try:
            # Try to get existing connection from pool
            try:
                conn = self._pool.get(block=True, timeout=5.0)
            except queue.Empty:
                # Create new connection if pool is empty and under limit
                with self._pool_lock:
                    if self._active_connections < self.max_connections:
                        conn = self._create_connection()
                        self._active_connections += 1
                    else:
                        # Wait for connection to become available
                        conn = self._pool.get(block=True, timeout=self.timeout)
            
            # Test connection and recreate if needed
            try:
                conn.execute("SELECT 1").fetchone()
            except (sqlite3.Error, sqlite3.OperationalError):
                logger.warning("Connection test failed, recreating...")
                try:
                    conn.close()
                except:
                    pass
                conn = self._create_connection()
            
            yield conn
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            # Try to rollback transaction on error
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise
        finally:
            # Return connection to pool
            if conn:
                try:
                    # Make sure we're not in a transaction
                    conn.rollback()
                    self._pool.put(conn, block=False)
                except queue.Full:
                    # Pool is full, close the connection
                    try:
                        conn.close()
                    except:
                        pass
                    with self._pool_lock:
                        self._active_connections -= 1
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    try:
                        conn.close()
                    except:
                        pass
                    with self._pool_lock:
                        self._active_connections -= 1
    
    def execute_transaction(self, operations: list, readonly: bool = False) -> Any:
        """Execute multiple operations in a single transaction"""
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                with self.get_connection(readonly=readonly) as conn:
                    cursor = conn.cursor()
                    
                    if not readonly:
                        cursor.execute("BEGIN IMMEDIATE")
                    
                    results = []
                    try:
                        for operation in operations:
                            if isinstance(operation, tuple):
                                query, params = operation
                                result = cursor.execute(query, params)
                                query_str = query
                            else:
                                result = cursor.execute(operation)
                                query_str = operation
                            
                            # Collect results for SELECT statements
                            if query_str.strip().upper().startswith('SELECT'):
                                results.append(result.fetchall())
                            else:
                                results.append(result.rowcount)
                        
                        if not readonly:
                            conn.commit()
                        
                        return results[0] if len(results) == 1 else results
                        
                    except Exception as e:
                        if not readonly:
                            conn.rollback()
                        raise e
                        
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying in {retry_delay}s (attempt {attempt + 1})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(f"Database error after {attempt + 1} attempts: {e}")
                    raise e
            except Exception as e:
                logger.error(f"Unexpected error in transaction: {e}")
                raise e
        
        raise RuntimeError(f"Failed to execute transaction after {max_retries} attempts")
    
    def _start_worker(self):
        """Start the database worker thread"""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Database worker thread started")
    
    def _worker_loop(self):
        """Main loop for the database worker thread"""
        while not self._worker_stop_event.is_set():
            try:
                # Get operation from queue with timeout
                try:
                    operation = self._operation_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if operation is None:  # Shutdown signal
                    break
                
                # Execute the operation
                try:
                    func, args, kwargs, result_queue = operation
                    result = func(*args, **kwargs)
                    if result_queue:
                        result_queue.put(('success', result))
                except Exception as e:
                    logger.error(f"Worker thread operation failed: {e}")
                    if result_queue:
                        result_queue.put(('error', e))
                
                self._operation_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
    
    def queue_operation(self, func: Callable, *args, **kwargs) -> Any:
        """Queue a database operation for execution by the worker thread"""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        result_queue = queue.Queue()
        operation = (func, args, kwargs, result_queue)
        
        self._operation_queue.put(operation)
        
        # Wait for result
        try:
            status, result = result_queue.get(timeout=30.0)
            if status == 'success':
                return result
            else:
                raise result
        except queue.Empty:
            raise TimeoutError("Database operation timed out")
    
    def queue_operation_async(self, func: Callable, *args, **kwargs):
        """Queue a database operation asynchronously (fire-and-forget)"""
        if self._closed:
            return
        
        operation = (func, args, kwargs, None)
        try:
            self._operation_queue.put(operation, block=False)
        except queue.Full:
            logger.warning("Operation queue is full, dropping operation")
    
    def close(self):
        """Close all connections and shutdown the pool"""
        if self._closed:
            return
        
        logger.info("Closing database connection pool...")
        self._closed = True
        
        # Stop worker thread
        self._worker_stop_event.set()
        self._operation_queue.put(None)  # Shutdown signal
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        
        # Close all connections in pool
        while not self._pool.empty():
            try:
                conn = self._pool.get(block=False)
                conn.close()
            except (queue.Empty, Exception):
                break
        
        logger.info("Database connection pool closed")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.close()
        except:
            pass

# Singleton pattern for global database pool
_db_pool = None
_pool_lock = threading.Lock()

def get_db_pool(db_path: str = None) -> SQLiteConnectionPool:
    """Get the global database pool instance"""
    global _db_pool
    
    if _db_pool is None:
        with _pool_lock:
            if _db_pool is None:
                if db_path is None:
                    # Default path
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    db_path = os.path.join(current_dir, "signals.db")
                
                _db_pool = SQLiteConnectionPool(db_path)
                logger.info(f"Initialized database pool: {db_path}")
    
    return _db_pool

def close_db_pool():
    """Close the global database pool"""
    global _db_pool
    
    if _db_pool:
        with _pool_lock:
            if _db_pool:
                _db_pool.close()
                _db_pool = None

# Decorator for database operations with retry logic
def db_operation(readonly=False, max_retries=3):
    """Decorator for database operations with automatic retry logic"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            pool = get_db_pool()
            retry_delay = 0.1
            
            for attempt in range(max_retries):
                try:
                    if readonly:
                        with pool.get_connection(readonly=True) as conn:
                            return func(conn, *args, **kwargs)
                    else:
                        # Use queued operation for write operations
                        return pool.queue_operation(func, *args, **kwargs)
                        
                except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                    if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                        logger.warning(f"Database locked, retrying {func.__name__} in {retry_delay}s")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        logger.error(f"Database error in {func.__name__}: {e}")
                        raise e
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {e}")
                    raise e
            
            raise RuntimeError(f"Failed to execute {func.__name__} after {max_retries} attempts")
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test the connection pool
    print("Testing SQLite Connection Pool...")
    
    # Initialize pool
    pool = get_db_pool()
    
    # Test transaction
    try:
        operations = [
            "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, data TEXT)",
            ("INSERT INTO test_table (data) VALUES (?)", ("test data",)),
            "SELECT * FROM test_table"
        ]
        
        result = pool.execute_transaction(operations)
        print(f"Transaction result: {result}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        close_db_pool()
    
    print("Test completed!")