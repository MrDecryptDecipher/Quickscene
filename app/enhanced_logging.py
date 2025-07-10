#!/usr/bin/env python3
"""
Enhanced Logging and Error Handling for Quickscene

Provides comprehensive logging, error tracking, and performance monitoring
for production deployment.
"""

import logging
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import functools
import time

class QuicksceneLogger:
    """Enhanced logger with structured logging and error tracking"""
    
    def __init__(self, name: str, log_file: Optional[str] = None, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self.file_formatter)
            self.logger.addHandler(file_handler)
        
        # Error tracking
        self.error_counts = {}
        self.performance_metrics = {}
    
    def log_structured(self, level: str, message: str, **kwargs):
        """Log with structured data"""
        extra_data = {
            'timestamp': datetime.now().isoformat(),
            'component': kwargs.get('component', 'unknown'),
            'operation': kwargs.get('operation', 'unknown'),
            **kwargs
        }
        
        log_message = f"{message} | {json.dumps(extra_data)}"
        getattr(self.logger, level.lower())(log_message)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context and tracking"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Track error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        error_data = {
            'error_type': error_type,
            'error_message': error_message,
            'error_count': self.error_counts[error_type],
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.log_structured('error', f"Error occurred: {error_message}", **error_data)
    
    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics"""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = {
                'count': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0
            }
        
        metrics = self.performance_metrics[operation]
        metrics['count'] += 1
        metrics['total_time'] += duration_ms
        metrics['min_time'] = min(metrics['min_time'], duration_ms)
        metrics['max_time'] = max(metrics['max_time'], duration_ms)
        metrics['avg_time'] = metrics['total_time'] / metrics['count']
        
        perf_data = {
            'operation': operation,
            'duration_ms': duration_ms,
            'avg_duration_ms': metrics['avg_time'],
            'min_duration_ms': metrics['min_time'],
            'max_duration_ms': metrics['max_time'],
            'operation_count': metrics['count'],
            **kwargs
        }
        
        self.log_structured('info', f"Performance: {operation} completed", **perf_data)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        total_errors = sum(self.error_counts.values())
        return {
            'total_errors': total_errors,
            'error_types': dict(self.error_counts),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        return dict(self.performance_metrics)

def performance_monitor(operation_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log performance if logger is available
                if hasattr(func, '__self__') and hasattr(func.__self__, 'logger'):
                    func.__self__.logger.log_performance(op_name, duration_ms)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                # Log error if logger is available
                if hasattr(func, '__self__') and hasattr(func.__self__, 'logger'):
                    func.__self__.logger.log_error(e, {
                        'operation': op_name,
                        'duration_ms': duration_ms,
                        'args': str(args)[:200],  # Truncate for safety
                        'kwargs': str(kwargs)[:200]
                    })
                
                raise
        
        return wrapper
    return decorator

def error_handler(default_return=None, reraise=True):
    """Decorator for comprehensive error handling"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error if logger is available
                if hasattr(func, '__self__') and hasattr(func.__self__, 'logger'):
                    func.__self__.logger.log_error(e, {
                        'function': f"{func.__module__}.{func.__name__}",
                        'args': str(args)[:200],
                        'kwargs': str(kwargs)[:200]
                    })
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self, logger: QuicksceneLogger):
        self.logger = logger
        self.health_checks = {}
    
    def register_check(self, name: str, check_func, critical: bool = False):
        """Register a health check function"""
        self.health_checks[name] = {
            'func': check_func,
            'critical': critical,
            'last_result': None,
            'last_check': None
        }
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        critical_failures = 0
        
        for name, check in self.health_checks.items():
            try:
                start_time = time.time()
                result = check['func']()
                duration_ms = (time.time() - start_time) * 1000
                
                check_result = {
                    'status': 'pass' if result else 'fail',
                    'duration_ms': duration_ms,
                    'timestamp': datetime.now().isoformat()
                }
                
                if not result and check['critical']:
                    critical_failures += 1
                
                check['last_result'] = result
                check['last_check'] = datetime.now()
                
                results['checks'][name] = check_result
                
            except Exception as e:
                self.logger.log_error(e, {'health_check': name})
                
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                
                if check['critical']:
                    critical_failures += 1
        
        if critical_failures > 0:
            results['status'] = 'unhealthy'
        elif any(check['status'] in ['fail', 'error'] for check in results['checks'].values()):
            results['status'] = 'degraded'
        
        return results

# Global logger instance
_global_logger = None

def get_logger(name: str = "quickscene", **kwargs) -> QuicksceneLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = QuicksceneLogger(name, **kwargs)
    return _global_logger

def setup_production_logging(log_file: str = "logs/quickscene.log", level: str = "INFO"):
    """Setup production logging configuration"""
    global _global_logger
    _global_logger = QuicksceneLogger("quickscene", log_file, level)
    
    # Configure root logger to prevent duplicate logs
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    return _global_logger
