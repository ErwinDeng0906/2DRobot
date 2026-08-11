"""
日志工具模块

提供统一的日志功能
"""

from __future__ import annotations

import logging
import locale
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


# 日志目录
_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)


def get_log_directory() -> Path:
    """获取日志目录"""
    return _LOG_DIR


class _SafeConsoleStream:
    """Text stream wrapper that replaces characters the console cannot encode."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self.encoding = getattr(stream, "encoding", None) or locale.getpreferredencoding(False)

    def write(self, text: str) -> int:
        try:
            return self._stream.write(text)
        except UnicodeEncodeError:
            encoding = self.encoding or "utf-8"
            safe_text = text.encode(encoding, errors="replace").decode(
                encoding,
                errors="replace",
            )
            return self._stream.write(safe_text)

    def flush(self) -> None:
        return self._stream.flush()


def _safe_console_stream(stream):
    try:
        stream.reconfigure(errors="replace")
        return stream
    except Exception:
        return _SafeConsoleStream(stream)


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    retention_days: int = 30,
) -> None:
    """
    设置日志系统
    
    Args:
        level: 日志级别
        log_dir: 日志目录，如果不指定则使用默认目录
        retention_days: 日志保留天数
    """
    if log_dir is None:
        log_dir = _LOG_DIR
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理旧日志
    cleanup_old_logs(log_dir, retention_days)
    
    # 日志文件名（按日期）
    today = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"system_{today}.log"
    
    # 配置日志格式
    log_format = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 控制台处理器
    console_handler = logging.StreamHandler(_safe_console_stream(sys.stdout))
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 清除可能存在的重复处理器
    root_logger.handlers = [file_handler, console_handler]


def cleanup_old_logs(log_dir: Optional[Path] = None, retention_days: int = 30) -> None:
    """
    清理旧日志文件
    
    Args:
        log_dir: 日志目录
        retention_days: 保留天数
    """
    if log_dir is None:
        log_dir = _LOG_DIR
    
    if not log_dir.exists():
        return
    
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    
    for log_file in log_dir.glob("*.log"):
        try:
            # 从文件名提取日期（格式：system_YYYYMMDD.log）
            if log_file.stem.startswith("system_"):
                date_str = log_file.stem.replace("system_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date < cutoff_date:
                    log_file.unlink()
                    logging.info(f"已删除旧日志文件: {log_file.name}")
        except (ValueError, OSError) as e:
            logging.warning(f"处理日志文件 {log_file.name} 时出错: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称（通常是模块名）
    
    Returns:
        日志记录器
    """
    return logging.getLogger(name)

