# -*- coding: utf-8 -*-
"""日志模块 - 处理 LLM 切换、回退、API Key 变更等事件的持久化日志"""

import json
import datetime
from pathlib import Path
from typing import Union


class LogConfig:
    """日志配置常量"""
    LOG_MAX_DIR_SIZE_MB = 100  # 日志目录最大总大小（MB），超过将删除最旧文件
    LOG_RETENTION_DAYS = 14  # 保留日志的天数，超过天数的文件将被删除
    LOG_COMPRESS_DAYS = 7  # 超过此天数的日志文件会被 gzip 压缩


def write_llm_log(event: str, model: str, detail: str, log_dir: Union[str, Path] = None, skip_retention: bool = False):
    """写入 LLM 切换/回退相关日志到 logs/llm_switch_YYYY-MM-DD.log 和 logs/llm_switch.log（JSONL 格式）
    
    参数：
        event: 事件类型（如 api_key_change, user_switch, fallback 等）
        model: 相关模型（如 deepseek, glm, local 等）
        detail: 详细信息
        log_dir: 日志目录路径，默认为项目根目录的 logs/
    """
    # 兼容接受 str 或 Path 类型参数
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / 'logs'
    else:
        log_dir = Path(log_dir)

    # 如果目录不存在则创建
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # 无法创建日志目录则退回到当前文件夹下的 logs
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # 按日期分片日志名，例如 llm_switch_2025-11-29.log
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    dated_log_file = log_dir / f'llm_switch_{date_str}.log'
    # 兼容旧文件名，保留最新汇总文件
    latest_log_file = log_dir / 'llm_switch.log'
    timestamp = datetime.datetime.now().isoformat()
    
    # 构造 JSONL 格式：每行一个 JSON 对象
    log_record = {
        "timestamp": timestamp,
        "event": event,
        "model": model,
        "detail": detail
    }
    line = json.dumps(log_record, ensure_ascii=False) + "\n"
    
    try:
        # 写入当天分片
        with open(dated_log_file, 'a', encoding='utf-8') as f:
            f.write(line)
        # 也写入 latest 汇总文件（追加）
        with open(latest_log_file, 'a', encoding='utf-8') as f2:
            f2.write(line)
        # 写入后执行保留策略（限制目录总大小并删除过旧文件）
        try:
            if not skip_retention:
                enforce_log_retention(log_dir, max_total_mb=LogConfig.LOG_MAX_DIR_SIZE_MB, keep_days=LogConfig.LOG_RETENTION_DAYS)
        except Exception as e_ret:
            print(f"[write_llm_log] 执行日志保留策略失败: {e_ret}")
    except Exception as e:
        print(f"[write_llm_log] 日志写入失败: {e}")


def enforce_log_retention(log_dir: Union[str, Path], max_total_mb: int = 100, keep_days: int = 14):
    """确保日志目录总大小不超过 `max_total_mb`，并删除早于 `keep_days` 的日志文件，超过 7 天的文件压缩。

    策略：
    - 先压缩超过 LOG_COMPRESS_DAYS 的日志文件（如果配置了）；
    - 删除修改时间早于 `keep_days` 的文件；
    - 如果目录总大小仍然大于限制，则按修改时间从旧到新删除文件直到满足限制。
    """
    # 计算阈值
    max_bytes = max_total_mb * 1024 * 1024
    now = datetime.datetime.now()

    # 兼容 str 或 Path，并容错目录不存在
    log_dir = Path(log_dir)
    if not log_dir.exists() or not log_dir.is_dir():
        return

    # 收集日志文件（仅文件）
    files = [p for p in log_dir.iterdir() if p.is_file()]

    # 先压缩超过 LOG_COMPRESS_DAYS 的日志文件（如果配置了）
    compress_days = LogConfig.LOG_COMPRESS_DAYS
    if compress_days and compress_days > 0:
        compress_cutoff = now - datetime.timedelta(days=compress_days)
        for p in files:
            try:
                if p.suffix == '.gz':
                    continue
                mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
                if mtime < compress_cutoff:
                    # gzip 压缩
                    try:
                        import gzip
                        with open(p, 'rb') as f_in, gzip.open(str(p) + '.gz', 'wb') as f_out:
                            f_out.writelines(f_in)
                        p.unlink()
                        # 由于 write_llm_log 会触发保留策略，避免递归调用，使用 skip_retention=True
                        write_llm_log('compress', p.name, f'compressed to {p.name}.gz', log_dir, skip_retention=True)
                    except Exception as e:
                        print(f"[enforce_log_retention] 压缩日志失败 {p}: {e}")
            except Exception as e:
                print(f"[enforce_log_retention] 检查并压缩日志失败 {p}: {e}")

    # 重新收集并按修改时间排序（最旧优先）
    files = sorted([p for p in log_dir.iterdir() if p.is_file()], key=lambda p: p.stat().st_mtime)

    # 删除超过保留天数的文件
    if keep_days is not None and keep_days > 0:
        cutoff = now - datetime.timedelta(days=keep_days)
        for p in list(files):
            try:
                mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
                if mtime < cutoff:
                    p.unlink()
                    write_llm_log('delete_old', p.name, f'deleted older than {keep_days} days', log_dir, skip_retention=True)
            except Exception as e:
                print(f"[enforce_log_retention] 删除过旧日志失败 {p}: {e}")

    # 重新收集并按修改时间排序（最旧优先）
    files = sorted([p for p in log_dir.iterdir() if p.is_file()], key=lambda p: p.stat().st_mtime)

    # 计算当前总大小（包括压缩文件）
    total = sum(p.stat().st_size for p in files)
    if total <= max_bytes:
        return

    # 逐个删除最旧文件直到总大小 <= 限制
    for p in files:
        try:
            size = p.stat().st_size
            p.unlink()
            write_llm_log('delete_oldest', p.name, f'deleted to enforce max size, freed {size} bytes', log_dir, skip_retention=True)
            total -= size
            if total <= max_bytes:
                break
        except Exception as e:
            print(f"[enforce_log_retention] 删除日志文件失败 {p}: {e}")
