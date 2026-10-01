"""
File cleaner implementation for removing old files from specified directories.
"""
import time
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from threading import Event, Thread, Lock, current_thread

from pynvr.thread_safe import ThreadSafeSet
from pynvr.utils import make_readable_ts

logger = getLogger("pynvr.cleaner")

class CleanerConfig:
    """Configuration for a file cleaner instance."""
    def __init__(self, folder: Path, filespec: str, age_seconds: int, period: int):
        self.folder: Path = folder
        self.filespec: str = filespec
        self.age_seconds: int = age_seconds
        self.period_seconds: int = period
        self.last_cleanup_time: float = time.time()

class FileCleaner():
    """A file cleaner that runs in a separate thread to delete old files."""
    do_not_delete_set: ThreadSafeSet = ThreadSafeSet()
    min_sleep_seconds = 5
    cleaner_config: dict[CleanerConfig] = {}
    stop_event: Event = None
    thread: Thread = None
    lock: Lock = Lock()

    @staticmethod
    def add(folder: Path, filespec: str, age: timedelta, period: timedelta):
        """
        Add a new folder/filespec to be cleaned up periodically.
        """
        config = CleanerConfig(folder, filespec, age.total_seconds(), period.total_seconds())
        logger.debug(
            f"adding cleaner folder={folder}, filespec={filespec}, age={age}, period={period}"
            )
        with FileCleaner.lock:
            FileCleaner.cleaner_config[str(folder)+filespec] = config
        FileCleaner.min_sleep_seconds = min(
            FileCleaner.min_sleep_seconds,
            period.total_seconds())
        if FileCleaner.thread is None and FileCleaner.stop_event is not None:
            FileCleaner.start()

    @staticmethod
    def start():
        """
        Start the file cleaner thread.
        """
        logger.debug("starting file cleaner thread")
        FileCleaner.thread = Thread(target=FileCleaner._cleanup, daemon=True)
        FileCleaner.thread.start()

    @staticmethod
    def _cleanup():
        current_thread().name = "cleanup_segments"

        while not FileCleaner.stop_event.is_set():
            now = time.time()

            with FileCleaner.lock:
                for config in FileCleaner.cleaner_config.values():
                    # Skip configs that are not due yet
                    if now - config.last_cleanup_time < config.period_seconds:
                        continue

                    cutoff = now - config.age_seconds

                    for file in config.folder.rglob(config.filespec):
                        # Skip protected or non-files
                        if file in FileCleaner.do_not_delete_set or not file.is_file():
                            continue

                        try:
                            stat_entry = file.stat()
                        except Exception:
                            continue  # File disappeared or became inaccessible

                        # Skip files newer than cutoff
                        if stat_entry.st_mtime >= cutoff:
                            continue

                        try:
                            file.unlink()
                            logger.debug(
                                "file cleaner deleted: %s %s", 
                                file, make_readable_ts(stat_entry.st_mtime)
                            )
                        except Exception:
                            pass  # File may have disappeared or been locked

                    config.last_cleanup_time = now

            time.sleep(FileCleaner.min_sleep_seconds)
