import logging
import os
from pathlib import Path

from core.ini_file import IniFile


class LogSetup:
    """Class to configure logging for a DeepJanus instance"""

    def __init__(self):
        self.base_logger = logging.getLogger("deepjanus")
        self._all_loggers: set[logging.Logger] = set()
        self._log_ini: IniFile | None = None

    def use_ini(self, ini_path):
        """Set the INI file from which to load the logging configurations."""
        self._log_ini = IniFile(ini_path)

        fmt = (self._log_ini
               .get_option_or_create('config', 'format',
                                     r'[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s'))
        date_fmt = self._log_ini.get_option_or_create('config', 'date_format', '%H:%M:%S')

        # If package 'rich' is installed, use its formatter
        try:
            from rich.logging import RichHandler
            terminal_handler = RichHandler()
            self.base_logger.addHandler(terminal_handler)
        except ModuleNotFoundError:
            terminal_handler = logging.StreamHandler()
            terminal_handler.setFormatter(logging.Formatter(fmt, date_fmt))
            self.base_logger.addHandler(terminal_handler)

        for logger in self._all_loggers:
            self._setup_log_level(logger)

    def setup_log_file(self, file_path: Path):
        """Configure DeepJanus logger to save logs to file."""
        file_handler = logging.FileHandler(file_path, 'w')
        file_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', '%H:%M:%S'))
        self.base_logger.addHandler(file_handler)

    def _setup_log_level(self, logger: logging.Logger):
        if self._log_ini:
            level = self._log_ini.get_option_or_create('log_levels', logger.name, 'INFO')
        else:
            level = 'INFO'
        logger.setLevel(level)

    def get_logger(self, logger_name_path):
        """
        Get a new logger for a Python file.
        :param logger_name_path: the path of the Python file that will use this logger (__file__)
        :return: the created logger
        """
        logger_name = os.path.basename(logger_name_path).replace(".py", "")
        log = logging.getLogger(f'deepjanus.{logger_name}')
        self._all_loggers.add(log)
        self._setup_log_level(log)
        return log


log_setup = LogSetup()


def get_logger(logger_name_path: str):
    """
    Get a new logger for a Python file.
    :param logger_name_path: the path of the Python file that will use this logger (__file__)
    :return: the created logger
    """
    return log_setup.get_logger(logger_name_path)
