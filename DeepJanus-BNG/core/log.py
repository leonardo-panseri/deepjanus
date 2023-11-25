import logging
import os

from core.ini_file import IniFile


class LogSetup:
    """Class to configure logging for a DeepJanus instance"""

    def __init__(self):
        self._all_loggers: set[logging.Logger] = set()
        self._log_ini: IniFile | None = None

    def use_ini(self, ini_path):
        """Set the INI file from which to load the logging configurations."""
        self._log_ini = IniFile(ini_path)
        fmt = self._log_ini.get_option_or_create('config', 'format',
                                                 '[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s')
        date_fmt = self._log_ini.get_option_or_create('config', 'date_format', '%H:%M:%S')
        logging.basicConfig(format=fmt, datefmt=date_fmt)
        for logger in self._all_loggers:
            self._setup_log_level(logger)

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
        logger_name = os.path.basename(logger_name_path)
        log = logging.getLogger(logger_name)
        self._all_loggers.add(log)
        self._setup_log_level(log)
        return log


log_setup = LogSetup()


def get_logger(logger_name_path):
    """
    Get a new logger for a Python file.
    :param logger_name_path: the path of the Python file that will use this logger (__file__)
    :return: the created logger
    """
    return log_setup.get_logger(logger_name_path)
