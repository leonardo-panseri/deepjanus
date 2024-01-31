import configparser


class IniFile:
    """Class for creating, reading, and modifying INI files"""
    def __init__(self, path):
        """
        Create an interface for an INI file. This will not make any changes on disk until methods are called.
        :param path: the path of the INI file
        """
        self.path = path
        self.config = configparser.ConfigParser(interpolation=None)

    def _save(self):
        """Save the config in memory to the file on disk"""
        with open(self.path, 'w') as configfile:
            self.config.write(configfile)

    def get_option_or_create(self, section, option, default_value):
        """
        Get the option from the file if it was set, otherwise write the default value to file.
        :param section: the section where to find the option
        :param option: the option to be read or created
        :param default_value: the default value to write to file if it was not present
        :return: the value stored in the file if present, the default value otherwise
        """
        val = self.get_option(section, option, None)
        if val and val != '':
            return val
        if not val:
            self.set_option(section, option, default_value)

        return default_value

    def set_option(self, section, option, value):
        """
        Write an option to the file, overriding the old one if it was already present.
        :param section: the section where to write the option
        :param option: the option to be written
        :param value: the value to write
        """
        self._reload()
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config[section][option] = value
        self._save()

    def get_option(self, section, option, default_value):
        """
        Get an option from file.
        :param section: the section where to find the option
        :param option: the option to be read
        :param default_value: the value to return if the option is not present
        :return: the value on file if present, the default_value otherwise
        """
        self._reload()
        if not self.config.has_section(section) or not self.config.has_option(section, option):
            return default_value
        val = self.config[section][option]
        if val == '':
            return default_value
        return val

    def _reload(self):
        """Read the content from the file and stores them in memory, overriding any previously stored options."""
        self.config.clear()
        self.config.read(self.path)
