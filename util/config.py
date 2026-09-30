import json


class Config:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._data = json.load(f)

    # allow attribute-style access to top-level configuration sections (e.g., config.model)
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f'No section (attribute) named {name}')

    # pretty-print the configuration
    def __str__(self):
        return json.dumps(self._data, indent=2, ensure_ascii=False, default=str)

    def get(self, key, default=None):
        keys = key.split('.')
        value = self._data
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    # update configuration from a command-line argument dictionary (for overriding settings)
    def update_from_cli(self, args_dict):
        for key, value in args_dict.items():
            if value is not None:  # only update parameters that were provided
                keys = key.split('.')
                data = self._data
                for k in keys[:-1]:
                    data = data.setdefault(k, {})
                data[keys[-1]] = value

    def save(self, path=None):
        save_path = path if path else self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False, default=str)  # use 'default=str' to handle non-JSON-serializable objects like Path


config = Config('config.json')
