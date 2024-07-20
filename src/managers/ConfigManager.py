import json

class ConfigManager:
    @staticmethod
    def get_config(config_file_path: str) -> dict:
        # Get the absolute path of the configuration file
        print(f"[INFO] Loading config file from {config_file_path}\n")
        # Open and read the configuration file
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    
    @staticmethod
    def get_weights_file_path(config_file_path: str, 
                             epoch: str, 
                             ) -> str:
        config = ConfigManager.get_config(config_file_path)
        model_folder = config['model_folder']
        model_basename = config['model_basename']
        model_filename = f"{model_basename}{epoch}.pt"
        # Join model folder and model filename to get the full path
        full_path = os.path.join(model_folder, model_filename)
        return os.path.abspath(full_path)

