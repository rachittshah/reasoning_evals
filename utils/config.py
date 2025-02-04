import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

class ConfigurationError(Exception):
    """Raised when there's an error in the configuration."""
    pass

class Config:
    """Configuration manager for the evaluation framework."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses default.
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "config", 
            "config.yaml"
        )
        self.config: Dict[str, Any] = {}
        self._load_env()
        self._load_config()
        self._validate_config()
    
    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        load_dotenv(env_path)
        
        # Validate required environment variables
        required_vars = [
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Error loading config file: {str(e)}")
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_sections = [
            "models",
            "tasks",
            "synthetic_data",
            "evaluation",
            "logging",
            "reporting"
        ]
        
        missing_sections = [
            section for section in required_sections 
            if section not in self.config
        ]
        
        if missing_sections:
            raise ConfigurationError(
                f"Missing required configuration sections: {', '.join(missing_sections)}"
            )
        
        # Validate models configuration
        for provider in ["openai", "deepseek"]:
            if provider not in self.config["models"]:
                raise ConfigurationError(f"Missing configuration for {provider}")
            if "models" not in self.config["models"][provider]:
                raise ConfigurationError(f"No models defined for {provider}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            The configuration value
        """
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self, provider: str, model_alias: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            provider: The model provider (e.g., 'openai', 'deepseek')
            model_alias: The model alias (e.g., 'o1', 'r1')
            
        Returns:
            Model configuration dictionary
        """
        models = self.config["models"][provider]["models"]
        for model in models:
            if model["alias"] == model_alias:
                return model
        raise ConfigurationError(f"Model {model_alias} not found for provider {provider}")
    
    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task configuration dictionary
        """
        if task_name not in self.config["tasks"]:
            raise ConfigurationError(f"Task {task_name} not found in configuration")
        return self.config["tasks"][task_name]

# Global configuration instance
config = Config() 