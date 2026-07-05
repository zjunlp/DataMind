#!/usr/bin/env python3
"""
Configuration module for DDR_Bench.

Provides centralized configuration management for all scenarios (mimic, 10-k, globem).
Configuration can be loaded from environment variables, YAML file, or CLI arguments.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


@dataclass
class ScenarioConfig:
    """Configuration for a specific scenario."""
    name: str
    db_path: str = ""
    data_path: str = ""
    qa_file: str = ""
    input_file: str = ""
    id_file: str = ""  # Pre-defined identifier list file
    log_dir: str = ""
    identifier_prefix: str = ""


@dataclass
class ProviderConfig:
    """Configuration for LLM providers."""
    default_provider: str = "gemini"
    default_model: str = "gemini-2.5-flash"
    
    # API keys (loaded from environment)
    gemini_api_key: str = ""
    openai_api_key: str = ""
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"
    minimax_api_key: str = ""
    
    # VLLM settings
    vllm_base_url: str = "http://localhost:8000"
    vllm_port: int = 8000
    vllm_api_key: str = "EMPTY"


@dataclass
class AgentConfig:
    """Configuration for the data agent."""
    max_turns: int = 100
    max_retries: int = 2
    auto_finish: bool = True
    log_level: str = "INFO"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    provider: str = "azure"
    model: str = "gpt-5-mini"
    max_retries: int = 5
    retry_delay: float = 2.0


@dataclass
class Config:
    """Main configuration class for DDR_Bench."""
    
    # Scenario configurations
    scenarios: Dict[str, ScenarioConfig] = field(default_factory=dict)
    
    # Provider configuration
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    
    # Agent configuration
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    # Evaluation configuration
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Base paths
    base_data_dir: str = ""
    base_log_dir: str = "./logs"
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load configuration from YAML file and environment variables.
        
        Priority: CLI args > Environment variables > YAML file > Defaults
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Loaded Config instance
        """
        config = cls()
        
        # Load from YAML file if provided
        if config_path and Path(config_path).exists():
            config._load_from_yaml(config_path)
        else:
            # Try default locations
            for default_path in ["config.yaml", "config.yml", ".config.yaml"]:
                if Path(default_path).exists():
                    config._load_from_yaml(default_path)
                    break
        
        # Override with environment variables
        config._load_from_env()
        
        # Initialize default scenarios if not configured
        config._init_default_scenarios()
        
        return config
    
    def _load_from_yaml(self, path: str) -> None:
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        
        # Load base paths
        self.base_data_dir = data.get("base_data_dir", self.base_data_dir)
        self.base_log_dir = data.get("base_log_dir", self.base_log_dir)
        
        # Load provider config
        if "provider" in data:
            prov = data["provider"]
            self.provider.default_provider = prov.get("default_provider", self.provider.default_provider)
            self.provider.default_model = prov.get("default_model", self.provider.default_model)
            self.provider.vllm_base_url = prov.get("vllm_base_url", self.provider.vllm_base_url)
            self.provider.vllm_port = prov.get("vllm_port", self.provider.vllm_port)
        
        # Load agent config
        if "agent" in data:
            agent = data["agent"]
            self.agent.max_turns = agent.get("max_turns", self.agent.max_turns)
            self.agent.max_retries = agent.get("max_retries", self.agent.max_retries)
            self.agent.auto_finish = agent.get("auto_finish", self.agent.auto_finish)
            self.agent.log_level = agent.get("log_level", self.agent.log_level)
        
        # Load evaluation config
        if "evaluation" in data:
            eval_cfg = data["evaluation"]
            self.evaluation.provider = eval_cfg.get("provider", self.evaluation.provider)
            self.evaluation.model = eval_cfg.get("model", self.evaluation.model)
            self.evaluation.max_retries = eval_cfg.get("max_retries", self.evaluation.max_retries)
            self.evaluation.retry_delay = eval_cfg.get("retry_delay", self.evaluation.retry_delay)
        
        # Load scenario configs
        if "scenarios" in data:
            for name, scenario_data in data["scenarios"].items():
                self.scenarios[name] = ScenarioConfig(
                    name=name,
                    db_path=scenario_data.get("db_path", ""),
                    data_path=scenario_data.get("data_path", ""),
                    qa_file=scenario_data.get("qa_file", ""),
                    input_file=scenario_data.get("input_file", ""),
                    id_file=scenario_data.get("id_file", ""),
                    log_dir=scenario_data.get("log_dir", f"./{name}_logs"),
                    identifier_prefix=scenario_data.get("identifier_prefix", name[:3])
                )
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # API keys
        self.provider.gemini_api_key = os.getenv("GEMINI_API_KEY", self.provider.gemini_api_key)
        self.provider.openai_api_key = os.getenv("OPENAI_API_KEY", self.provider.openai_api_key)
        self.provider.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", self.provider.azure_openai_api_key)
        self.provider.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", self.provider.azure_openai_endpoint)
        self.provider.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", self.provider.azure_openai_api_version)
        self.provider.minimax_api_key = os.getenv("MINIMAX_API_KEY", os.getenv("MINIMAX_APIKEY", self.provider.minimax_api_key))
        
        # VLLM settings
        vllm_url = os.getenv("VLLM_BASE_URL")
        if vllm_url:
            self.provider.vllm_base_url = vllm_url
        
        vllm_port = os.getenv("VLLM_PORT")
        if vllm_port:
            self.provider.vllm_port = int(vllm_port)
        
        # Log level
        log_level = os.getenv("DDR_LOG_LEVEL")
        if log_level:
            self.agent.log_level = log_level
    
    def _init_default_scenarios(self) -> None:
        """Initialize default scenario configurations if not already set."""
        if "mimic" not in self.scenarios:
            self.scenarios["mimic"] = ScenarioConfig(
                name="mimic",
                db_path="",  # User must configure
                input_file="",  # User must configure
                id_file="",  # User must configure
                qa_file="",  # User must configure
                log_dir="./mimic_logs",
                identifier_prefix="patient"
            )
        
        if "10k" not in self.scenarios:
            self.scenarios["10k"] = ScenarioConfig(
                name="10k",
                db_path="",  # User must configure
                id_file="",  # User must configure
                qa_file="",  # User must configure
                log_dir="./10k_logs",
                identifier_prefix="company"
            )
        
        if "globem" not in self.scenarios:
            self.scenarios["globem"] = ScenarioConfig(
                name="globem",
                data_path="",  # User must configure
                id_file="",  # User must configure
                qa_file="",  # User must configure
                log_dir="./globem_logs",
                identifier_prefix="user"
            )
    
    def get_scenario(self, name: str) -> ScenarioConfig:
        """
        Get configuration for a specific scenario.
        
        Args:
            name: Scenario name (mimic, 10k, globem)
            
        Returns:
            ScenarioConfig for the requested scenario
            
        Raises:
            ValueError: If scenario is not configured
        """
        name_lower = name.lower()
        if name_lower not in self.scenarios:
            raise ValueError(f"Unknown scenario: {name}. Available: {list(self.scenarios.keys())}")
        return self.scenarios[name_lower]
    
    def get_provider_api_key(self, provider: str) -> str:
        """
        Get API key for a specific provider.
        
        Args:
            provider: Provider name (gemini, openai, azure, minimax)
            
        Returns:
            API key string
        """
        provider_lower = provider.lower()
        if provider_lower == "gemini":
            return self.provider.gemini_api_key
        elif provider_lower == "openai":
            return self.provider.openai_api_key
        elif provider_lower in ("azure", "azure_openai"):
            return self.provider.azure_openai_api_key
        elif provider_lower == "minimax":
            return self.provider.minimax_api_key
        else:
            return ""


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Global Config instance
    """
    global _config
    if _config is None:
        _config = Config.load(config_path)
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Reload configuration from file.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        New Config instance
    """
    global _config
    _config = Config.load(config_path)
    return _config
