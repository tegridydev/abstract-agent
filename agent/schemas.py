# schemas.py
# Author: tegridydev
# Repo: https://github.com/tegridydev/abstract-agent
# License: MIT
# Year: 2025

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import yaml

class AgentConfig(BaseModel):
    name: str = Field(..., description="Agent name")
    model: str = Field(default="qwen3:0.6b", description="AI model to use")
    persona: str = Field(..., description="Agent persona description")
    prompt: str = Field(..., description="Prompt template")

    @validator('name')
    def validate_name(cls, v):
        if not v or not isinstance(v, str) or len(v) < 3:
            raise ValueError("Agent name must be a string of at least 3 characters")
        return v

    @validator('model')
    def validate_model(cls, v):
        if not v or not isinstance(v, str) or ':' not in v:
            raise ValueError("Model must be in format 'name:version'")
        return v

    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or not isinstance(v, str) or len(v) < 10:
            raise ValueError("Prompt must be a string of at least 10 characters")
        return v

class ConfigSchema(BaseModel):
    agents: List[AgentConfig] = Field(..., description="List of agent configurations")

    @validator('agents')
    def validate_agents(cls, v):
        if len(v) < 1:
            raise ValueError("At least one agent must be configured")
        return v

    @classmethod
    def validate_yaml(cls, yaml_content: str) -> 'ConfigSchema':
        """Validate YAML configuration content"""
        try:
            config_dict = yaml.safe_load(yaml_content)
            return cls(**config_dict)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")
