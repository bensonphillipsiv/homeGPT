"""
Configuration management for the assistant.
"""
import os
from dataclasses import dataclass
from typing import Literal
import json

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration"""
    
    # API settings
    enable_api: bool
    api_host: str
    api_port: int
    
    # Audio device settings
    audio_type: Literal["local", "remote"]
    audio_device_ip: str
    audio_listener_port: int
    audio_speaker_port: int

    # Model paths
    whisper_model: str
    whisper_device: str
    tts_model_path: str
    
    # Agent settings
    model_provider: Literal["openai", "bedrock"]
    openai_api_key: str | None
    openai_model: str
    bedrock_model: str
    bedrock_region: str
    
    # MCP servers
    mcps: list[str]
    
    # Logging
    log_level: str


def load_config() -> Config:
    """Load configuration from environment variables"""
    load_dotenv()
    
    return Config(
        # API
        enable_api=os.getenv("ENABLE_API", "true").lower() == "true",
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        
        # Audio
        audio_type=os.getenv("AUDIO_TYPE", "local"),
        audio_device_ip=os.getenv("AUDIO_DEVICE_IP", "192.168.1.132"),
        audio_listener_port=int(os.getenv("AUDIO_LISTENER_PORT", "4712")),
        audio_speaker_port=int(os.getenv("AUDIO_SPEAKER_PORT", "4713")),
        
        # Models
        whisper_model=os.getenv("WHISPER_MODEL", "base.en"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        tts_model_path=os.getenv("TTS_MODEL_PATH", "./models/lessac_low.onnx"),
        
        # Agent - Model Provider
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),  # "openai" or "bedrock"
        
        # OpenAI settings
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        
        # Bedrock settings (uses AWS credentials from environment)
        bedrock_model=os.getenv("BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"),
        bedrock_region=os.getenv("BEDROCK_REGION", "us-east-2"),
        
        # MCPs
        mcps=(os.getenv("MCPS", "mcps.hass.main,mcps.common.main")).split(','),

        # Logging
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )


def setup_logging(level: str = "INFO"):
    """Configure logging for the application"""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Reduce noise from some libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)