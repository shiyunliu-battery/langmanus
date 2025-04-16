from .env import (
    # Azure DeepSeek-R1 Configuration
    AZURE_ENDPOINT,
    AZURE_DEPLOYMENT_NAME,
    AZURE_INFERENCE_SDK_KEY,
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_API_KEY,
    # Reasoning LLM
    REASONING_MODEL,
    REASONING_BASE_URL,
    REASONING_API_KEY,
    # Basic LLM
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    # Vision-language LLM
    VL_MODEL,
    VL_BASE_URL,
    VL_API_KEY,
    # Other configurations
    CHROME_INSTANCE_PATH,
)
from .tools import TAVILY_MAX_RESULTS

# Team configuration
TEAM_MEMBERS = ["researcher", "coder", "browser", "reporter"]

__all__ = [
    # Azure DeepSeek-R1 Configuration
    "AZURE_ENDPOINT",
    "AZURE_DEPLOYMENT_NAME",
    "AZURE_INFERENCE_SDK_KEY",
    # Azure OpenAI Configuration
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
    # Reasoning LLM
    "REASONING_MODEL",
    "REASONING_BASE_URL",
    "REASONING_API_KEY",
    # Basic LLM
    "BASIC_MODEL",
    "BASIC_BASE_URL",
    "BASIC_API_KEY",
    # Vision-language LLM
    "VL_MODEL",
    "VL_BASE_URL",
    "VL_API_KEY",
    # Other configurations
    "TEAM_MEMBERS",
    "TAVILY_MAX_RESULTS",
    "CHROME_INSTANCE_PATH",
]
