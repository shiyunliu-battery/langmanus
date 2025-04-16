import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure DeepSeek R1 configuration
AZURE_ENDPOINT = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT", "https://ai-code764649025142.services.ai.azure.com/models")
AZURE_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "DeepSeek-R1")
AZURE_INFERENCE_SDK_KEY = os.getenv("AZURE_INFERENCE_SDK_KEY", "YOUR_KEY_HERE")

# Azure OpenAI configuration with API Key
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://ionera.openai.azure.com/")
# Use a different env var for OpenAI deployment to avoid conflicts with DeepSeek
AZURE_OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT_NAME", "gpt-4o")  # Changed default from gpt-4o to gpt4o
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_KEY_HERE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# Reasoning LLM configuration (for complex reasoning tasks) - Using Azure DeepSeek-R1
REASONING_MODEL = os.getenv("REASONING_MODEL", "DeepSeek-R1")
REASONING_BASE_URL = os.getenv("REASONING_BASE_URL")
REASONING_API_KEY = os.getenv("REASONING_API_KEY")

# Non-reasoning LLM configuration (for straightforward tasks) - Using Azure OpenAI GPT-4o
BASIC_MODEL = os.getenv("BASIC_MODEL", AZURE_OPENAI_DEPLOYMENT)
BASIC_BASE_URL = os.getenv("BASIC_BASE_URL", AZURE_OPENAI_ENDPOINT)
BASIC_API_KEY = os.getenv("BASIC_API_KEY", AZURE_OPENAI_API_KEY)

# Vision-language LLM configuration (for tasks requiring visual understanding) - Azure OpenAI GPT-4o
VL_MODEL = os.getenv("VL_MODEL", AZURE_OPENAI_DEPLOYMENT)
VL_BASE_URL = os.getenv("VL_BASE_URL", AZURE_OPENAI_ENDPOINT)
VL_API_KEY = os.getenv("VL_API_KEY", AZURE_OPENAI_API_KEY)

# Chrome Instance configuration
CHROME_INSTANCE_PATH = os.getenv("CHROME_INSTANCE_PATH")
