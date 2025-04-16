import os
import sys
# Add project root to Python path to make 'src' importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_deepseek import ChatDeepSeek
from typing import Optional
import base64
from openai import AzureOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import Any, Dict, List, Mapping, Optional, Iterator, Union
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage as AzureSystemMessage
from azure.ai.inference.models import UserMessage as AzureUserMessage
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from pydantic import Field

from src.config import (
    REASONING_MODEL,
    REASONING_BASE_URL,
    REASONING_API_KEY,
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    VL_MODEL,
    VL_BASE_URL,
    VL_API_KEY,
    AZURE_ENDPOINT,
    AZURE_DEPLOYMENT_NAME,
    AZURE_INFERENCE_SDK_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
)
from src.config.agents import LLMType


def create_openai_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with the specified configuration
    """
    # Only include base_url in the arguments if it's not None or empty
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if base_url:  # This will handle None or empty string
        llm_kwargs["base_url"] = base_url

    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key

    return ChatOpenAI(**llm_kwargs)


def create_deepseek_llm(
    model: str = "deepseek-chat",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatDeepSeek:
    """
    Create a ChatDeepSeek instance with the specified configuration
    """
    # Only include base_url in the arguments if it's not None or empty
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if base_url:  # This will handle None or empty string
        llm_kwargs["api_base"] = base_url

    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key

    return ChatDeepSeek(**llm_kwargs)


class AzureDeepseekR1(BaseChatModel):
    """
    A custom LangChain integration for Azure DeepSeek-R1 using Azure AI Inference SDK
    """
    
    model_name: str = Field(default="DeepSeek-R1")
    endpoint: str = Field()
    deployment_name: str = Field()
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=1000)
    client: Optional[ChatCompletionsClient] = None
    
    def __init__(
        self, 
        endpoint: str, 
        deployment_name: str, 
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        """Initialize the AzureDeepseekR1 model."""
        super().__init__(
            endpoint=endpoint,
            deployment_name=deployment_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        try:
            # Use AzureKeyCredential if API key is provided, otherwise fall back to DefaultAzureCredential
            if api_key:
                self.client = ChatCompletionsClient(
                    endpoint=self.endpoint, 
                    credential=AzureKeyCredential(api_key)
                )
            else:
                print("No API key provided for Azure DeepSeek-R1. Using DefaultAzureCredential.")
                self.client = ChatCompletionsClient(
                    endpoint=self.endpoint, 
                    credential=DefaultAzureCredential()
                )
        except Exception as e:
            print(f"Error initializing ChatCompletionsClient: {e}")
            self.client = None
    
    def _convert_messages_to_azure_format(self, messages: List[BaseMessage]) -> List[Union[AzureSystemMessage, AzureUserMessage]]:
        azure_messages = []
        for message in messages:
            if message.type == "system":
                azure_messages.append(AzureSystemMessage(content=message.content))
            elif message.type == "human":
                azure_messages.append(AzureUserMessage(content=message.content))
            # For simplicity, we're treating assistant messages as user messages containing the AI's response
            elif message.type == "ai" and len(azure_messages) > 0:
                # Combine with previous messages if possible
                if isinstance(azure_messages[-1], AzureUserMessage):
                    user_message = azure_messages[-1]
                    azure_messages[-1] = AzureUserMessage(content=f"{user_message.content}\n\n{message.content}")
                else:
                    azure_messages.append(AzureUserMessage(content=message.content))
        return azure_messages
    
    def _generate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        if not self.client:
            raise ValueError("ChatCompletionsClient not initialized")
        
        azure_messages = self._convert_messages_to_azure_format(messages)
        
        response = self.client.complete(
            messages=azure_messages,
            model=self.deployment_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **kwargs
        )
        
        ai_message = AIMessage(content=response.choices[0].message.content)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])
    
    async def _agenerate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        # For simplicity, we'll use the synchronous method for now
        return self._generate(messages, stop, run_manager, **kwargs)
    
    def _stream(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> Iterator[ChatGeneration]:
        # Currently, Azure DeepSeek-R1 with Azure SDK doesn't have a streaming option in the same way
        # For now, we'll simulate streaming by returning the whole response at once
        response = self._generate(messages, stop, run_manager, **kwargs)
        yield response.generations[0]
    
    @property
    def _llm_type(self) -> str:
        return "azure-deepseek-r1"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "endpoint": self.endpoint,
            "deployment_name": self.deployment_name,
        }


class AzureOpenAIWithAPIKey(BaseChatModel):
    """
    A custom LangChain integration for Azure OpenAI using API Key authentication
    """
    
    endpoint: str = Field()
    deployment_name: str = Field()
    api_key: str = Field()
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=800)
    api_version: str = Field(default="2025-01-01-preview")
    client: Optional[AzureOpenAI] = None

    def __init__(
        self, 
        endpoint: str, 
        deployment_name: str,
        api_key: str, 
        temperature: float = 0.7,
        max_tokens: int = 800,
        api_version: str = "2025-01-01-preview",
        **kwargs
    ):
        """Initialize the AzureOpenAIWithAPIKey model."""
        super().__init__(
            endpoint=endpoint, 
            deployment_name=deployment_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            api_version=api_version,
            **kwargs
        )
        
        try:
            # Initialize Azure OpenAI client with API key authentication
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
    
    def _convert_messages_to_openai_format(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        openai_messages = []
        for message in messages:
            if message.type == "system":
                openai_messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": message.content}]
                })
            elif message.type == "human":
                openai_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": message.content}]
                })
            elif message.type == "ai":
                openai_messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": message.content}]
                })
        return openai_messages
    
    def _generate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        if not self.client:
            raise ValueError("AzureOpenAI client not initialized")
        
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        completion = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=openai_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop,
            stream=False,
            **kwargs
        )
        
        ai_message = AIMessage(content=completion.choices[0].message.content)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])
    
    async def _agenerate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        # For simplicity, we'll use the synchronous method for now
        return self._generate(messages, stop, run_manager, **kwargs)
    
    def _stream(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> Iterator[ChatGeneration]:
        if not self.client:
            raise ValueError("AzureOpenAI client not initialized")
        
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        stream = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=openai_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield ChatGeneration(message=AIMessage(content=chunk.choices[0].delta.content))
    
    @property
    def _llm_type(self) -> str:
        return "azure-openai-api-key"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "deployment_name": self.deployment_name,
            "endpoint": self.endpoint,
            "api_version": self.api_version
        }


def create_azure_openai_with_api_key(
    endpoint: str = BASIC_BASE_URL,
    deployment_name: str = BASIC_MODEL,
    api_key: str = BASIC_API_KEY,
    temperature: float = 0.7,
    api_version: str = "2025-01-01-preview",
    **kwargs,
) -> Union[AzureOpenAIWithAPIKey, ChatOpenAI]:
    """
    Create an AzureOpenAIWithAPIKey instance with API key authentication
    """
    if not endpoint or not api_key:
        print("Warning: Azure OpenAI endpoint or API key not specified. Using fallback to OpenAI.")
        return create_openai_llm(model="gpt-4o")
        
    try:
        return AzureOpenAIWithAPIKey(
            endpoint=endpoint,
            deployment_name=deployment_name,
            api_key=api_key,
            temperature=temperature,
            api_version=api_version,
            **kwargs
        )
    except Exception as e:
        print(f"Error creating AzureOpenAIWithAPIKey: {e}, using fallback to OpenAI")
        return create_openai_llm(model="gpt-4o")


def create_azure_deepseek_r1(
    endpoint: str = AZURE_ENDPOINT,
    deployment_name: str = AZURE_DEPLOYMENT_NAME,
    api_key: str = AZURE_INFERENCE_SDK_KEY,
    temperature: float = 0.0,
    **kwargs,
) -> Union[AzureDeepseekR1, ChatDeepSeek]:
    """
    Create an AzureDeepseekR1 instance with the specified configuration
    """
    if not endpoint or not deployment_name:
        print(f"Warning: Azure DeepSeek-R1 configuration missing. Using fallback to ChatDeepSeek.")
        # Fallback to regular ChatDeepSeek if Azure config is missing
        return create_deepseek_llm(
            model=REASONING_MODEL,
            base_url=REASONING_BASE_URL,
            api_key=REASONING_API_KEY,
        )
    
    try:
        return AzureDeepseekR1(
            endpoint=endpoint,
            deployment_name=deployment_name,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )
    except Exception as e:
        print(f"Error creating AzureDeepseekR1: {e}, using fallback to ChatDeepSeek")
        return create_deepseek_llm(
            model=REASONING_MODEL,
            base_url=REASONING_BASE_URL,
            api_key=REASONING_API_KEY,
        )


# Cache for LLM instances
_llm_cache: dict[LLMType, Union[ChatOpenAI, AzureOpenAIWithAPIKey, ChatDeepSeek, AzureDeepseekR1]] = {}


def get_llm_by_type(llm_type: LLMType) -> Union[ChatOpenAI, AzureOpenAIWithAPIKey, ChatDeepSeek, AzureDeepseekR1]:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    try:
        if llm_type == "reasoning":
            # Try to use Azure DeepSeek-R1
            llm = create_azure_deepseek_r1()
        elif llm_type == "basic":
            # Try to use Azure OpenAI GPT-4o with API key authentication
            llm = create_azure_openai_with_api_key(
                endpoint=BASIC_BASE_URL,
                deployment_name=BASIC_MODEL,
                api_key=BASIC_API_KEY,
            )
        elif llm_type == "vision":
            # Try to use Azure OpenAI GPT-4o with vision capabilities and API key authentication
            llm = create_azure_openai_with_api_key(
                endpoint=VL_BASE_URL,
                deployment_name=VL_MODEL,
                api_key=VL_API_KEY,
            )
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")

        _llm_cache[llm_type] = llm
        return llm
    except Exception as e:
        print(f"Error initializing LLM of type {llm_type}: {e}")
        # Fallback options if Azure integration fails
        if llm_type == "reasoning":
            fallback = create_openai_llm(model="gpt-4o")
        else:
            fallback = create_openai_llm(model="gpt-4o")
        _llm_cache[llm_type] = fallback
        return fallback


# Initialize LLMs for different purposes - now these will be cached
try:
    reasoning_llm = get_llm_by_type("reasoning")
    basic_llm = get_llm_by_type("basic")
    vl_llm = get_llm_by_type("vision")
except Exception as e:
    print(f"Error initializing LLMs: {e}")
    # Set fallback LLMs - using OpenAI models
    reasoning_llm = create_openai_llm(model="gpt-4o")
    basic_llm = create_openai_llm(model="gpt-4o")
    vl_llm = create_openai_llm(model="gpt-4o")


if __name__ == "__main__":
    # Test Azure OpenAI with API key authentication
    print("Testing Azure OpenAI with API key authentication...")
    try:
        # Example of using Azure OpenAI API with key-based authentication
        endpoint = AZURE_OPENAI_ENDPOINT
        deployment = AZURE_OPENAI_DEPLOYMENT  # Use the OpenAI-specific deployment name
        subscription_key = AZURE_OPENAI_API_KEY
        
        print(f"Azure OpenAI Endpoint: {endpoint}")
        print(f"Azure OpenAI Deployment: {deployment}")
        print(f"API Key available: {'Yes' if subscription_key and subscription_key != 'YOUR_KEY_HERE' else 'No'}")
        
        # Initialize Azure OpenAI Service client with key-based authentication    
        azure_client = AzureOpenAI(  
            azure_endpoint=endpoint,  
            api_key=subscription_key,  
            api_version=AZURE_OPENAI_API_VERSION,
        )
            
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that helps people find information."
                    }
                ]
            }
        ] 
            
        # Generate the completion  
        completion = azure_client.chat.completions.create(  
            model=deployment,
            messages=chat_prompt,
            max_tokens=800,  
            temperature=0.7,  
            top_p=0.95,  
            frequency_penalty=0,  
            presence_penalty=0,
            stop=None,  
            stream=False
        )
        print("Azure OpenAI API Response:", completion.choices[0].message.content)
    except Exception as e:
        print(f"Error testing Azure OpenAI API: {e}")

    # Test Azure DeepSeek-R1
    print("\nTesting Azure DeepSeek-R1...")
    try:
        # Example of using Azure DeepSeek-R1
        endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT", "https://ai-code764649025142.services.ai.azure.com/models")
        model_name = os.getenv("DEPLOYMENT_NAME", "DeepSeek-R1")
        key = os.getenv("AZURE_INFERENCE_SDK_KEY", "YOUR_KEY_HERE")
        
        if key == "YOUR_KEY_HERE":
            print("⚠️ No Azure DeepSeek-R1 API key provided in environment. Using placeholder.")
        
        deepseek_client = ChatCompletionsClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )

        response = deepseek_client.complete(
            messages=[
                AzureSystemMessage(content="You are a helpful assistant."),
                AzureUserMessage(content="What are 3 things to visit in Seattle?")
            ],
            model=model_name,
            max_tokens=1000
        )
        print("Azure DeepSeek-R1 Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error testing Azure DeepSeek-R1: {e}")

    # Test LangChain integration
    print("\nTesting LangChain integration...")
    try:
        response = reasoning_llm.invoke("What is MCP?")
        print(f"Response from reasoning LLM: {response.content}")
        
        response = basic_llm.invoke("Hello, who are you?")
        print(f"Response from basic LLM: {response.content}")
    except Exception as e:
        print(f"Error testing LangChain LLMs: {e}")
