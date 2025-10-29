"""
Multi-LLM Orchestration Engine
Intelligently routes tasks to optimal LLMs based on requirements
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMType(Enum):
    """Available LLM types"""
    GPT4_TURBO = "gpt4_turbo"
    CLAUDE_3_OPUS = "claude_3_opus"
    CLAUDE_3_SONNET = "claude_3_sonnet"
    LLAMA_2_70B = "llama_2_70b"
    MISTRAL_LARGE = "mistral_large"
    MIXTRAL_8X7B = "mixtral_8x7b"
    QWEN_72B = "qwen_72b"
    DEEPSEEK_67B = "deepseek_67b"


@dataclass
class LLMConfig:
    """Configuration for each LLM"""
    name: LLMType
    api_key: str
    endpoint: str
    max_tokens: int
    temperature: float
    cost_per_1k_tokens: float
    latency_ms: int
    reasoning_strength: float  # 0-1
    code_generation: float  # 0-1
    creativity: float  # 0-1
    reliability: float  # 0-1
    context_window: int


class BaseLLM(ABC):
    """Base class for LLM implementations"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.usage_stats = {"calls": 0, "tokens": 0, "cost": 0.0}
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs):
        """Stream response from LLM"""
        pass
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for semantic understanding"""
        pass


class GPT4Turbo(BaseLLM):
    """GPT-4 Turbo implementation"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate using GPT-4 Turbo"""
        try:
            import openai
            openai.api_key = self.config.api_key
            
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
            )
            
            self.usage_stats["calls"] += 1
            self.usage_stats["tokens"] += response.usage.total_tokens
            self.usage_stats["cost"] += (response.usage.total_tokens / 1000) * self.config.cost_per_1k_tokens
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT-4 error: {e}")
            raise
    
    async def stream_generate(self, prompt: str, **kwargs):
        """Stream response from GPT-4"""
        try:
            import openai
            openai.api_key = self.config.api_key
            
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )
            
            for chunk in response:
                if "choices" in chunk:
                    yield chunk.choices[0].delta.get("content", "")
        except Exception as e:
            logger.error(f"GPT-4 stream error: {e}")
            raise


class ClaudeOpus(BaseLLM):
    """Claude 3 Opus implementation"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate using Claude 3 Opus"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.config.api_key)
            
            message = await asyncio.to_thread(
                client.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                messages=[{"role": "user", "content": prompt}],
            )
            
            self.usage_stats["calls"] += 1
            self.usage_stats["tokens"] += message.usage.input_tokens + message.usage.output_tokens
            self.usage_stats["cost"] += (message.usage.input_tokens + message.usage.output_tokens) / 1000 * self.config.cost_per_1k_tokens
            
            return message.content[0].text
        except Exception as e:
            logger.error(f"Claude error: {e}")
            raise
    
    async def stream_generate(self, prompt: str, **kwargs):
        """Stream response from Claude"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.config.api_key)
            
            with client.messages.stream(
                model="claude-3-opus-20240229",
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Claude stream error: {e}")
            raise


class LocalLLM(BaseLLM):
    """Local LLM implementation (Ollama, vLLM, etc.)"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate using local LLM"""
        try:
            import requests
            
            response = await asyncio.to_thread(
                requests.post,
                self.config.endpoint,
                json={
                    "prompt": prompt,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                },
                timeout=300
            )
            
            self.usage_stats["calls"] += 1
            result = response.json()
            return result.get("text", "")
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            raise
    
    async def stream_generate(self, prompt: str, **kwargs):
        """Stream response from local LLM"""
        try:
            import requests
            
            response = await asyncio.to_thread(
                requests.post,
                self.config.endpoint,
                json={
                    "prompt": prompt,
                    "stream": True,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                },
                stream=True,
                timeout=300
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    yield data.get("token", {}).get("text", "")
        except Exception as e:
            logger.error(f"Local LLM stream error: {e}")
            raise


class LLMOrchestrator:
    """Orchestrates multiple LLMs for optimal task execution"""
    
    def __init__(self):
        self.llms: Dict[LLMType, BaseLLM] = {}
        self.routing_history = []
        self.performance_metrics = {}
    
    def register_llm(self, llm_type: LLMType, llm: BaseLLM):
        """Register an LLM"""
        self.llms[llm_type] = llm
        self.performance_metrics[llm_type] = {
            "success_rate": 1.0,
            "avg_latency": 0,
            "avg_quality": 0.8,
            "cost_efficiency": 1.0
        }
        logger.info(f"Registered LLM: {llm_type.value}")
    
    async def select_optimal_llm(self, task: str, requirements: Dict[str, float]) -> LLMType:
        """
        Select optimal LLM based on task requirements
        
        Requirements:
        - reasoning_strength: 0-1
        - code_generation: 0-1
        - creativity: 0-1
        - speed: 0-1 (1 = fast)
        - cost_efficiency: 0-1 (1 = cheap)
        """
        
        best_score = -1
        best_llm = None
        
        for llm_type, llm in self.llms.items():
            config = llm.config
            metrics = self.performance_metrics[llm_type]
            
            # Calculate match score
            score = (
                config.reasoning_strength * requirements.get("reasoning_strength", 0.5) +
                config.code_generation * requirements.get("code_generation", 0.5) +
                config.creativity * requirements.get("creativity", 0.3) +
                (1 / config.latency_ms * 1000) * requirements.get("speed", 0.2) +
                (1 / config.cost_per_1k_tokens) * requirements.get("cost_efficiency", 0.2)
            ) * metrics["success_rate"]
            
            if score > best_score:
                best_score = score
                best_llm = llm_type
        
        logger.info(f"Selected LLM: {best_llm.value} (score: {best_score:.2f})")
        return best_llm
    
    async def execute_with_fallback(
        self,
        prompt: str,
        requirements: Dict[str, float],
        max_retries: int = 3
    ) -> str:
        """Execute with automatic fallback to other LLMs on failure"""
        
        llm_type = await self.select_optimal_llm(prompt, requirements)
        attempted_llms = [llm_type]
        
        for attempt in range(max_retries):
            try:
                llm = self.llms[llm_type]
                result = await llm.generate(prompt)
                
                # Record success
                self.routing_history.append({
                    "llm": llm_type.value,
                    "success": True,
                    "attempt": attempt + 1
                })
                
                return result
            
            except Exception as e:
                logger.warning(f"LLM {llm_type.value} failed: {e}")
                
                # Try next best LLM
                if attempt < max_retries - 1:
                    remaining_llms = [l for l in self.llms.keys() if l not in attempted_llms]
                    if remaining_llms:
                        llm_type = remaining_llms[0]
                        attempted_llms.append(llm_type)
                        continue
                
                raise
        
        raise RuntimeError("All LLMs failed")
    
    async def ensemble_reasoning(
        self,
        prompt: str,
        num_models: int = 3
    ) -> Dict[str, Any]:
        """
        Use multiple LLMs in parallel for ensemble reasoning
        Combines outputs for better quality
        """
        
        selected_llms = list(self.llms.keys())[:num_models]
        
        tasks = [
            self.llms[llm_type].generate(prompt)
            for llm_type in selected_llms
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "responses": {
                llm_type.value: result
                for llm_type, result in zip(selected_llms, results)
            },
            "consensus": self._compute_consensus(results),
            "confidence": self._compute_confidence(results)
        }
    
    def _compute_consensus(self, responses: List[str]) -> str:
        """Compute consensus from multiple responses"""
        # Simple implementation - can be enhanced with semantic similarity
        return responses[0] if responses else ""
    
    def _compute_confidence(self, responses: List[str]) -> float:
        """Compute confidence score for ensemble"""
        # Simple implementation - can be enhanced
        return 0.8
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "total_calls": sum(llm.usage_stats["calls"] for llm in self.llms.values()),
            "total_tokens": sum(llm.usage_stats["tokens"] for llm in self.llms.values()),
            "total_cost": sum(llm.usage_stats["cost"] for llm in self.llms.values()),
            "routing_history": self.routing_history[-100:],  # Last 100
            "performance_metrics": self.performance_metrics
        }


# Initialize orchestrator
orchestrator = LLMOrchestrator()


async def initialize_orchestrator(config_file: str = None):
    """Initialize orchestrator with all available LLMs"""
    
    # Register GPT-4 Turbo
    gpt4_config = LLMConfig(
        name=LLMType.GPT4_TURBO,
        api_key="sk-...",  # Set from env
        endpoint="https://api.openai.com/v1/chat/completions",
        max_tokens=4096,
        temperature=0.7,
        cost_per_1k_tokens=0.03,
        latency_ms=2000,
        reasoning_strength=0.95,
        code_generation=0.9,
        creativity=0.8,
        reliability=0.98,
        context_window=128000
    )
    orchestrator.register_llm(LLMType.GPT4_TURBO, GPT4Turbo(gpt4_config))
    
    # Register Claude 3 Opus
    claude_config = LLMConfig(
        name=LLMType.CLAUDE_3_OPUS,
        api_key="sk-ant-...",  # Set from env
        endpoint="https://api.anthropic.com/v1/messages",
        max_tokens=4096,
        temperature=0.7,
        cost_per_1k_tokens=0.015,
        latency_ms=3000,
        reasoning_strength=0.97,
        code_generation=0.85,
        creativity=0.9,
        reliability=0.99,
        context_window=200000
    )
    orchestrator.register_llm(LLMType.CLAUDE_3_OPUS, ClaudeOpus(claude_config))
    
    # Register local Llama 2 70B
    llama_config = LLMConfig(
        name=LLMType.LLAMA_2_70B,
        api_key="",
        endpoint="http://localhost:8000/v1/completions",
        max_tokens=4096,
        temperature=0.7,
        cost_per_1k_tokens=0.0,
        latency_ms=5000,
        reasoning_strength=0.75,
        code_generation=0.8,
        creativity=0.85,
        reliability=0.85,
        context_window=4096
    )
    orchestrator.register_llm(LLMType.LLAMA_2_70B, LocalLLM(llama_config))
    
    logger.info("LLM Orchestrator initialized with 3 models")
    return orchestrator


if __name__ == "__main__":
    # Test orchestrator
    async def test():
        orch = await initialize_orchestrator()
        
        # Test LLM selection
        requirements = {
            "reasoning_strength": 0.9,
            "code_generation": 0.8,
            "speed": 0.5,
            "cost_efficiency": 0.3
        }
        
        selected = await orch.select_optimal_llm("Test task", requirements)
        print(f"Selected LLM: {selected.value}")
        
        # Get stats
        stats = orch.get_stats()
        print(f"Stats: {json.dumps(stats, indent=2, default=str)}")
    
    asyncio.run(test())
