"""
Distributed Multi-Agent Framework
Specialized agents for different domains working in parallel
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Agent specializations"""
    ORCHESTRATOR = "orchestrator"
    WEB_AUTOMATION = "web_automation"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_CREATION = "content_creation"
    PLANNING = "planning"
    REFLECTION = "reflection"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    INTEGRATION = "integration"


@dataclass
class AgentMessage:
    """Message between agents"""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0  # 0-10, higher = more urgent


@dataclass
class AgentCapabilities:
    """Agent capabilities and constraints"""
    max_parallel_tasks: int = 5
    timeout_seconds: int = 300
    memory_limit_mb: int = 512
    allowed_tools: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, capabilities: AgentCapabilities):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.state = "idle"
        self.current_tasks = []
        self.message_queue = asyncio.Queue()
        self.execution_history = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_execution_time": 0,
            "success_rate": 1.0
        }
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task"""
        pass
    
    @abstractmethod
    async def handle_message(self, message: AgentMessage):
        """Handle incoming message from another agent"""
        pass
    
    async def send_message(self, receiver_id: str, message_type: str, content: Dict[str, Any]):
        """Send message to another agent"""
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content
        )
        # This would be handled by message broker in production
        logger.info(f"Agent {self.agent_id} -> {receiver_id}: {message_type}")
    
    async def add_task(self, task: Dict[str, Any]):
        """Add task to queue"""
        if len(self.current_tasks) >= self.capabilities.max_parallel_tasks:
            raise RuntimeError(f"Agent {self.agent_id} at max capacity")
        
        self.current_tasks.append(task)
        logger.info(f"Agent {self.agent_id} added task: {task.get('id', 'unknown')}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "state": self.state,
            "current_tasks": len(self.current_tasks),
            "performance_metrics": self.performance_metrics
        }


class OrchestratorAgent(BaseAgent):
    """Master orchestrator agent - coordinates all other agents"""
    
    def __init__(self):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type=AgentType.ORCHESTRATOR,
            capabilities=AgentCapabilities(max_parallel_tasks=20)
        )
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.task_assignments: Dict[str, List[str]] = {}
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose task and assign to specialized agents"""
        
        self.state = "planning"
        task_id = task.get("id", str(uuid.uuid4()))
        
        # Decompose task
        subtasks = await self._decompose_task(task)
        logger.info(f"Decomposed task {task_id} into {len(subtasks)} subtasks")
        
        # Assign to agents
        assignments = await self._assign_subtasks(subtasks)
        self.task_assignments[task_id] = assignments
        
        # Execute in parallel
        self.state = "executing"
        results = await self._execute_parallel(assignments)
        
        # Aggregate results
        self.state = "aggregating"
        final_result = await self._aggregate_results(results)
        
        self.state = "idle"
        return final_result
    
    async def _decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down task into subtasks for different agents"""
        
        task_type = task.get("type", "general")
        subtasks = []
        
        if task_type == "web_automation":
            subtasks = [
                {"type": "planning", "description": "Plan web automation steps"},
                {"type": "web_automation", "description": "Execute web tasks"},
                {"type": "validation", "description": "Validate results"}
            ]
        elif task_type == "code_generation":
            subtasks = [
                {"type": "planning", "description": "Plan code generation"},
                {"type": "code_generation", "description": "Generate code"},
                {"type": "validation", "description": "Validate code"},
                {"type": "optimization", "description": "Optimize code"}
            ]
        elif task_type == "data_analysis":
            subtasks = [
                {"type": "data_analysis", "description": "Analyze data"},
                {"type": "content_creation", "description": "Create report"},
                {"type": "validation", "description": "Validate analysis"}
            ]
        else:
            subtasks = [{"type": "general", "description": task.get("description", "")}]
        
        return subtasks
    
    async def _assign_subtasks(self, subtasks: List[Dict[str, Any]]) -> List[str]:
        """Assign subtasks to appropriate agents"""
        
        assignments = []
        for subtask in subtasks:
            agent_type = subtask.get("type")
            # Find agent with matching type
            for agent in self.registered_agents.values():
                if agent.agent_type.value == agent_type:
                    await agent.add_task(subtask)
                    assignments.append(agent.agent_id)
                    break
        
        return assignments
    
    async def _execute_parallel(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Execute all agents in parallel"""
        
        tasks = []
        for agent_id in agent_ids:
            agent = self.registered_agents.get(agent_id)
            if agent and agent.current_tasks:
                tasks.append(agent.execute(agent.current_tasks[0]))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {agent_id: result for agent_id, result in zip(agent_ids, results)}
    
    async def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from all agents"""
        
        return {
            "status": "completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def handle_message(self, message: AgentMessage):
        """Handle messages from agents"""
        
        if message.message_type == "status_update":
            logger.info(f"Status from {message.sender_id}: {message.content}")
        elif message.message_type == "error":
            logger.error(f"Error from {message.sender_id}: {message.content}")
        elif message.message_type == "request_help":
            # Delegate to another agent
            await self._handle_help_request(message)
    
    async def _handle_help_request(self, message: AgentMessage):
        """Handle agent requesting help from another agent"""
        
        required_capability = message.content.get("required_capability")
        for agent in self.registered_agents.values():
            if required_capability in agent.capabilities.allowed_tools:
                await agent.send_message(
                    message.sender_id,
                    "help_response",
                    {"help": "available"}
                )
                break
    
    def register_agent(self, agent: BaseAgent):
        """Register a specialized agent"""
        self.registered_agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type.value})")


class WebAutomationAgent(BaseAgent):
    """Agent specialized in web automation"""
    
    def __init__(self):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type=AgentType.WEB_AUTOMATION,
            capabilities=AgentCapabilities(
                allowed_tools=["browser_control", "web_scraping", "form_filling"],
                required_permissions=["network_access", "browser_control"]
            )
        )
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web automation task"""
        
        self.state = "executing"
        start_time = datetime.now()
        
        try:
            # Simulate web automation
            result = {
                "status": "success",
                "data": "Web automation completed",
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
            self.performance_metrics["tasks_completed"] += 1
            self.execution_history.append(result)
            
            return result
        
        except Exception as e:
            self.performance_metrics["tasks_failed"] += 1
            return {"status": "failed", "error": str(e)}
        
        finally:
            self.state = "idle"
            if self.current_tasks:
                self.current_tasks.pop(0)
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming messages"""
        logger.info(f"WebAutomationAgent received: {message.message_type}")


class CodeGenerationAgent(BaseAgent):
    """Agent specialized in code generation"""
    
    def __init__(self):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type=AgentType.CODE_GENERATION,
            capabilities=AgentCapabilities(
                allowed_tools=["code_generation", "code_execution", "code_analysis"],
                required_permissions=["code_execution"]
            )
        )
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code generation task"""
        
        self.state = "executing"
        start_time = datetime.now()
        
        try:
            # Simulate code generation
            result = {
                "status": "success",
                "code": "# Generated code",
                "language": "python",
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
            self.performance_metrics["tasks_completed"] += 1
            self.execution_history.append(result)
            
            return result
        
        except Exception as e:
            self.performance_metrics["tasks_failed"] += 1
            return {"status": "failed", "error": str(e)}
        
        finally:
            self.state = "idle"
            if self.current_tasks:
                self.current_tasks.pop(0)
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming messages"""
        logger.info(f"CodeGenerationAgent received: {message.message_type}")


class DataAnalysisAgent(BaseAgent):
    """Agent specialized in data analysis"""
    
    def __init__(self):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type=AgentType.DATA_ANALYSIS,
            capabilities=AgentCapabilities(
                allowed_tools=["data_analysis", "visualization", "statistics"],
                required_permissions=["data_access"]
            )
        )
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data analysis task"""
        
        self.state = "executing"
        start_time = datetime.now()
        
        try:
            # Simulate data analysis
            result = {
                "status": "success",
                "analysis": "Data analysis completed",
                "insights": ["insight1", "insight2"],
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
            self.performance_metrics["tasks_completed"] += 1
            self.execution_history.append(result)
            
            return result
        
        except Exception as e:
            self.performance_metrics["tasks_failed"] += 1
            return {"status": "failed", "error": str(e)}
        
        finally:
            self.state = "idle"
            if self.current_tasks:
                self.current_tasks.pop(0)
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming messages"""
        logger.info(f"DataAnalysisAgent received: {message.message_type}")


class ValidationAgent(BaseAgent):
    """Agent specialized in validation and quality assurance"""
    
    def __init__(self):
        super().__init__(
            agent_id=str(uuid.uuid4()),
            agent_type=AgentType.VALIDATION,
            capabilities=AgentCapabilities(
                allowed_tools=["validation", "testing", "quality_check"],
                required_permissions=["code_execution"]
            )
        )
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation task"""
        
        self.state = "executing"
        start_time = datetime.now()
        
        try:
            # Simulate validation
            result = {
                "status": "success",
                "validation_passed": True,
                "quality_score": 0.95,
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
            self.performance_metrics["tasks_completed"] += 1
            self.execution_history.append(result)
            
            return result
        
        except Exception as e:
            self.performance_metrics["tasks_failed"] += 1
            return {"status": "failed", "error": str(e)}
        
        finally:
            self.state = "idle"
            if self.current_tasks:
                self.current_tasks.pop(0)
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming messages"""
        logger.info(f"ValidationAgent received: {message.message_type}")


async def create_agent_network() -> OrchestratorAgent:
    """Create a network of specialized agents"""
    
    orchestrator = OrchestratorAgent()
    
    # Register specialized agents
    orchestrator.register_agent(WebAutomationAgent())
    orchestrator.register_agent(CodeGenerationAgent())
    orchestrator.register_agent(DataAnalysisAgent())
    orchestrator.register_agent(ValidationAgent())
    
    logger.info("Agent network created with 4 specialized agents")
    return orchestrator


if __name__ == "__main__":
    # Test agent network
    async def test():
        orchestrator = await create_agent_network()
        
        # Execute a task
        task = {
            "id": "task_001",
            "type": "code_generation",
            "description": "Generate a Python function"
        }
        
        result = await orchestrator.execute(task)
        print(json.dumps(result, indent=2, default=str))
        
        # Get status
        print(f"\nOrchestrator status: {orchestrator.get_status()}")
    
    asyncio.run(test())
