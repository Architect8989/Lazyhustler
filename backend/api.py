"""
Ultimate AGI Platform - Complete Backend API
Combines all Manus.im features with self-improving AGI capabilities
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import uuid
from datetime import datetime
import logging

from core.llm_orchestrator import LLMOrchestrator
from agents.agent_framework import AgentNetwork
from learning.agi_learning_engine import AGILearningEngine, TaskExecution, TaskDifficulty
from connectors.connector_manager import ConnectorManager
from reasoning.reasoning_engine import ReasoningEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Ultimate AGI Platform",
    description="Combines Manus.im features with self-improving AGI",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
llm_orchestrator: Optional[LLMOrchestrator] = None
agent_network: Optional[AgentNetwork] = None
learning_engine: Optional[AGILearningEngine] = None
connector_manager: Optional[ConnectorManager] = None
reasoning_engine: Optional[ReasoningEngine] = None

# Request/Response Models

class TaskRequest(BaseModel):
    """Task execution request"""
    description: str
    task_type: str  # code_generation, web_automation, data_analysis, content_creation, etc.
    priority: str = "normal"  # low, normal, high
    timeout: int = 300
    use_connectors: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None


class TaskResponse(BaseModel):
    """Task execution response"""
    task_id: str
    status: str
    result: Optional[str]
    cost: float
    execution_time: float
    confidence: float
    reasoning_steps: List[str]


class ConnectorRequest(BaseModel):
    """Connector registration request"""
    connector_name: str
    credentials: Dict[str, str]


class AGIStatusResponse(BaseModel):
    """AGI status response"""
    intelligence_score: float
    skills_learned: int
    tasks_completed: int
    success_rate: float
    growth_trajectory: List[Dict[str, Any]]
    next_learning_goal: Dict[str, Any]


# Initialization

@app.on_event("startup")
async def startup():
    """Initialize system on startup"""
    global llm_orchestrator, agent_network, learning_engine, connector_manager, reasoning_engine
    
    logger.info("Initializing Ultimate AGI Platform...")
    
    llm_orchestrator = LLMOrchestrator()
    await llm_orchestrator.initialize()
    
    agent_network = AgentNetwork()
    await agent_network.initialize()
    
    learning_engine = AGILearningEngine()
    
    connector_manager = ConnectorManager()
    
    reasoning_engine = ReasoningEngine()
    
    logger.info("✅ System initialized successfully")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global llm_orchestrator, agent_network, connector_manager
    
    if llm_orchestrator:
        await llm_orchestrator.close()
    
    if agent_network:
        await agent_network.close()
    
    if connector_manager:
        await connector_manager.close_all()
    
    logger.info("System shutdown complete")


# Health & Status Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/status")
async def get_system_status():
    """Get system status"""
    
    if not learning_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    summary = learning_engine.get_performance_summary()
    
    return {
        "system": "online",
        "timestamp": datetime.now().isoformat(),
        "performance": summary,
        "llm_models": len(llm_orchestrator.models) if llm_orchestrator else 0,
        "agents": len(agent_network.agents) if agent_network else 0,
        "connectors": len(connector_manager.connectors) if connector_manager else 0
    }


@app.get("/api/agi/status")
async def get_agi_status() -> AGIStatusResponse:
    """Get AGI status and growth"""
    
    if not learning_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    intelligence = learning_engine.get_intelligence_score()
    summary = learning_engine.get_performance_summary()
    trajectory = learning_engine.get_growth_trajectory()
    next_goal = await learning_engine.get_next_learning_goal()
    
    return AGIStatusResponse(
        intelligence_score=intelligence,
        skills_learned=summary.get("skills_learned", 0),
        tasks_completed=summary.get("total_tasks", 0),
        success_rate=summary.get("success_rate", 0.0),
        growth_trajectory=trajectory,
        next_learning_goal=next_goal
    )


# Task Execution Endpoints

@app.post("/api/tasks/execute")
async def execute_task(request: TaskRequest, background_tasks: BackgroundTasks) -> TaskResponse:
    """Execute a task"""
    
    if not llm_orchestrator or not agent_network or not learning_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    task_id = str(uuid.uuid4())
    
    try:
        # Perform reasoning
        reasoning_result = await reasoning_engine.reason(
            goal=request.description,
            task_type=request.task_type
        )
        
        # Select best LLM
        selected_llm = await llm_orchestrator.select_llm(
            task_type=request.task_type,
            requirements={
                "reasoning": 0.9,
                "speed": 0.5,
                "cost": 0.3
            }
        )
        
        # Execute with agents
        start_time = datetime.now()
        
        result = await agent_network.execute_task(
            task_id=task_id,
            description=request.description,
            task_type=request.task_type,
            reasoning=reasoning_result,
            llm=selected_llm,
            connectors=request.use_connectors or []
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Record execution for learning
        execution = TaskExecution(
            task_id=task_id,
            task_description=request.description,
            task_type=request.task_type,
            difficulty=TaskDifficulty.MEDIUM,  # Can be calculated
            status="success" if result.get("status") == "success" else "failed",
            execution_time=execution_time,
            tokens_used=result.get("tokens_used", 0),
            cost=result.get("cost", 0.0),
            result=str(result.get("result", "")),
            reasoning_steps=reasoning_result.get("steps", []),
            decisions_made=result.get("decisions", []),
            errors=result.get("errors", []),
            lessons_learned=[],
            success_rate=result.get("confidence", 0.5),
            model_used=selected_llm.name if selected_llm else "unknown",
            confidence=result.get("confidence", 0.5)
        )
        
        # Learn from execution (async)
        background_tasks.add_task(learning_engine.record_task_execution, execution)
        
        return TaskResponse(
            task_id=task_id,
            status=result.get("status", "unknown"),
            result=result.get("result"),
            cost=result.get("cost", 0.0),
            execution_time=execution_time,
            confidence=result.get("confidence", 0.5),
            reasoning_steps=reasoning_result.get("steps", [])
        )
    
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    
    if not agent_network:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    status = await agent_network.get_task_status(task_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return status


@app.get("/api/tasks")
async def list_tasks(limit: int = 50, offset: int = 0):
    """List recent tasks"""
    
    if not agent_network:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    tasks = await agent_network.list_tasks(limit=limit, offset=offset)
    return {"tasks": tasks, "total": len(tasks)}


# Connector Endpoints

@app.post("/api/connectors/register")
async def register_connector(request: ConnectorRequest):
    """Register a connector"""
    
    if not connector_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    success = await connector_manager.register_connector(
        request.connector_name,
        request.credentials
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to register connector")
    
    return {"status": "registered", "connector": request.connector_name}


@app.get("/api/connectors")
async def list_connectors():
    """List available connectors"""
    
    if not connector_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    available = connector_manager.list_available_connectors()
    registered = connector_manager.get_registered_connectors()
    
    return {
        "available": available,
        "registered": registered,
        "total_available": len(available),
        "total_registered": len(registered)
    }


@app.get("/api/connectors/{connector_name}/status")
async def get_connector_status(connector_name: str):
    """Get connector status"""
    
    if not connector_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    status = connector_manager.get_connector_status(connector_name)
    
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    
    return status


@app.post("/api/connectors/{connector_name}/action")
async def execute_connector_action(
    connector_name: str,
    action: str,
    params: Dict[str, Any]
):
    """Execute connector action"""
    
    if not connector_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    result = await connector_manager.execute_connector_action(
        connector_name,
        action,
        params
    )
    
    return result


# AGI Learning Endpoints

@app.get("/api/agi/skills")
async def get_agi_skills():
    """Get AGI's learned skills"""
    
    if not learning_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    skills = learning_engine.get_all_skills()
    
    return {
        "skills": skills,
        "total_skills": len(skills),
        "average_proficiency": sum(skills.values()) / len(skills) if skills else 0.0
    }


@app.get("/api/agi/intelligence")
async def get_agi_intelligence():
    """Get AGI intelligence score"""
    
    if not learning_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    intelligence = learning_engine.get_intelligence_score()
    summary = learning_engine.get_performance_summary()
    
    return {
        "intelligence_score": intelligence,
        "performance": summary,
        "level": "beginner" if intelligence < 0.3 else "intermediate" if intelligence < 0.6 else "advanced"
    }


@app.get("/api/agi/growth")
async def get_agi_growth():
    """Get AGI growth trajectory"""
    
    if not learning_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    trajectory = learning_engine.get_growth_trajectory()
    
    return {
        "trajectory": trajectory,
        "total_checkpoints": len(trajectory)
    }


# LLM Endpoints

@app.get("/api/llms")
async def list_llms():
    """List available LLMs"""
    
    if not llm_orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    models = await llm_orchestrator.get_available_models()
    
    return {
        "models": models,
        "total": len(models)
    }


# Analytics Endpoints

@app.get("/api/analytics/performance")
async def get_performance_analytics():
    """Get performance analytics"""
    
    if not learning_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    summary = learning_engine.get_performance_summary()
    
    return {
        "analytics": summary,
        "timestamp": datetime.now().isoformat()
    }


# Docs
@app.get("/docs")
async def get_docs():
    """API documentation"""
    return {
        "title": "Ultimate AGI Platform API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/api/status",
            "agi_status": "/api/agi/status",
            "execute_task": "POST /api/tasks/execute",
            "list_tasks": "/api/tasks",
            "list_connectors": "/api/connectors",
            "agi_skills": "/api/agi/skills",
            "agi_intelligence": "/api/agi/intelligence"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
