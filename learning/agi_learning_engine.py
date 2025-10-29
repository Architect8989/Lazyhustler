"""
Self-Improving AGI Learning Engine
Learns from every task execution and grows smarter over time
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import sqlite3
import numpy as np
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskDifficulty(Enum):
    """Task difficulty levels"""
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


@dataclass
class TaskExecution:
    """Record of a task execution"""
    task_id: str
    task_description: str
    task_type: str
    difficulty: TaskDifficulty
    status: str  # success, partial, failed
    execution_time: float  # seconds
    tokens_used: int
    cost: float
    result: str
    reasoning_steps: List[str]
    decisions_made: List[Dict[str, Any]]
    errors: List[str]
    lessons_learned: List[str]
    success_rate: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    model_used: str = ""
    confidence: float = 0.5


@dataclass
class SkillProfile:
    """AGI's skill profile"""
    skill_name: str
    proficiency: float  # 0.0 to 1.0
    tasks_completed: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    avg_cost: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    examples: List[str] = field(default_factory=list)


@dataclass
class KnowledgeNode:
    """Knowledge graph node"""
    node_id: str
    concept: str
    description: str
    related_concepts: List[str] = field(default_factory=list)
    confidence: float = 0.5
    times_used: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class AGILearningEngine:
    """Self-improving AGI learning system"""
    
    def __init__(self, db_path: str = "agi_learning.db"):
        self.db_path = db_path
        self.task_history: List[TaskExecution] = []
        self.skill_profiles: Dict[str, SkillProfile] = {}
        self.knowledge_graph: Dict[str, KnowledgeNode] = {}
        self.meta_learnings: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        self._init_database()
        self._load_from_database()
    
    def _init_database(self):
        """Initialize learning database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Task execution history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_executions (
                task_id TEXT PRIMARY KEY,
                task_description TEXT,
                task_type TEXT,
                difficulty INTEGER,
                status TEXT,
                execution_time REAL,
                tokens_used INTEGER,
                cost REAL,
                result TEXT,
                reasoning_steps TEXT,
                decisions_made TEXT,
                errors TEXT,
                lessons_learned TEXT,
                success_rate REAL,
                timestamp TIMESTAMP,
                model_used TEXT,
                confidence REAL
            )
        """)
        
        # Skill profiles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_profiles (
                skill_name TEXT PRIMARY KEY,
                proficiency REAL,
                tasks_completed INTEGER,
                success_rate REAL,
                avg_execution_time REAL,
                avg_cost REAL,
                last_updated TIMESTAMP,
                examples TEXT
            )
        """)
        
        # Knowledge graph
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                node_id TEXT PRIMARY KEY,
                concept TEXT,
                description TEXT,
                related_concepts TEXT,
                confidence REAL,
                times_used INTEGER,
                success_rate REAL,
                created_at TIMESTAMP
            )
        """)
        
        # Meta-learnings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meta_learnings (
                id INTEGER PRIMARY KEY,
                learning_type TEXT,
                content TEXT,
                confidence REAL,
                created_at TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Learning database initialized: {self.db_path}")
    
    async def record_task_execution(self, execution: TaskExecution) -> None:
        """Record a task execution and learn from it"""
        
        self.task_history.append(execution)
        
        # Update skill profile
        await self._update_skill_profile(execution)
        
        # Extract lessons
        lessons = await self._extract_lessons(execution)
        
        # Update knowledge graph
        await self._update_knowledge_graph(execution)
        
        # Perform meta-learning
        await self._perform_meta_learning(execution, lessons)
        
        # Store in database
        self._store_execution(execution)
        
        logger.info(f"Task recorded: {execution.task_id} (Status: {execution.status})")
    
    async def _update_skill_profile(self, execution: TaskExecution) -> None:
        """Update AGI's skill profile based on execution"""
        
        skill_name = execution.task_type
        
        if skill_name not in self.skill_profiles:
            self.skill_profiles[skill_name] = SkillProfile(
                skill_name=skill_name,
                proficiency=0.1  # Start low
            )
        
        profile = self.skill_profiles[skill_name]
        
        # Update metrics
        profile.tasks_completed += 1
        profile.last_updated = datetime.now()
        
        # Update success rate
        if execution.status == "success":
            profile.success_rate = (
                (profile.success_rate * (profile.tasks_completed - 1) + 1.0) /
                profile.tasks_completed
            )
        else:
            profile.success_rate = (
                (profile.success_rate * (profile.tasks_completed - 1) + 0.0) /
                profile.tasks_completed
            )
        
        # Update proficiency (0.0 to 1.0)
        # Proficiency = success_rate * (1 - execution_time_penalty) * confidence
        time_penalty = min(execution.execution_time / 300, 1.0)  # Normalize to 300s
        profile.proficiency = min(
            profile.success_rate * (1 - time_penalty * 0.2) * execution.confidence,
            1.0
        )
        
        # Update execution time average
        profile.avg_execution_time = (
            (profile.avg_execution_time * (profile.tasks_completed - 1) + execution.execution_time) /
            profile.tasks_completed
        )
        
        # Update cost average
        profile.avg_cost = (
            (profile.avg_cost * (profile.tasks_completed - 1) + execution.cost) /
            profile.tasks_completed
        )
        
        # Store example
        if execution.status == "success" and len(profile.examples) < 5:
            profile.examples.append(execution.task_description)
        
        logger.info(f"Updated skill: {skill_name} (Proficiency: {profile.proficiency:.2f})")
    
    async def _extract_lessons(self, execution: TaskExecution) -> List[str]:
        """Extract lessons from task execution"""
        
        lessons = execution.lessons_learned.copy()
        
        # Automatic lesson extraction
        if execution.status == "success":
            lessons.append(f"Successfully completed {execution.task_type} tasks")
            if execution.confidence > 0.8:
                lessons.append(f"High confidence approach for {execution.task_type}")
        
        elif execution.status == "failed":
            if execution.errors:
                lessons.append(f"Error pattern: {execution.errors[0]}")
            lessons.append(f"Need to improve {execution.task_type} approach")
        
        # Time-based lessons
        if execution.execution_time < 5:
            lessons.append("Fast execution achieved")
        elif execution.execution_time > 60:
            lessons.append("Slow execution - optimize approach")
        
        # Cost-based lessons
        if execution.cost < 0.01:
            lessons.append("Cost-efficient execution")
        elif execution.cost > 1.0:
            lessons.append("High cost - consider cheaper models")
        
        return lessons
    
    async def _update_knowledge_graph(self, execution: TaskExecution) -> None:
        """Update knowledge graph from task execution"""
        
        # Create nodes for task type
        task_node_id = f"task_{execution.task_type}"
        if task_node_id not in self.knowledge_graph:
            self.knowledge_graph[task_node_id] = KnowledgeNode(
                node_id=task_node_id,
                concept=execution.task_type,
                description=f"Task type: {execution.task_type}"
            )
        
        node = self.knowledge_graph[task_node_id]
        node.times_used += 1
        
        # Update success rate
        if execution.status == "success":
            node.success_rate = (
                (node.success_rate * (node.times_used - 1) + 1.0) /
                node.times_used
            )
        else:
            node.success_rate = (
                (node.success_rate * (node.times_used - 1) + 0.0) /
                node.times_used
            )
        
        # Update confidence
        node.confidence = min(node.success_rate * (node.times_used / 10), 1.0)
        
        # Extract concepts from reasoning steps
        for step in execution.reasoning_steps:
            concept_id = f"concept_{hash(step) % 10000}"
            if concept_id not in self.knowledge_graph:
                self.knowledge_graph[concept_id] = KnowledgeNode(
                    node_id=concept_id,
                    concept=step[:50],
                    description=step
                )
            
            # Link concepts
            if concept_id not in node.related_concepts:
                node.related_concepts.append(concept_id)
    
    async def _perform_meta_learning(
        self,
        execution: TaskExecution,
        lessons: List[str]
    ) -> None:
        """Perform meta-learning (learning about learning)"""
        
        # Analyze patterns across tasks
        if len(self.task_history) >= 10:
            # Calculate success trend
            recent_tasks = self.task_history[-10:]
            recent_success = sum(1 for t in recent_tasks if t.status == "success") / 10
            
            meta_learning = {
                "type": "success_trend",
                "value": recent_success,
                "timestamp": datetime.now().isoformat(),
                "insight": f"Recent success rate: {recent_success:.1%}"
            }
            self.meta_learnings.append(meta_learning)
            
            # Identify best performing models
            model_performance = {}
            for task in self.task_history:
                if task.model_used not in model_performance:
                    model_performance[task.model_used] = {"success": 0, "total": 0}
                
                model_performance[task.model_used]["total"] += 1
                if task.status == "success":
                    model_performance[task.model_used]["success"] += 1
            
            best_model = max(
                model_performance.items(),
                key=lambda x: x[1]["success"] / max(x[1]["total"], 1)
            )
            
            meta_learning = {
                "type": "best_model",
                "value": best_model[0],
                "success_rate": best_model[1]["success"] / max(best_model[1]["total"], 1),
                "timestamp": datetime.now().isoformat()
            }
            self.meta_learnings.append(meta_learning)
        
        # Identify patterns in errors
        if execution.errors:
            error_pattern = {
                "type": "error_pattern",
                "error": execution.errors[0],
                "task_type": execution.task_type,
                "frequency": sum(1 for t in self.task_history if execution.errors[0] in t.errors),
                "timestamp": datetime.now().isoformat()
            }
            self.meta_learnings.append(error_pattern)
    
    def _store_execution(self, execution: TaskExecution) -> None:
        """Store execution in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO task_executions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution.task_id,
            execution.task_description,
            execution.task_type,
            execution.difficulty.value,
            execution.status,
            execution.execution_time,
            execution.tokens_used,
            execution.cost,
            execution.result,
            json.dumps(execution.reasoning_steps),
            json.dumps(execution.decisions_made),
            json.dumps(execution.errors),
            json.dumps(execution.lessons_learned),
            execution.success_rate,
            execution.timestamp.isoformat(),
            execution.model_used,
            execution.confidence
        ))
        
        conn.commit()
        conn.close()
    
    def _load_from_database(self) -> None:
        """Load learning history from database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load skill profiles
            cursor.execute("SELECT * FROM skill_profiles")
            for row in cursor.fetchall():
                self.skill_profiles[row[0]] = SkillProfile(
                    skill_name=row[0],
                    proficiency=row[1],
                    tasks_completed=row[2],
                    success_rate=row[3],
                    avg_execution_time=row[4],
                    avg_cost=row[5],
                    last_updated=datetime.fromisoformat(row[6]),
                    examples=json.loads(row[7])
                )
            
            conn.close()
            logger.info(f"Loaded {len(self.skill_profiles)} skill profiles")
        
        except Exception as e:
            logger.warning(f"Could not load from database: {e}")
    
    def get_skill_proficiency(self, task_type: str) -> float:
        """Get AGI's proficiency in a task type"""
        
        if task_type in self.skill_profiles:
            return self.skill_profiles[task_type].proficiency
        return 0.0
    
    def get_all_skills(self) -> Dict[str, float]:
        """Get all skills and proficiencies"""
        
        return {
            name: profile.proficiency
            for name, profile in self.skill_profiles.items()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get AGI performance summary"""
        
        if not self.task_history:
            return {
                "total_tasks": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "total_cost": 0.0
            }
        
        successful = sum(1 for t in self.task_history if t.status == "success")
        total_time = sum(t.execution_time for t in self.task_history)
        total_cost = sum(t.cost for t in self.task_history)
        
        return {
            "total_tasks": len(self.task_history),
            "successful_tasks": successful,
            "success_rate": successful / len(self.task_history),
            "avg_execution_time": total_time / len(self.task_history),
            "total_cost": total_cost,
            "avg_cost": total_cost / len(self.task_history),
            "skills_learned": len(self.skill_profiles),
            "knowledge_nodes": len(self.knowledge_graph),
            "meta_learnings": len(self.meta_learnings)
        }
    
    def get_growth_trajectory(self) -> List[Dict[str, Any]]:
        """Get AGI's growth trajectory over time"""
        
        trajectory = []
        
        # Group tasks by time periods
        if not self.task_history:
            return trajectory
        
        # Calculate metrics for each 10-task window
        for i in range(0, len(self.task_history), 10):
            window = self.task_history[i:i+10]
            successful = sum(1 for t in window if t.status == "success")
            
            trajectory.append({
                "task_number": i + 10,
                "success_rate": successful / len(window),
                "avg_time": sum(t.execution_time for t in window) / len(window),
                "avg_cost": sum(t.cost for t in window) / len(window),
                "confidence": sum(t.confidence for t in window) / len(window)
            })
        
        return trajectory
    
    async def get_next_learning_goal(self) -> Dict[str, Any]:
        """Determine next learning goal for AGI"""
        
        # Find weakest skill
        if self.skill_profiles:
            weakest_skill = min(
                self.skill_profiles.items(),
                key=lambda x: x[1].proficiency
            )
            
            return {
                "goal_type": "improve_skill",
                "skill": weakest_skill[0],
                "current_proficiency": weakest_skill[1].proficiency,
                "target_proficiency": min(weakest_skill[1].proficiency + 0.2, 1.0),
                "reason": f"Improve {weakest_skill[0]} from {weakest_skill[1].proficiency:.2f}"
            }
        
        # Default: explore new skills
        return {
            "goal_type": "explore",
            "reason": "Explore new task types"
        }
    
    def get_intelligence_score(self) -> float:
        """Calculate AGI's overall intelligence score (0.0 to 1.0)"""
        
        if not self.skill_profiles:
            return 0.0
        
        # Average proficiency across all skills
        avg_proficiency = sum(p.proficiency for p in self.skill_profiles.values()) / len(self.skill_profiles)
        
        # Bonus for number of skills learned
        skill_bonus = min(len(self.skill_profiles) / 20, 0.2)
        
        # Bonus for meta-learnings
        meta_bonus = min(len(self.meta_learnings) / 100, 0.1)
        
        # Bonus for knowledge graph
        knowledge_bonus = min(len(self.knowledge_graph) / 100, 0.1)
        
        intelligence = min(avg_proficiency + skill_bonus + meta_bonus + knowledge_bonus, 1.0)
        
        return intelligence


if __name__ == "__main__":
    async def test():
        engine = AGILearningEngine()
        
        # Simulate task execution
        execution = TaskExecution(
            task_id="task_001",
            task_description="Generate Python web scraper",
            task_type="code_generation",
            difficulty=TaskDifficulty.MEDIUM,
            status="success",
            execution_time=12.5,
            tokens_used=1500,
            cost=0.05,
            result="Generated working web scraper",
            reasoning_steps=["Analyze requirements", "Design scraper", "Generate code"],
            decisions_made=[{"decision": "use_beautifulsoup", "confidence": 0.9}],
            errors=[],
            lessons_learned=["BeautifulSoup is effective for web scraping"],
            success_rate=0.95,
            model_used="gpt-4",
            confidence=0.85
        )
        
        await engine.record_task_execution(execution)
        
        # Get performance
        summary = engine.get_performance_summary()
        print(f"Performance: {json.dumps(summary, indent=2)}")
        
        # Get intelligence score
        intelligence = engine.get_intelligence_score()
        print(f"Intelligence Score: {intelligence:.2f}")
    
    asyncio.run(test())
