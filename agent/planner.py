"""
Planning System for Finance Agent
=================================

This module provides intelligent planning capabilities:
- Task decomposition and planning
- Plan generation and optimization
- Dynamic plan adjustment based on results
- Integration with LLM for intelligent planning
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


class TaskType(Enum):
    """Types of tasks the agent can perform"""
    DATA_FETCH = "data_fetch"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    PREDICTION = "prediction"
    REPORT_GENERATION = "report_generation"
    PARAMETER_TUNING = "parameter_tuning"
    DATA_QUALITY_CHECK = "data_quality_check"
    INVESTMENT_GOAL_ANALYSIS = "investment_goal_analysis"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Individual task in a plan"""
    id: str
    task_type: TaskType
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str]  # Task IDs this task depends on
    estimated_duration: float  # Estimated duration in seconds
    priority: int  # Higher number = higher priority
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['task_type'] = self.task_type.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        data['task_type'] = TaskType(data['task_type'])
        data['status'] = TaskStatus(data['status'])
        return cls(**data)


@dataclass
class Plan:
    """Complete execution plan"""
    id: str
    goal: str
    tasks: List[Task]
    created_at: float
    updated_at: float
    status: str = "created"
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'goal': self.goal,
            'tasks': [task.to_dict() for task in self.tasks],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'status': self.status,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Plan':
        tasks = [Task.from_dict(task_data) for task_data in data['tasks']]
        return cls(
            id=data['id'],
            goal=data['goal'],
            tasks=tasks,
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            status=data['status'],
            metadata=data.get('metadata', {})
        )


class Planner:
    """
    Intelligent planning system for finance analysis tasks
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize planner
        
        Args:
            llm_client: LLM client for intelligent planning (optional)
        """
        self.llm_client = llm_client
        self.available_tools = {
            "fetch_price_data": "Fetch historical price data for a symbol",
            "fetch_news_sentiment": "Fetch and analyze news sentiment",
            "calculate_indicators": "Calculate technical indicators",
            "create_features": "Create feature matrix for ML",
            "train_model": "Train machine learning model",
            "evaluate_model": "Evaluate model performance",
            "assess_risk": "Perform risk assessment",
            "generate_prediction": "Generate price predictions",
            "tune_parameters": "Tune model parameters",
            "check_data_quality": "Check data quality and completeness"
        }
        
        # Default task templates
        self.task_templates = {
            "basic_analysis": [
                TaskType.DATA_FETCH,
                TaskType.DATA_QUALITY_CHECK,
                TaskType.FEATURE_ENGINEERING,
                TaskType.MODEL_TRAINING,
                TaskType.MODEL_EVALUATION,
                TaskType.PREDICTION,
                TaskType.REPORT_GENERATION
            ],
            "comprehensive_analysis": [
                TaskType.DATA_FETCH,
                TaskType.DATA_QUALITY_CHECK,
                TaskType.FEATURE_ENGINEERING,
                TaskType.SENTIMENT_ANALYSIS,
                TaskType.MODEL_TRAINING,
                TaskType.MODEL_EVALUATION,
                TaskType.RISK_ASSESSMENT,
                TaskType.PREDICTION,
                TaskType.REPORT_GENERATION
            ],
            "model_optimization": [
                TaskType.DATA_FETCH,
                TaskType.FEATURE_ENGINEERING,
                TaskType.MODEL_TRAINING,
                TaskType.MODEL_EVALUATION,
                TaskType.PARAMETER_TUNING,
                TaskType.MODEL_TRAINING,  # Retrain with tuned parameters
                TaskType.MODEL_EVALUATION,
                TaskType.PREDICTION,
                TaskType.REPORT_GENERATION
            ]
        }
    
    def create_plan(self, goal: str, context: Dict[str, Any] = None, 
                   plan_type: str = "basic_analysis") -> Plan:
        """
        Create an execution plan for a given goal
        
        Args:
            goal: Goal description
            context: Additional context for planning
            plan_type: Type of plan to create
        
        Returns:
            Plan object with tasks
        """
        try:
            plan_id = f"plan_{int(time.time())}"
            context = context or {}
            
            if self.llm_client and hasattr(self.llm_client, 'available') and self.llm_client.available:
                # Use LLM for intelligent planning
                plan = self._create_llm_plan(goal, context, plan_id)
            else:
                # Use template-based planning
                plan = self._create_template_plan(goal, context, plan_id, plan_type)
            
            return plan
            
        except Exception as e:
            # Fallback to basic plan
            return self._create_fallback_plan(goal, context, f"plan_{int(time.time())}")
    
    def _create_llm_plan(self, goal: str, context: Dict[str, Any], plan_id: str) -> Plan:
        """Create plan using LLM with enhanced investment decision logic"""
        try:
            # Use the new LLM Planner prompt for investment decisions
            system_prompt = """You are a financial research planner for the "Finance Agent".
Given a USER GOAL and CONTEXT, produce an executable plan for the agent.

Decision first:
- Choose one of: ["INDEX_SET", "SINGLE_STOCK", "NO_TRADE"] with rationale.
- If SINGLE_STOCK, propose 3 candidate tickers with reasons (Thai market suffix .BK).

Constraints & Criteria:
- Focus on Thai market (Yahoo Finance symbols, e.g., ^SETI, ^SET50, PTT.BK).
- time_horizon: short (5-10d) / medium (1-3m) / long (6-12m).
- Min quality gate to proceed: Sharpe > 0 (backtest mini) AND MDD within user's tolerance if provided.
- Respect confidence_threshold from config.

Output MUST be valid JSON following PLAN_SCHEMA and do not add extra text."""

            user_prompt = f"""
USER_GOAL: "{goal}"
CONTEXT: {json.dumps(context, ensure_ascii=False)}

PLAN_SCHEMA:
{{
  "decision": "INDEX_SET|SINGLE_STOCK|NO_TRADE",
  "why_decision": "string",
  "targets": ["^SETI"|"^SET50"|["PTT.BK","KBANK.BK","ADVANC.BK"]],
  "subtasks": [
    {{"id":"fetch_data", "tool":"data.load_prices", "args":{{"symbols":[...], "period":"2y", "interval":"1d"}}}},
    {{"id":"features", "tool":"indicators.compute", "args":{{"indicators":["RSI","MACD","EMA_10","EMA_20","BBANDS"]}}}},
    {{"id":"model", "tool":"ml.train_predict", "args":{{"algo":"RandomForest", "horizon_days":20}}}},
    {{"id":"risk", "tool":"risk.assess", "args":{{"mdd_window":"1y"}}}},
    {{"id":"mini_backtest", "tool":"backtest.quick", "args":{{"metric":["Sharpe","MDD"]}}}},
    {{"id":"decide", "tool":"decision.summarize", "args":{{}}}}
  ],
  "acceptance": {{"min_sharpe": 0.0, "max_mdd_pct": 20}},
  "report_needs": ["executive_summary","rationale_th_en","disclaimer"]
}}
"""

            response = self.llm_client.chat(system_prompt, user_prompt)
            
            if response:
                try:
                    plan_data = json.loads(response)
                    
                    # Convert LLM plan to our task structure
                    tasks = []
                    decision = plan_data.get('decision', 'SINGLE_STOCK')
                    targets = plan_data.get('targets', ['PTT.BK'])
                    subtasks = plan_data.get('subtasks', [])
                    
                    # Create investment goal analysis task first
                    task_id = f"{plan_id}_task_1"
                    investment_task = Task(
                        id=task_id,
                        task_type=TaskType.INVESTMENT_GOAL_ANALYSIS,
                        description=f"Analyze investment goal: {decision}",
                        parameters={
                            "investment_goal": goal,
                            "analysis_type": decision.lower(),
                            "targets": targets,
                            "decision_rationale": plan_data.get('why_decision', ''),
                            "acceptance_criteria": plan_data.get('acceptance', {}),
                            "report_needs": plan_data.get('report_needs', [])
                        },
                        dependencies=[],
                        estimated_duration=10.0,
                        priority=1
                    )
                    tasks.append(investment_task)
                    
                    # Convert subtasks to our task format
                    for i, subtask in enumerate(subtasks):
                        task_id = f"{plan_id}_task_{i+2}"
                        tool = subtask.get('tool', '')
                        args = subtask.get('args', {})
                        
                        # Map tool names to task types
                        task_type = self._map_tool_to_task_type(tool)
                        
                        task = Task(
                            id=task_id,
                            task_type=task_type,
                            description=subtask.get('id', f"Execute {tool}"),
                            parameters=args,
                            dependencies=[f"{plan_id}_task_1"] if i == 0 else [f"{plan_id}_task_{i+1}"],
                            estimated_duration=30.0,
                            priority=i+2
                        )
                        tasks.append(task)
                    
                    return Plan(
                        id=plan_id,
                        goal=goal,
                        tasks=tasks,
                        created_at=time.time(),
                        updated_at=time.time(),
                        metadata={
                            **context,
                            "llm_decision": decision,
                            "llm_targets": targets,
                            "llm_rationale": plan_data.get('why_decision', ''),
                            "acceptance_criteria": plan_data.get('acceptance', {}),
                            "report_needs": plan_data.get('report_needs', [])
                        }
                    )
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse LLM response as JSON: {str(e)}")
                    print(f"Response: {response}")
        
        except Exception as e:
            print(f"LLM planning failed: {str(e)}")
        
        # Fallback to template planning
        return self._create_template_plan(goal, context, plan_id, "basic_analysis")
    
    def _map_tool_to_task_type(self, tool: str) -> TaskType:
        """Map tool names to task types"""
        tool_mapping = {
            "data.load_prices": TaskType.DATA_FETCH,
            "indicators.compute": TaskType.FEATURE_ENGINEERING,
            "ml.train_predict": TaskType.MODEL_TRAINING,
            "risk.assess": TaskType.RISK_ASSESSMENT,
            "backtest.quick": TaskType.MODEL_EVALUATION,
            "decision.summarize": TaskType.REPORT_GENERATION
        }
        return tool_mapping.get(tool, TaskType.DATA_FETCH)
    
    def _create_template_plan(self, goal: str, context: Dict[str, Any], 
                            plan_id: str, plan_type: str) -> Plan:
        """Create plan using predefined templates"""
        try:
            # Determine plan type based on goal
            if "comprehensive" in goal.lower() or "full" in goal.lower():
                template = self.task_templates["comprehensive_analysis"]
            elif "optimize" in goal.lower() or "tune" in goal.lower():
                template = self.task_templates["model_optimization"]
            else:
                template = self.task_templates["basic_analysis"]
            
            # Create tasks from template
            tasks = []
            symbol = context.get('symbol', 'PTT.BK')
            horizon = context.get('horizon', 5)
            
            for i, task_type in enumerate(template):
                task_id = f"{plan_id}_task_{i+1}"
                
                # Set up task parameters based on type
                if task_type == TaskType.DATA_FETCH:
                    task = Task(
                        id=task_id,
                        task_type=task_type,
                        description=f"Fetch price data for {symbol}",
                        parameters={"symbol": symbol, "period": "2y"},
                        dependencies=[],
                        estimated_duration=10.0,
                        priority=1
                    )
                
                elif task_type == TaskType.DATA_QUALITY_CHECK:
                    task = Task(
                        id=task_id,
                        task_type=task_type,
                        description="Check data quality and completeness",
                        parameters={},
                        dependencies=[f"{plan_id}_task_1"],
                        estimated_duration=5.0,
                        priority=2
                    )
                
                elif task_type == TaskType.FEATURE_ENGINEERING:
                    task = Task(
                        id=task_id,
                        task_type=task_type,
                        description="Calculate technical indicators and create features",
                        parameters={"indicators": ["rsi", "sma", "macd", "bollinger"]},
                        dependencies=[f"{plan_id}_task_1"],
                        estimated_duration=15.0,
                        priority=2
                    )
                
                elif task_type == TaskType.SENTIMENT_ANALYSIS:
                    task = Task(
                        id=task_id,
                        task_type=task_type,
                        description="Analyze news sentiment",
                        parameters={"symbol": symbol, "days_back": 7},
                        dependencies=[f"{plan_id}_task_1"],
                        estimated_duration=10.0,
                        priority=2
                    )
                
                elif task_type == TaskType.MODEL_TRAINING:
                    task = Task(
                        id=task_id,
                        task_type=task_type,
                        description="Train machine learning model",
                        parameters={"model_type": "random_forest", "target_horizon": horizon},
                        dependencies=[f"{plan_id}_task_3"],
                        estimated_duration=20.0,
                        priority=3
                    )
                
                elif task_type == TaskType.MODEL_EVALUATION:
                    task = Task(
                        id=task_id,
                        task_type=task_type,
                        description="Evaluate model performance",
                        parameters={"test_size": 0.2},
                        dependencies=[f"{plan_id}_task_{i}"],  # Depends on previous training task
                        estimated_duration=10.0,
                        priority=3
                    )
                
                elif task_type == TaskType.PARAMETER_TUNING:
                    task = Task(
                        id=task_id,
                        task_type=task_type,
                        description="Tune model parameters for better performance",
                        parameters={"param_grid": {"n_estimators": [100, 200, 300]}},
                        dependencies=[f"{plan_id}_task_{i}"],  # Depends on previous evaluation
                        estimated_duration=30.0,
                        priority=3
                    )
                
                elif task_type == TaskType.RISK_ASSESSMENT:
                    task = Task(
                        id=task_id,
                        task_type=task_type,
                        description="Perform risk assessment",
                        parameters={"confidence_level": 0.95},
                        dependencies=[f"{plan_id}_task_{i}"],  # Depends on model evaluation
                        estimated_duration=10.0,
                        priority=4
                    )
                
                elif task_type == TaskType.PREDICTION:
                    task = Task(
                        id=task_id,
                        task_type=task_type,
                        description="Generate price predictions",
                        parameters={"horizon": horizon},
                        dependencies=[f"{plan_id}_task_{i}"],  # Depends on model training/evaluation
                        estimated_duration=5.0,
                        priority=4
                    )
                
                elif task_type == TaskType.REPORT_GENERATION:
                    task = Task(
                        id=task_id,
                        task_type=task_type,
                        description="Generate analysis report",
                        parameters={"format": "comprehensive"},
                        dependencies=[f"{plan_id}_task_{i}"],  # Depends on prediction
                        estimated_duration=15.0,
                        priority=5
                    )
                
                else:
                    # Generic task
                    task = Task(
                        id=task_id,
                        task_type=task_type,
                        description=f"Execute {task_type.value}",
                        parameters={},
                        dependencies=[],
                        estimated_duration=10.0,
                        priority=1
                    )
                
                tasks.append(task)
            
            return Plan(
                id=plan_id,
                goal=goal,
                tasks=tasks,
                created_at=time.time(),
                updated_at=time.time(),
                metadata=context
            )
            
        except Exception as e:
            print(f"Template planning failed: {str(e)}")
            return self._create_fallback_plan(goal, context, plan_id)
    
    def _create_fallback_plan(self, goal: str, context: Dict[str, Any], plan_id: str) -> Plan:
        """Create minimal fallback plan"""
        symbol = context.get('symbol', 'PTT.BK')
        
        tasks = [
            Task(
                id=f"{plan_id}_task_1",
                task_type=TaskType.DATA_FETCH,
                description=f"Fetch price data for {symbol}",
                parameters={"symbol": symbol, "period": "2y"},
                dependencies=[],
                estimated_duration=10.0,
                priority=1
            ),
            Task(
                id=f"{plan_id}_task_2",
                task_type=TaskType.FEATURE_ENGINEERING,
                description="Create features for analysis",
                parameters={},
                dependencies=[f"{plan_id}_task_1"],
                estimated_duration=15.0,
                priority=2
            ),
            Task(
                id=f"{plan_id}_task_3",
                task_type=TaskType.MODEL_TRAINING,
                description="Train basic model",
                parameters={},
                dependencies=[f"{plan_id}_task_2"],
                estimated_duration=20.0,
                priority=3
            ),
            Task(
                id=f"{plan_id}_task_4",
                task_type=TaskType.REPORT_GENERATION,
                description="Generate basic report",
                parameters={},
                dependencies=[f"{plan_id}_task_3"],
                estimated_duration=10.0,
                priority=4
            )
        ]
        
        return Plan(
            id=plan_id,
            goal=goal,
            tasks=tasks,
            created_at=time.time(),
            updated_at=time.time(),
            metadata=context
        )
    
    def optimize_plan(self, plan: Plan, constraints: Dict[str, Any] = None) -> Plan:
        """
        Optimize plan based on constraints and priorities
        
        Args:
            plan: Plan to optimize
            constraints: Optimization constraints
        
        Returns:
            Optimized plan
        """
        try:
            constraints = constraints or {}
            max_duration = constraints.get('max_duration', 300)  # 5 minutes default
            max_tasks = constraints.get('max_tasks', 10)
            
            # Sort tasks by priority and dependencies
            optimized_tasks = self._sort_tasks_by_priority(plan.tasks)
            
            # Remove tasks if plan is too long
            total_duration = sum(task.estimated_duration for task in optimized_tasks)
            if total_duration > max_duration:
                optimized_tasks = self._trim_tasks_by_duration(optimized_tasks, max_duration)
            
            # Limit number of tasks
            if len(optimized_tasks) > max_tasks:
                optimized_tasks = optimized_tasks[:max_tasks]
            
            # Update plan
            plan.tasks = optimized_tasks
            plan.updated_at = time.time()
            plan.status = "optimized"
            
            return plan
            
        except Exception as e:
            print(f"Plan optimization failed: {str(e)}")
            return plan
    
    def _sort_tasks_by_priority(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks by priority and dependencies"""
        # Create dependency graph
        task_dict = {task.id: task for task in tasks}
        sorted_tasks = []
        visited = set()
        
        def visit(task_id: str):
            if task_id in visited:
                return
            visited.add(task_id)
            
            task = task_dict[task_id]
            # Visit dependencies first
            for dep_id in task.dependencies:
                if dep_id in task_dict:
                    visit(dep_id)
            
            sorted_tasks.append(task)
        
        # Visit all tasks
        for task in tasks:
            visit(task.id)
        
        # Sort by priority within each dependency level
        return sorted(sorted_tasks, key=lambda t: t.priority, reverse=True)
    
    def _trim_tasks_by_duration(self, tasks: List[Task], max_duration: float) -> List[Task]:
        """Remove tasks to fit within duration constraint"""
        total_duration = 0
        trimmed_tasks = []
        
        for task in tasks:
            if total_duration + task.estimated_duration <= max_duration:
                trimmed_tasks.append(task)
                total_duration += task.estimated_duration
            else:
                break
        
        return trimmed_tasks
    
    def get_next_task(self, plan: Plan) -> Optional[Task]:
        """
        Get the next task to execute based on dependencies and status
        
        Args:
            plan: Current plan
        
        Returns:
            Next task to execute, or None if no tasks available
        """
        try:
            # Find tasks that are ready to execute
            ready_tasks = []
            
            for task in plan.tasks:
                if task.status != TaskStatus.PENDING:
                    continue
                
                # Check if all dependencies are completed
                dependencies_met = True
                for dep_id in task.dependencies:
                    dep_task = next((t for t in plan.tasks if t.id == dep_id), None)
                    if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    ready_tasks.append(task)
            
            if not ready_tasks:
                return None
            
            # Return highest priority task
            return max(ready_tasks, key=lambda t: t.priority)
            
        except Exception as e:
            print(f"Error getting next task: {str(e)}")
            return None
    
    def update_task_status(self, plan: Plan, task_id: str, status: TaskStatus, 
                          result: Dict[str, Any] = None, error: str = None) -> bool:
        """
        Update task status in plan
        
        Args:
            plan: Plan containing the task
            task_id: ID of task to update
            status: New status
            result: Task result (if completed)
            error: Error message (if failed)
        
        Returns:
            True if task was updated, False otherwise
        """
        try:
            task = next((t for t in plan.tasks if t.id == task_id), None)
            if task is None:
                return False
            
            task.status = status
            task.result = result
            task.error = error
            
            if status == TaskStatus.IN_PROGRESS:
                task.start_time = time.time()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task.end_time = time.time()
            
            plan.updated_at = time.time()
            return True
            
        except Exception as e:
            print(f"Error updating task status: {str(e)}")
            return False
    
    def get_plan_progress(self, plan: Plan) -> Dict[str, Any]:
        """
        Get plan execution progress
        
        Args:
            plan: Plan to analyze
        
        Returns:
            Progress information
        """
        try:
            total_tasks = len(plan.tasks)
            completed_tasks = sum(1 for t in plan.tasks if t.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED)
            in_progress_tasks = sum(1 for t in plan.tasks if t.status == TaskStatus.IN_PROGRESS)
            pending_tasks = sum(1 for t in plan.tasks if t.status == TaskStatus.PENDING)
            
            # Calculate estimated completion time
            remaining_tasks = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
            estimated_remaining_time = sum(t.estimated_duration for t in remaining_tasks)
            
            # Calculate actual vs estimated time for completed tasks
            completed_with_time = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED and t.start_time and t.end_time]
            actual_time = sum(t.end_time - t.start_time for t in completed_with_time)
            estimated_time = sum(t.estimated_duration for t in completed_with_time)
            
            return {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "in_progress_tasks": in_progress_tasks,
                "pending_tasks": pending_tasks,
                "completion_percentage": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
                "estimated_remaining_time": estimated_remaining_time,
                "actual_vs_estimated_ratio": (actual_time / estimated_time) if estimated_time > 0 else 1.0,
                "plan_status": plan.status,
                "is_complete": completed_tasks == total_tasks,
                "has_failures": failed_tasks > 0
            }
            
        except Exception as e:
            print(f"Error calculating plan progress: {str(e)}")
            return {"error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    print("Testing Planning System...")
    
    # Test planner
    planner = Planner()
    
    # Create a plan
    goal = "Analyze PTT.BK stock and provide investment recommendation"
    context = {"symbol": "PTT.BK", "horizon": 5, "period": "2y"}
    
    plan = planner.create_plan(goal, context, "comprehensive_analysis")
    print(f"✅ Created plan with {len(plan.tasks)} tasks")
    print(f"   Goal: {plan.goal}")
    print(f"   Tasks: {[t.task_type.value for t in plan.tasks]}")
    
    # Test plan optimization
    optimized_plan = planner.optimize_plan(plan, {"max_duration": 120, "max_tasks": 5})
    print(f"✅ Optimized plan: {len(optimized_plan.tasks)} tasks")
    
    # Test getting next task
    next_task = planner.get_next_task(plan)
    if next_task:
        print(f"✅ Next task: {next_task.description}")
    
    # Test progress tracking
    progress = planner.get_plan_progress(plan)
    print(f"✅ Plan progress: {progress['completion_percentage']:.1f}% complete")
    
    print("Planning system testing completed!")
