"""
Optimized Planning System - Streamlined Version
==============================================

Simplified planner with core LLM planning features:
- Investment decision logic (INDEX_SET/SINGLE_STOCK/NO_TRADE)
- Task mapping and execution
- Plan metadata management
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .planner import Task, TaskType, TaskStatus, Plan


class OptimizedPlanner:
    """Streamlined planner with LLM decision logic"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def create_plan(self, goal: str, context: Dict[str, Any] = None) -> Plan:
        """Create execution plan with LLM decision logic"""
        try:
            plan_id = f"plan_{int(time.time())}"
            context = context or {}
            
            if self.llm_client and hasattr(self.llm_client, 'available') and self.llm_client.available:
                return self._create_llm_plan(goal, context, plan_id)
            else:
                return self._create_simple_plan(goal, context, plan_id)
                
        except Exception as e:
            return self._create_simple_plan(goal, context, f"plan_{int(time.time())}")
    
    def _create_llm_plan(self, goal: str, context: Dict[str, Any], plan_id: str) -> Plan:
        """Create plan using LLM with investment decision logic"""
        try:
            system_prompt = """You are a financial research planner. Create an executable plan.

Decision first:
- Choose one of: ["INDEX_SET", "SINGLE_STOCK", "NO_TRADE"] with rationale.
- If SINGLE_STOCK, propose 3 candidate tickers (Thai market .BK suffix).

Output JSON:
{
  "decision": "INDEX_SET|SINGLE_STOCK|NO_TRADE",
  "why_decision": "string",
  "targets": ["^SETI"|"^SET50"|["PTT.BK","KBANK.BK","ADVANC.BK"]],
  "subtasks": [
    {"id":"fetch_data", "tool":"data.load_prices", "args":{"symbols":[...], "period":"2y"}},
    {"id":"features", "tool":"indicators.compute", "args":{"indicators":["RSI","MACD","EMA_10"]}},
    {"id":"model", "tool":"ml.train_predict", "args":{"algo":"RandomForest", "horizon_days":20}},
    {"id":"risk", "tool":"risk.assess", "args":{"mdd_window":"1y"}},
    {"id":"decide", "tool":"decision.summarize", "args":{}}
  ],
  "acceptance": {"min_sharpe": 0.0, "max_mdd_pct": 20}
}"""

            user_prompt = f"""
USER_GOAL: "{goal}"
CONTEXT: {json.dumps(context, ensure_ascii=False)}
"""

            response = self.llm_client.chat(system_prompt, user_prompt)
            
            if response:
                try:
                    plan_data = json.loads(response)
                    return self._convert_llm_plan_to_tasks(plan_data, goal, context, plan_id)
                except json.JSONDecodeError:
                    pass
        
        except Exception as e:
            pass
        
        # Fallback to simple plan
        return self._create_simple_plan(goal, context, plan_id)
    
    def _convert_llm_plan_to_tasks(self, plan_data: Dict[str, Any], goal: str, context: Dict[str, Any], plan_id: str) -> Plan:
        """Convert LLM plan to task structure"""
        decision = plan_data.get('decision', 'SINGLE_STOCK')
        targets = plan_data.get('targets', ['PTT.BK'])
        subtasks = plan_data.get('subtasks', [])
        
        tasks = []
        
        # Create investment goal analysis task
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
                "acceptance_criteria": plan_data.get('acceptance', {})
            },
            dependencies=[],
            estimated_duration=10.0,
            priority=1
        )
        tasks.append(investment_task)
        
        # Convert subtasks to task format
        for i, subtask in enumerate(subtasks):
            task_id = f"{plan_id}_task_{i+2}"
            tool = subtask.get('tool', '')
            args = subtask.get('args', {})
            
            # Map tool to task type
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
                "acceptance_criteria": plan_data.get('acceptance', {})
            }
        )
    
    def _map_tool_to_task_type(self, tool: str) -> TaskType:
        """Map tool names to task types"""
        tool_mapping = {
            "data.load_prices": TaskType.DATA_FETCH,
            "indicators.compute": TaskType.FEATURE_ENGINEERING,
            "ml.train_predict": TaskType.MODEL_TRAINING,
            "risk.assess": TaskType.RISK_ASSESSMENT,
            "decision.summarize": TaskType.REPORT_GENERATION
        }
        return tool_mapping.get(tool, TaskType.DATA_FETCH)
    
    def _create_simple_plan(self, goal: str, context: Dict[str, Any], plan_id: str) -> Plan:
        """Create simple fallback plan"""
        symbol = context.get('symbol', 'PTT.BK')
        
        tasks = [
            Task(
                id=f"{plan_id}_task_1",
                task_type=TaskType.DATA_FETCH,
                description=f"Fetch price data for {symbol}",
                parameters={"symbols": [symbol], "period": "2y"},
                dependencies=[],
                estimated_duration=10.0,
                priority=1
            ),
            Task(
                id=f"{plan_id}_task_2",
                task_type=TaskType.FEATURE_ENGINEERING,
                description="Calculate technical indicators",
                parameters={"indicators": ["rsi", "sma", "macd"]},
                dependencies=[f"{plan_id}_task_1"],
                estimated_duration=15.0,
                priority=2
            ),
            Task(
                id=f"{plan_id}_task_3",
                task_type=TaskType.MODEL_TRAINING,
                description="Train machine learning model",
                parameters={"model_type": "random_forest"},
                dependencies=[f"{plan_id}_task_2"],
                estimated_duration=20.0,
                priority=3
            ),
            Task(
                id=f"{plan_id}_task_4",
                task_type=TaskType.REPORT_GENERATION,
                description="Generate analysis report",
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
    
    def get_next_task(self, plan: Plan) -> Optional[Task]:
        """Get the next task to execute"""
        try:
            # Find tasks that are ready to execute
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
                    return task
            
            return None
            
        except Exception as e:
            return None
    
    def update_task_status(self, plan: Plan, task_id: str, status: TaskStatus, 
                          result: Dict[str, Any] = None, error: str = None) -> bool:
        """Update task status in plan"""
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
            return False
    
    def get_plan_progress(self, plan: Plan) -> Dict[str, Any]:
        """Get plan execution progress"""
        try:
            total_tasks = len(plan.tasks)
            completed_tasks = sum(1 for t in plan.tasks if t.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED)
            
            return {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "completion_percentage": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
                "is_complete": completed_tasks == total_tasks,
                "has_failures": failed_tasks > 0
            }
            
        except Exception as e:
            return {"error": str(e)}
