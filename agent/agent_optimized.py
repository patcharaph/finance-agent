"""
Optimized Finance Agent - Streamlined Version
============================================

Simplified agent with core features:
- LLM Planner with investment decisions
- Think-Act-Evaluate-Reflect loop
- Reflection and learning
- Bilingual reports
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .tools import DataLoader, IndicatorCalculator, ModelOps
from .memory import ShortTermMemory, LongTermMemory
from .evaluator import Evaluator
from .planner import Planner, Plan, Task, TaskType, TaskStatus


@dataclass
class AgentConfig:
    """Simplified agent configuration"""
    max_loops: int = 3
    confidence_threshold: float = 0.7
    enable_llm_planning: bool = True
    storage_path: str = "agent_storage"


class OptimizedFinanceAgent:
    """Streamlined Finance Agent with core features"""
    
    def __init__(self, config: AgentConfig = None, llm_client=None):
        self.config = config or AgentConfig()
        self.llm_client = llm_client
        
        # Initialize core components
        self.tools = {
            'data_loader': DataLoader(),
            'indicator_calculator': IndicatorCalculator(),
            'model_ops': ModelOps()
        }
        
        self.memory = {
            'short_term': ShortTermMemory(),
            'long_term': LongTermMemory(self.config.storage_path)
        }
        
        self.evaluator = Evaluator()
        self.planner = Planner(llm_client)
        
        # State
        self.current_plan = None
        self.loop_count = 0
        self.start_time = None
    
    def run(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main execution method - simplified loop"""
        try:
            self.start_time = time.time()
            context = context or {}
            
            # Store goal in memory
            self.memory['short_term'].store("current_goal", goal)
            self.memory['short_term'].store("execution_context", context)
            
            # Main agent loop
            for loop in range(self.config.max_loops):
                self.loop_count = loop + 1
                
                # Execute one iteration
                result = self._execute_loop(goal, context)
                
                # Check if we should continue
                if not result.get('should_continue', True):
                    break
            
            # Generate final report
            return self._generate_final_report(goal, context)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - self.start_time if self.start_time else 0
            }
    
    def _execute_loop(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one iteration of the agent loop"""
        try:
            # THINK: Create plan or get next task
            if self.current_plan is None:
                self.current_plan = self.planner.create_plan(goal, context)
            
            next_task = self.planner.get_next_task(self.current_plan)
            if next_task is None:
                return {"should_continue": False, "reason": "Plan completed"}
            
            # ACT: Execute the task
            task_result = self._execute_task(next_task, context)
            
            # EVALUATE: Assess results
            evaluation = self._evaluate_task(next_task, task_result)
            
            # REFLECT: Learn from results
            reflection = self._reflect_on_results(next_task, task_result, evaluation)
            
            # Check acceptance criteria
            should_continue = self._should_continue(evaluation, reflection)
            
            return {
                "success": evaluation.get('success', False),
                "should_continue": should_continue,
                "evaluation": evaluation,
                "reflection": reflection
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "should_continue": False}
    
    def _execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task - simplified"""
        try:
            # Update task status
            self.planner.update_task_status(self.current_plan, task.id, TaskStatus.IN_PROGRESS)
            
            # Route to appropriate tool
            if task.task_type == TaskType.DATA_FETCH:
                symbol = task.parameters.get('symbols', ['PTT.BK'])[0]
                result = self.tools['data_loader'].fetch_price_data(symbol, "2y")
                if result.success:
                    self.memory['short_term'].store("price_data", result.data)
                    return {"success": True, "data": result.data}
                else:
                    return {"success": False, "error": result.error}
            
            elif task.task_type == TaskType.FEATURE_ENGINEERING:
                price_data = self.memory['short_term'].retrieve("price_data")
                if price_data is None:
                    return {"success": False, "error": "No price data available"}
                
                indicators = task.parameters.get('indicators', ['rsi', 'sma', 'macd'])
                result = self.tools['indicator_calculator'].calculate_indicators(price_data, indicators)
                
                if result.success:
                    self.memory['short_term'].store("indicators_data", result.data)
                    return {"success": True, "indicators": result.data}
                else:
                    return {"success": False, "error": result.error}
            
            elif task.task_type == TaskType.MODEL_TRAINING:
                indicators_data = self.memory['short_term'].retrieve("indicators_data")
                if indicators_data is None:
                    return {"success": False, "error": "No indicators data available"}
                
                # Create features
                features_result = self.tools['indicator_calculator'].create_features(
                    indicators_data, context.get('horizon', 5)
                )
                
                if features_result.success:
                    # Train model
                    X, y = features_result.data['X'], features_result.data['y']
                    model_result = self.tools['model_ops'].train_model(X, y, 'random_forest')
                    
                    if model_result.success:
                        self.memory['short_term'].store("trained_model", model_result.data)
                        return {"success": True, "model": model_result.data}
                    else:
                        return {"success": False, "error": model_result.error}
                else:
                    return {"success": False, "error": features_result.error}
            
            elif task.task_type == TaskType.MODEL_EVALUATION:
                model_data = self.memory['short_term'].retrieve("trained_model")
                if model_data is None:
                    return {"success": False, "error": "No trained model available"}
                
                # Evaluate model
                y_test, y_pred = model_data['y_test'], model_data['y_pred']
                y_naive = [y_test.iloc[0]] + y_test.iloc[:-1].tolist()
                
                report = self.evaluator.evaluate_model_performance(y_test.values, y_pred, y_naive)
                self.memory['short_term'].store("model_evaluation", report.to_dict())
                
                return {
                    "success": report.result.value != "fail",
                    "evaluation_report": report.to_dict()
                }
            
            elif task.task_type == TaskType.REPORT_GENERATION:
                return self._generate_report(context)
            
            else:
                return {"success": True, "message": f"Task {task.task_type.value} completed"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _evaluate_task(self, task: Task, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate task execution result - simplified"""
        success = result.get('success', False)
        
        if not success:
            return {"success": False, "confidence": 0.0, "error": result.get('error')}
        
        # Task-specific evaluation
        if task.task_type == TaskType.MODEL_EVALUATION:
            evaluation = result.get('evaluation_report', {})
            return {
                "success": evaluation.get('result') != 'fail',
                "confidence": evaluation.get('confidence', 0.5),
                "metrics": evaluation.get('metrics', {})
            }
        
        return {"success": True, "confidence": 0.8}
    
    def _reflect_on_results(self, task: Task, result: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on results and learn - simplified"""
        try:
            confidence = evaluation.get('confidence', 0.0)
            
            # Only reflect if confidence is low or task failed
            if confidence >= self.config.confidence_threshold and evaluation.get('success', False):
                return {"insights": [], "suggestions": []}
            
            # Use LLM for reflection if available
            if self.llm_client and hasattr(self.llm_client, 'available') and self.llm_client.available:
                return self._llm_reflect(task, result, evaluation)
            
            # Basic reflection
            insights = []
            suggestions = []
            
            if not evaluation.get('success', False):
                insights.append("Task execution failed")
                suggestions.append("Check data quality and parameters")
            
            if confidence < self.config.confidence_threshold:
                insights.append("Low confidence in results")
                suggestions.append("Consider additional validation or different approach")
            
            return {"insights": insights, "suggestions": suggestions}
            
        except Exception as e:
            return {"insights": [], "suggestions": [], "error": str(e)}
    
    def _llm_reflect(self, task: Task, result: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """LLM-powered reflection - simplified"""
        try:
            goal = self.memory['short_term'].retrieve("current_goal", "")
            context = self.memory['short_term'].retrieve("execution_context", {})
            
            system_prompt = """You are a reflection analyst. Analyze the task execution and provide insights.

Output JSON:
{
  "insights": ["insight1", "insight2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "lesson": {
    "pattern": "market condition pattern",
    "action": "recommended action",
    "example": "example scenario"
  }
}"""

            user_prompt = f"""
Goal: {goal}
Task: {task.task_type.value} - {task.description}
Result: {json.dumps(result, default=str, ensure_ascii=False)}
Evaluation: {json.dumps(evaluation, default=str, ensure_ascii=False)}
Context: {json.dumps(context, ensure_ascii=False)}
"""

            response = self.llm_client.chat(system_prompt, user_prompt)
            
            if response:
                try:
                    reflection = json.loads(response)
                    
                    # Store lesson in long-term memory
                    if reflection.get('lesson'):
                        lesson = reflection['lesson']
                        self.memory['long_term'].store_lesson(
                            lesson_type="reflection_analysis",
                            description=f"Pattern: {lesson.get('pattern', '')} | Action: {lesson.get('action', '')}",
                            context={"goal": goal, "task_type": task.task_type.value},
                            confidence=evaluation.get('confidence', 0.5)
                        )
                    
                    return reflection
                    
                except json.JSONDecodeError:
                    return {"insights": [response], "suggestions": [], "lesson": {}}
            
            return {"insights": [], "suggestions": [], "lesson": {}}
            
        except Exception as e:
            return {"insights": [], "suggestions": [], "lesson": {}, "error": str(e)}
    
    def _should_continue(self, evaluation: Dict[str, Any], reflection: Dict[str, Any]) -> bool:
        """Determine if agent should continue - simplified"""
        # Check if we've reached max loops
        if self.loop_count >= self.config.max_loops:
            return False
        
        # Check confidence threshold
        confidence = evaluation.get('confidence', 0.0)
        if confidence >= self.config.confidence_threshold:
            return False
        
        # Check if reflection suggests stopping
        suggestions = reflection.get('suggestions', [])
        if any("stop" in suggestion.lower() for suggestion in suggestions):
            return False
        
        return True
    
    def _generate_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final report - simplified"""
        try:
            # Collect data
            price_data = self.memory['short_term'].retrieve("price_data")
            evaluation = self.memory['short_term'].retrieve("model_evaluation")
            goal = self.memory['short_term'].retrieve("current_goal")
            
            # Basic report
            report = {
                "timestamp": datetime.now().isoformat(),
                "goal": goal,
                "symbol": context.get('symbol', 'Unknown'),
                "model_performance": evaluation or {},
                "confidence": evaluation.get('confidence', 0.5) if evaluation else 0.5,
                "data_summary": {
                    "price_data_points": len(price_data) if price_data is not None else 0,
                    "last_price": float(price_data['close'].iloc[-1]) if price_data is not None else None
                }
            }
            
            # Generate LLM explanation if available
            if self.llm_client and hasattr(self.llm_client, 'available') and self.llm_client.available:
                explanation = self._generate_llm_explanation(report, context)
                report['llm_explanation'] = explanation
            
            return {"success": True, "report": report}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_llm_explanation(self, report: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM explanation - simplified"""
        try:
            system_prompt = """You are a financial report writer. Write a concise report for Thai investors.

Output JSON:
{
  "executive_summary_th": "สรุปผู้บริหาร (ภาษาไทย)",
  "executive_summary_en": "Executive Summary (English)",
  "reasons_evidence_th": ["เหตุผล 1", "เหตุผล 2"],
  "risk_caveats_th": ["ความเสี่ยง 1", "ความเสี่ยง 2"],
  "investment_recommendation": "BUY/HOLD/WAIT with reasoning",
  "confidence_level": "High/Medium/Low"
}"""

            user_prompt = f"""
Analysis Report: {json.dumps(report, default=str, ensure_ascii=False)}
Context: {json.dumps(context, ensure_ascii=False)}
"""

            response = self.llm_client.chat(system_prompt, user_prompt)
            
            if response:
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    return {
                        "executive_summary_th": response,
                        "executive_summary_en": "See Thai summary",
                        "reasons_evidence_th": ["ดูผลการวิเคราะห์"],
                        "risk_caveats_th": ["ความเสี่ยงจากการลงทุน"],
                        "investment_recommendation": "HOLD - ดูผลการวิเคราะห์",
                        "confidence_level": "Medium"
                    }
            
            return {
                "executive_summary_th": "การวิเคราะห์เสร็จสิ้น",
                "executive_summary_en": "Analysis completed",
                "reasons_evidence_th": ["ข้อมูลจากการวิเคราะห์"],
                "risk_caveats_th": ["ความเสี่ยงจากการลงทุน"],
                "investment_recommendation": "HOLD - ดูผลการวิเคราะห์",
                "confidence_level": "Medium"
            }
            
        except Exception as e:
            return {
                "executive_summary_th": "การวิเคราะห์เสร็จสิ้น แต่เกิดข้อผิดพลาด",
                "executive_summary_en": "Analysis completed with errors",
                "reasons_evidence_th": ["ดูผลการวิเคราะห์"],
                "risk_caveats_th": ["ความเสี่ยงจากการลงทุน"],
                "investment_recommendation": "HOLD - ดูผลการวิเคราะห์",
                "confidence_level": "Low"
            }
    
    def _generate_final_report(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final execution summary - simplified"""
        try:
            final_report = self.memory['short_term'].retrieve("final_report")
            
            return {
                "success": final_report is not None,
                "goal": goal,
                "context": context,
                "execution_time": time.time() - self.start_time if self.start_time else 0,
                "loops_completed": self.loop_count,
                "final_report": final_report,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - self.start_time if self.start_time else 0,
                "loops_completed": self.loop_count
            }
