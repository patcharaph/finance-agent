"""
Main Finance Agent
==================

This is the core agent that orchestrates all components:
- Planning: Creates and manages execution plans
- Acting: Executes tasks using available tools
- Evaluating: Assesses results and performance
- Reflecting: Learns from results and adjusts strategy
- Memory: Stores and retrieves learned knowledge
"""

import os
import time
import json
import warnings
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

import numpy as np
import pandas as pd

# Import our custom modules
from .tools import DataLoader, IndicatorCalculator, SentimentAnalyzer, ModelOps
from .memory import ShortTermMemory, LongTermMemory
from .evaluator import Evaluator, EvaluationResult
from .planner import Planner, Plan, Task, TaskType, TaskStatus

warnings.filterwarnings("ignore")


@dataclass
class AgentConfig:
    """Configuration for the Finance Agent"""
    max_loops: int = 3
    max_execution_time: float = 300.0  # 5 minutes
    confidence_threshold: float = 0.7
    enable_llm_planning: bool = True
    enable_learning: bool = True
    log_level: str = "INFO"
    storage_path: str = "agent_storage"


@dataclass
class AgentState:
    """Current state of the agent"""
    current_plan: Optional[Plan] = None
    current_task: Optional[Task] = None
    execution_history: List[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = None
    learning_enabled: bool = True
    
    def __post_init__(self):
        if self.execution_history is None:
            self.execution_history = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


class FinanceAgent:
    """
    Main Finance Agent with planning, acting, evaluating, and reflecting capabilities
    """
    
    def __init__(self, config: AgentConfig = None, llm_client=None, logger: Callable = None):
        """
        Initialize the Finance Agent
        
        Args:
            config: Agent configuration
            llm_client: LLM client for intelligent operations
            logger: Logging function
        """
        self.config = config or AgentConfig()
        self.llm_client = llm_client
        self.logger = logger or self._default_logger
        
        # Initialize components
        self.tools = {
            'data_loader': DataLoader(),
            'indicator_calculator': IndicatorCalculator(),
            'sentiment_analyzer': SentimentAnalyzer(),
            'model_ops': ModelOps()
        }
        
        self.memory = {
            'short_term': ShortTermMemory(),
            'long_term': LongTermMemory(self.config.storage_path)
        }
        
        self.evaluator = Evaluator()
        self.planner = Planner(llm_client)
        
        # Agent state
        self.state = AgentState(learning_enabled=self.config.enable_learning)
        
        # Execution tracking
        self.start_time = None
        self.loop_count = 0
        
        self.log("INFO", "Finance Agent initialized", {
            "config": asdict(self.config),
            "tools_available": list(self.tools.keys()),
            "memory_enabled": self.config.enable_learning
        })
    
    def run(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main execution method - runs the complete agent loop
        
        Args:
            goal: Goal description
            context: Additional context
        
        Returns:
            Final results and analysis
        """
        try:
            self.start_time = time.time()
            context = context or {}
            
            self.log("INFO", "Starting agent execution", {
                "goal": goal,
                "context": context,
                "max_loops": self.config.max_loops
            })
            
            # Store goal in memory
            self.memory['short_term'].store("current_goal", goal, {"timestamp": time.time()})
            self.memory['short_term'].store("execution_context", context, {"timestamp": time.time()})
            
            # Main agent loop
            for loop in range(self.config.max_loops):
                self.loop_count = loop + 1
                
                self.log("INFO", f"Starting loop {self.loop_count}", {
                    "loop": self.loop_count,
                    "max_loops": self.config.max_loops
                })
                
                # Check execution time limit
                if time.time() - self.start_time > self.config.max_execution_time:
                    self.log("WARNING", "Execution time limit reached", {
                        "elapsed_time": time.time() - self.start_time,
                        "max_time": self.config.max_execution_time
                    })
                    break
                
                # Execute one iteration of the agent loop
                result = self._execute_agent_loop(goal, context)
                
                # Check if we should continue
                if result.get('should_continue', True) == False:
                    self.log("INFO", "Agent decided to stop execution", {
                        "reason": result.get('reason', 'Unknown'),
                        "loop": self.loop_count
                    })
                    break
                
                # Store loop results
                self.state.execution_history.append({
                    "loop": self.loop_count,
                    "result": result,
                    "timestamp": time.time()
                })
                
                # Small delay between loops
                time.sleep(0.1)
            
            # Generate final summary
            final_result = self._generate_final_summary(goal, context)
            
            self.log("INFO", "Agent execution completed", {
                "total_loops": self.loop_count,
                "execution_time": time.time() - self.start_time,
                "final_result_keys": list(final_result.keys())
            })
            
            return final_result
            
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            self.log("ERROR", error_msg, {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "loop": self.loop_count
            })
            
            return {
                "success": False,
                "error": error_msg,
                "execution_time": time.time() - self.start_time if self.start_time else 0,
                "loops_completed": self.loop_count
            }
    
    def _execute_agent_loop(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one iteration of the agent loop (Think-Act-Evaluate-Reflect)
        
        Args:
            goal: Current goal
            context: Execution context
        
        Returns:
            Loop execution result
        """
        try:
            # THINK: Plan or get next task
            if self.state.current_plan is None:
                self.log("INFO", "Creating new execution plan", {"goal": goal})
                self.state.current_plan = self.planner.create_plan(goal, context)
                
                if not self.state.current_plan or not self.state.current_plan.tasks:
                    return {
                        "success": False,
                        "error": "Failed to create execution plan",
                        "should_continue": False
                    }
            
            # Get next task to execute
            next_task = self.planner.get_next_task(self.state.current_plan)
            
            if next_task is None:
                # No more tasks, check if plan is complete
                progress = self.planner.get_plan_progress(self.state.current_plan)
                if progress.get('is_complete', False):
                    return {
                        "success": True,
                        "message": "Plan completed successfully",
                        "should_continue": False,
                        "progress": progress
                    }
                else:
                    return {
                        "success": False,
                        "error": "No tasks available and plan not complete",
                        "should_continue": False
                    }
            
            # ACT: Execute the task
            self.log("INFO", f"Executing task: {next_task.description}", {
                "task_id": next_task.id,
                "task_type": next_task.task_type.value
            })
            
            self.state.current_task = next_task
            self.planner.update_task_status(self.state.current_plan, next_task.id, TaskStatus.IN_PROGRESS)
            
            task_result = self._execute_task(next_task, context)
            
            # EVALUATE: Assess task results
            evaluation_result = self._evaluate_task_result(next_task, task_result)
            
            # Update task status based on evaluation
            if evaluation_result.get('success', False):
                self.planner.update_task_status(
                    self.state.current_plan, next_task.id, TaskStatus.COMPLETED, task_result
                )
            else:
                self.planner.update_task_status(
                    self.state.current_plan, next_task.id, TaskStatus.FAILED, 
                    error=evaluation_result.get('error', 'Task evaluation failed')
                )
            
            # REFLECT: Learn from results and adjust
            reflection_result = self._reflect_on_results(next_task, task_result, evaluation_result)
            
            # Store learning in memory
            if self.state.learning_enabled:
                self._store_learning(next_task, task_result, evaluation_result, reflection_result)
            
            return {
                "success": evaluation_result.get('success', False),
                "task_result": task_result,
                "evaluation": evaluation_result,
                "reflection": reflection_result,
                "should_continue": True,
                "progress": self.planner.get_plan_progress(self.state.current_plan)
            }
            
        except Exception as e:
            error_msg = f"Agent loop execution failed: {str(e)}"
            self.log("ERROR", error_msg, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            return {
                "success": False,
                "error": error_msg,
                "should_continue": False
            }
    
    def _execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific task
        
        Args:
            task: Task to execute
            context: Execution context
        
        Returns:
            Task execution result
        """
        try:
            self.log("INFO", f"Executing {task.task_type.value}", {
                "task_id": task.id,
                "parameters": task.parameters
            })
            
            # Route to appropriate tool based on task type
            if task.task_type == TaskType.DATA_FETCH:
                return self._execute_data_fetch(task, context)
            
            elif task.task_type == TaskType.DATA_QUALITY_CHECK:
                return self._execute_data_quality_check(task, context)
            
            elif task.task_type == TaskType.FEATURE_ENGINEERING:
                return self._execute_feature_engineering(task, context)
            
            elif task.task_type == TaskType.SENTIMENT_ANALYSIS:
                return self._execute_sentiment_analysis(task, context)
            
            elif task.task_type == TaskType.MODEL_TRAINING:
                return self._execute_model_training(task, context)
            
            elif task.task_type == TaskType.MODEL_EVALUATION:
                return self._execute_model_evaluation(task, context)
            
            elif task.task_type == TaskType.PARAMETER_TUNING:
                return self._execute_parameter_tuning(task, context)
            
            elif task.task_type == TaskType.RISK_ASSESSMENT:
                return self._execute_risk_assessment(task, context)
            
            elif task.task_type == TaskType.PREDICTION:
                return self._execute_prediction(task, context)
            
            elif task.task_type == TaskType.REPORT_GENERATION:
                return self._execute_report_generation(task, context)
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown task type: {task.task_type.value}"
                }
                
        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            self.log("ERROR", error_msg, {
                "task_id": task.id,
                "task_type": task.task_type.value,
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": error_msg
            }
    
    def _execute_data_fetch(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data fetching task"""
        try:
            symbol = task.parameters.get('symbol', context.get('symbol', 'PTT.BK'))
            period = task.parameters.get('period', '2y')
            
            result = self.tools['data_loader'].fetch_price_data(symbol, period)
            
            if result.success:
                # Store data in short-term memory
                self.memory['short_term'].store("price_data", result.data, result.metadata)
                return {
                    "success": True,
                    "data": result.data,
                    "metadata": result.metadata
                }
            else:
                return {
                    "success": False,
                    "error": result.error
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_data_quality_check(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data quality check task"""
        try:
            price_data = self.memory['short_term'].retrieve("price_data")
            if price_data is None:
                return {"success": False, "error": "No price data available for quality check"}
            
            report = self.evaluator.evaluate_data_quality(price_data)
            
            return {
                "success": report.result != EvaluationResult.FAIL,
                "report": report.to_dict(),
                "data_quality": report.result.value
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_feature_engineering(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature engineering task"""
        try:
            price_data = self.memory['short_term'].retrieve("price_data")
            if price_data is None:
                return {"success": False, "error": "No price data available for feature engineering"}
            
            indicators = task.parameters.get('indicators', ['rsi', 'sma', 'macd'])
            
            # Calculate indicators
            indicators_result = self.tools['indicator_calculator'].calculate_indicators(
                price_data, indicators
            )
            
            if not indicators_result.success:
                return {"success": False, "error": indicators_result.error}
            
            # Create features
            target_horizon = task.parameters.get('target_horizon', context.get('horizon', 5))
            features_result = self.tools['indicator_calculator'].create_features(
                indicators_result.data, target_horizon
            )
            
            if not features_result.success:
                return {"success": False, "error": features_result.error}
            
            # Store results in memory
            self.memory['short_term'].store("features_data", features_result.data, features_result.metadata)
            self.memory['short_term'].store("indicators_data", indicators_result.data, indicators_result.metadata)
            
            return {
                "success": True,
                "features": features_result.data,
                "indicators": indicators_result.data,
                "metadata": features_result.metadata
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_sentiment_analysis(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sentiment analysis task"""
        try:
            symbol = task.parameters.get('symbol', context.get('symbol', 'PTT.BK'))
            days_back = task.parameters.get('days_back', 7)
            
            result = self.tools['data_loader'].fetch_news_sentiment(symbol, days_back)
            
            if result.success:
                self.memory['short_term'].store("sentiment_data", result.data, result.metadata)
                return {
                    "success": True,
                    "sentiment_data": result.data,
                    "metadata": result.metadata
                }
            else:
                return {
                    "success": False,
                    "error": result.error
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_model_training(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training task"""
        try:
            features_data = self.memory['short_term'].retrieve("features_data")
            if features_data is None:
                return {"success": False, "error": "No features data available for model training"}
            
            X = features_data['X']
            y = features_data['y']
            
            model_type = task.parameters.get('model_type', 'random_forest')
            model_params = task.parameters.get('model_params', {})
            
            result = self.tools['model_ops'].train_model(X, y, model_type, **model_params)
            
            if result.success:
                self.memory['short_term'].store("trained_model", result.data, result.metadata)
                return {
                    "success": True,
                    "model": result.data,
                    "metrics": result.metadata
                }
            else:
                return {
                    "success": False,
                    "error": result.error
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_model_evaluation(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model evaluation task"""
        try:
            model_data = self.memory['short_term'].retrieve("trained_model")
            if model_data is None:
                return {"success": False, "error": "No trained model available for evaluation"}
            
            # Extract test data and predictions
            y_test = model_data['y_test']
            y_pred = model_data['y_pred']
            
            # Create naive baseline
            y_naive = np.roll(y_test.values, 1)
            y_naive[0] = y_test.iloc[0]
            
            # Evaluate model
            report = self.evaluator.evaluate_model_performance(y_test.values, y_pred, y_naive)
            
            # Store evaluation results
            self.memory['short_term'].store("model_evaluation", report.to_dict(), {
                "timestamp": time.time(),
                "task_id": task.id
            })
            
            return {
                "success": report.result != EvaluationResult.FAIL,
                "evaluation_report": report.to_dict(),
                "passed": report.result == EvaluationResult.PASS
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_parameter_tuning(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parameter tuning task"""
        try:
            # Get current model performance
            evaluation = self.memory['short_term'].retrieve("model_evaluation")
            if evaluation is None:
                return {"success": False, "error": "No model evaluation available for parameter tuning"}
            
            # Simple parameter tuning logic
            current_rel_performance = evaluation['metrics']['rel_performance']
            
            if current_rel_performance > 0.98:  # Model worse than naive
                # Suggest parameter changes
                suggestions = {
                    "n_estimators": 500,  # Increase for Random Forest
                    "min_samples_leaf": 2,  # Increase regularization
                    "max_depth": 10  # Limit depth
                }
                
                self.memory['short_term'].store("parameter_suggestions", suggestions, {
                    "reason": "Model performance below threshold",
                    "current_rel_performance": current_rel_performance
                })
                
                return {
                    "success": True,
                    "suggestions": suggestions,
                    "reason": "Model performance below threshold"
                }
            else:
                return {
                    "success": True,
                    "suggestions": {},
                    "reason": "Model performance is acceptable"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_risk_assessment(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk assessment task"""
        try:
            # Get model evaluation
            evaluation = self.memory['short_term'].retrieve("model_evaluation")
            if evaluation is None:
                return {"success": False, "error": "No model evaluation available for risk assessment"}
            
            # Get market data
            price_data = self.memory['short_term'].retrieve("price_data")
            if price_data is None:
                return {"success": False, "error": "No price data available for risk assessment"}
            
            # Calculate basic risk metrics
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            max_drawdown = self._calculate_max_drawdown(price_data['close'])
            
            # Assess risk level
            risk_level = "LOW"
            if volatility > 0.3 or abs(max_drawdown) > 0.2:
                risk_level = "HIGH"
            elif volatility > 0.2 or abs(max_drawdown) > 0.1:
                risk_level = "MEDIUM"
            
            risk_assessment = {
                "volatility": float(volatility),
                "max_drawdown": float(max_drawdown),
                "risk_level": risk_level,
                "model_confidence": evaluation.get('confidence', 0.5)
            }
            
            self.memory['short_term'].store("risk_assessment", risk_assessment, {
                "timestamp": time.time(),
                "task_id": task.id
            })
            
            return {
                "success": True,
                "risk_assessment": risk_assessment
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_prediction(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute prediction task"""
        try:
            model_data = self.memory['short_term'].retrieve("trained_model")
            if model_data is None:
                return {"success": False, "error": "No trained model available for prediction"}
            
            # Get latest features for prediction
            features_data = self.memory['short_term'].retrieve("features_data")
            if features_data is None:
                return {"success": False, "error": "No features data available for prediction"}
            
            # Use last few samples for prediction
            X_latest = features_data['X'].tail(5)  # Last 5 samples
            
            model = model_data['model']
            scaler = model_data['scaler']
            
            result = self.tools['model_ops'].predict(model, scaler, X_latest)
            
            if result.success:
                predictions = result.data
                
                # Store predictions
                self.memory['short_term'].store("predictions", predictions, {
                    "timestamp": time.time(),
                    "horizon": task.parameters.get('horizon', 5),
                    "samples": len(predictions)
                })
                
                return {
                    "success": True,
                    "predictions": predictions.tolist(),
                    "metadata": result.metadata
                }
            else:
                return {
                    "success": False,
                    "error": result.error
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_report_generation(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute report generation task"""
        try:
            # Collect all available data
            price_data = self.memory['short_term'].retrieve("price_data")
            evaluation = self.memory['short_term'].retrieve("model_evaluation")
            risk_assessment = self.memory['short_term'].retrieve("risk_assessment")
            predictions = self.memory['short_term'].retrieve("predictions")
            
            # Generate comprehensive report
            report = {
                "timestamp": datetime.now().isoformat(),
                "goal": self.memory['short_term'].retrieve("current_goal"),
                "symbol": context.get('symbol', 'Unknown'),
                "data_summary": {
                    "price_data_points": len(price_data) if price_data is not None else 0,
                    "last_price": float(price_data['close'].iloc[-1]) if price_data is not None else None,
                    "data_period": f"{price_data.index[0].date()} to {price_data.index[-1].date()}" if price_data is not None else None
                },
                "model_performance": evaluation if evaluation else {},
                "risk_assessment": risk_assessment if risk_assessment else {},
                "predictions": {
                    "values": predictions.tolist() if predictions is not None else [],
                    "horizon": task.parameters.get('horizon', 5)
                },
                "recommendations": self._generate_recommendations(evaluation, risk_assessment, predictions),
                "confidence": evaluation.get('confidence', 0.5) if evaluation else 0.5
            }
            
            # Store final report
            self.memory['short_term'].store("final_report", report, {
                "timestamp": time.time(),
                "task_id": task.id
            })
            
            return {
                "success": True,
                "report": report
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _evaluate_task_result(self, task: Task, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate task execution result"""
        try:
            success = result.get('success', False)
            
            if not success:
                return {
                    "success": False,
                    "error": result.get('error', 'Task execution failed'),
                    "confidence": 0.0
                }
            
            # Task-specific evaluation
            if task.task_type == TaskType.MODEL_EVALUATION:
                evaluation = result.get('evaluation_report', {})
                confidence = evaluation.get('confidence', 0.5)
                passed = evaluation.get('passed', False)
                
                return {
                    "success": passed,
                    "confidence": confidence,
                    "metrics": evaluation.get('metrics', {}),
                    "reasoning": evaluation.get('reasoning', '')
                }
            
            elif task.task_type == TaskType.DATA_QUALITY_CHECK:
                data_quality = result.get('data_quality', 'fail')
                return {
                    "success": data_quality != 'fail',
                    "confidence": 0.8 if data_quality == 'pass' else 0.4,
                    "data_quality": data_quality
                }
            
            else:
                # Default evaluation for other tasks
                return {
                    "success": True,
                    "confidence": 0.8,
                    "message": "Task completed successfully"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Evaluation failed: {str(e)}",
                "confidence": 0.0
            }
    
    def _reflect_on_results(self, task: Task, result: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on task results and generate insights"""
        try:
            reflection = {
                "task_type": task.task_type.value,
                "success": evaluation.get('success', False),
                "confidence": evaluation.get('confidence', 0.0),
                "insights": [],
                "suggestions": []
            }
            
            # Task-specific reflection
            if task.task_type == TaskType.MODEL_EVALUATION:
                metrics = evaluation.get('metrics', {})
                rel_performance = metrics.get('rel_performance', 1.0)
                
                if rel_performance > 0.98:
                    reflection["insights"].append("Model performs worse than naive baseline")
                    reflection["suggestions"].append("Consider feature engineering or different model type")
                elif rel_performance < 0.9:
                    reflection["insights"].append("Model shows good performance improvement")
                    reflection["suggestions"].append("Model is ready for prediction tasks")
                else:
                    reflection["insights"].append("Model performance is marginal")
                    reflection["suggestions"].append("Consider parameter tuning or additional features")
            
            elif task.task_type == TaskType.DATA_QUALITY_CHECK:
                data_quality = evaluation.get('data_quality', 'unknown')
                if data_quality == 'fail':
                    reflection["insights"].append("Data quality issues detected")
                    reflection["suggestions"].append("Address data quality before proceeding")
                else:
                    reflection["insights"].append("Data quality is acceptable")
            
            # General reflection
            if evaluation.get('confidence', 0) < 0.6:
                reflection["insights"].append("Low confidence in results")
                reflection["suggestions"].append("Consider additional validation or data")
            
            return reflection
            
        except Exception as e:
            return {
                "error": f"Reflection failed: {str(e)}",
                "insights": [],
                "suggestions": []
            }
    
    def _store_learning(self, task: Task, result: Dict[str, Any], 
                       evaluation: Dict[str, Any], reflection: Dict[str, Any]) -> None:
        """Store learning in long-term memory"""
        try:
            if not self.state.learning_enabled:
                return
            
            # Store performance metrics
            if task.task_type == TaskType.MODEL_EVALUATION and result.get('success'):
                symbol = self.memory['short_term'].retrieve("execution_context", {}).get('symbol', 'Unknown')
                metrics = evaluation.get('metrics', {})
                
                self.memory['long_term'].record_performance(
                    symbol=symbol,
                    model_type="random_forest",  # Default for now
                    metrics=metrics,
                    context={
                        "task_id": task.id,
                        "timestamp": time.time(),
                        "confidence": evaluation.get('confidence', 0.0)
                    }
                )
            
            # Store lessons learned
            if reflection.get('insights'):
                self.memory['long_term'].store_lesson(
                    lesson_type=f"{task.task_type.value}_insight",
                    description="; ".join(reflection['insights']),
                    context={
                        "task_id": task.id,
                        "success": evaluation.get('success', False),
                        "confidence": evaluation.get('confidence', 0.0)
                    },
                    confidence=evaluation.get('confidence', 0.5)
                )
            
        except Exception as e:
            self.log("WARNING", f"Failed to store learning: {str(e)}")
    
    def _generate_final_summary(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final execution summary"""
        try:
            # Get final report
            final_report = self.memory['short_term'].retrieve("final_report")
            
            # Get execution statistics
            progress = self.planner.get_plan_progress(self.state.current_plan) if self.state.current_plan else {}
            
            # Get memory statistics
            memory_stats = {
                "short_term": self.memory['short_term'].get_session_info(),
                "long_term": self.memory['long_term'].get_memory_stats()
            }
            
            summary = {
                "success": final_report is not None,
                "goal": goal,
                "context": context,
                "execution_time": time.time() - self.start_time if self.start_time else 0,
                "loops_completed": self.loop_count,
                "plan_progress": progress,
                "final_report": final_report,
                "memory_stats": memory_stats,
                "execution_history": self.state.execution_history,
                "timestamp": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate final summary: {str(e)}",
                "execution_time": time.time() - self.start_time if self.start_time else 0,
                "loops_completed": self.loop_count
            }
    
    def _generate_recommendations(self, evaluation: Dict[str, Any], 
                                risk_assessment: Dict[str, Any], 
                                predictions: np.ndarray) -> List[str]:
        """Generate investment recommendations"""
        try:
            recommendations = []
            
            if evaluation:
                confidence = evaluation.get('confidence', 0.5)
                rel_performance = evaluation.get('metrics', {}).get('rel_performance', 1.0)
                
                if confidence > 0.8 and rel_performance < 0.9:
                    recommendations.append("BUY: Model shows strong predictive performance")
                elif confidence > 0.6 and rel_performance < 0.95:
                    recommendations.append("HOLD: Model shows moderate predictive performance")
                else:
                    recommendations.append("WAIT: Model performance is uncertain")
            
            if risk_assessment:
                risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
                if risk_level == 'HIGH':
                    recommendations.append("HIGH RISK: Consider position sizing and stop-loss")
                elif risk_level == 'MEDIUM':
                    recommendations.append("MEDIUM RISK: Monitor market conditions closely")
            
            if predictions is not None and len(predictions) > 0:
                avg_prediction = np.mean(predictions)
                if avg_prediction > 0.02:  # 2% positive return
                    recommendations.append("POSITIVE OUTLOOK: Model predicts positive returns")
                elif avg_prediction < -0.02:  # 2% negative return
                    recommendations.append("NEGATIVE OUTLOOK: Model predicts negative returns")
                else:
                    recommendations.append("NEUTRAL OUTLOOK: Model predicts minimal change")
            
            if not recommendations:
                recommendations.append("INSUFFICIENT DATA: Unable to generate recommendations")
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating recommendations: {str(e)}"]
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return float(drawdown.min())
        except:
            return 0.0
    
    def _default_logger(self, level: str, message: str, data: Dict[str, Any] = None) -> None:
        """Default logging function"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        if data:
            print(f"  Data: {json.dumps(data, indent=2, default=str)}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "config": asdict(self.config),
            "state": asdict(self.state),
            "loop_count": self.loop_count,
            "execution_time": time.time() - self.start_time if self.start_time else 0,
            "memory_stats": {
                "short_term": self.memory['short_term'].get_session_info(),
                "long_term": self.memory['long_term'].get_memory_stats()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Finance Agent...")
    
    # Create agent
    config = AgentConfig(max_loops=2, max_execution_time=60)
    agent = FinanceAgent(config)
    
    # Test goal
    goal = "Analyze PTT.BK stock and provide investment recommendation"
    context = {"symbol": "PTT.BK", "horizon": 5, "period": "1y"}
    
    # Run agent
    result = agent.run(goal, context)
    
    print(f"✅ Agent execution completed")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Execution time: {result.get('execution_time', 0):.2f} seconds")
    print(f"   Loops completed: {result.get('loops_completed', 0)}")
    
    if result.get('final_report'):
        report = result['final_report']
        print(f"   Symbol: {report.get('symbol', 'Unknown')}")
        print(f"   Recommendations: {report.get('recommendations', [])}")
    
    print("Finance Agent testing completed!")
