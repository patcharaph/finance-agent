"""
Evaluation System for Finance Agent
===================================

This module provides comprehensive evaluation capabilities:
- Model performance evaluation
- Risk assessment
- Decision criteria and thresholds
- Quality gates for agent decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


class EvaluationResult(Enum):
    """Evaluation result types"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Model Performance
    mae: float
    mse: float
    rmse: float
    r2: float
    mae_naive: float
    rel_performance: float  # MAE / MAE_naive
    
    # Risk Metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    
    # Additional Metrics
    directional_accuracy: Optional[float] = None
    hit_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'r2': self.r2,
            'mae_naive': self.mae_naive,
            'rel_performance': self.rel_performance,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'directional_accuracy': self.directional_accuracy,
            'hit_rate': self.hit_rate
        }


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    result: EvaluationResult
    metrics: EvaluationMetrics
    thresholds: Dict[str, float]
    recommendations: List[str]
    warnings: List[str]
    confidence: float
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'result': self.result.value,
            'metrics': self.metrics.to_dict(),
            'thresholds': self.thresholds,
            'recommendations': self.recommendations,
            'warnings': self.warnings,
            'confidence': self.confidence,
            'reasoning': self.reasoning
        }


class Evaluator:
    """
    Comprehensive evaluation system for finance analysis
    """
    
    def __init__(self, default_thresholds: Dict[str, float] = None):
        """
        Initialize evaluator with default thresholds
        
        Args:
            default_thresholds: Custom thresholds for evaluation
        """
        self.default_thresholds = {
            'rel_performance_max': 0.98,  # Model should be better than naive
            'r2_min': 0.1,  # Minimum R² score
            'mae_max': 0.1,  # Maximum MAE (10% error)
            'directional_accuracy_min': 0.5,  # Minimum directional accuracy
            'sharpe_min': 0.2,  # Minimum Sharpe ratio
            'max_drawdown_max': 0.2,  # Maximum drawdown (20%)
            'confidence_min': 0.6,  # Minimum confidence level
            'min_samples': 50  # Minimum samples for reliable evaluation
        }
        
        if default_thresholds:
            self.default_thresholds.update(default_thresholds)
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_naive: np.ndarray = None,
                                 thresholds: Dict[str, float] = None) -> EvaluationReport:
        """
        Evaluate model performance comprehensively
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_naive: Naive baseline predictions
            thresholds: Custom thresholds
        
        Returns:
            EvaluationReport with comprehensive analysis
        """
        try:
            # Use provided thresholds or defaults
            thresh = thresholds or self.default_thresholds
            
            # Calculate basic metrics
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            # R² score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-9))
            
            # Naive baseline comparison
            if y_naive is not None:
                mae_naive = np.mean(np.abs(y_true - y_naive))
            else:
                # Use simple naive baseline (previous value)
                y_naive = np.roll(y_true, 1)
                y_naive[0] = y_true[0]
                mae_naive = np.mean(np.abs(y_true - y_naive))
            
            rel_performance = mae / (mae_naive + 1e-9)
            
            # Directional accuracy
            true_direction = np.sign(y_true[1:] - y_true[:-1])
            pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
            directional_accuracy = np.mean(true_direction == pred_direction) if len(true_direction) > 0 else 0.5
            
            # Hit rate (predictions within acceptable range)
            error_threshold = np.std(y_true) * 0.5  # Within 0.5 standard deviations
            hit_rate = np.mean(np.abs(y_true - y_pred) <= error_threshold)
            
            # Risk metrics
            returns = y_true[1:] - y_true[:-1] if len(y_true) > 1 else np.array([0])
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(y_true)
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # Create metrics object
            metrics = EvaluationMetrics(
                mae=float(mae),
                mse=float(mse),
                rmse=float(rmse),
                r2=float(r2),
                mae_naive=float(mae_naive),
                rel_performance=float(rel_performance),
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                directional_accuracy=directional_accuracy,
                hit_rate=hit_rate
            )
            
            # Evaluate against thresholds
            result, recommendations, warnings, confidence, reasoning = self._evaluate_against_thresholds(
                metrics, thresh
            )
            
            return EvaluationReport(
                result=result,
                metrics=metrics,
                thresholds=thresh,
                recommendations=recommendations,
                warnings=warnings,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            return EvaluationReport(
                result=EvaluationResult.FAIL,
                metrics=EvaluationMetrics(0, 0, 0, 0, 0, 1),
                thresholds=thresh or self.default_thresholds,
                recommendations=[],
                warnings=[f"Evaluation error: {str(e)}"],
                confidence=0.0,
                reasoning=f"Failed to evaluate model: {str(e)}"
            )
    
    def evaluate_data_quality(self, data: pd.DataFrame, 
                            required_columns: List[str] = None) -> EvaluationReport:
        """
        Evaluate data quality for analysis
        
        Args:
            data: DataFrame to evaluate
            required_columns: Required columns for analysis
        
        Returns:
            EvaluationReport for data quality
        """
        try:
            thresh = self.default_thresholds
            recommendations = []
            warnings = []
            
            # Check data size
            if len(data) < thresh['min_samples']:
                return EvaluationReport(
                    result=EvaluationResult.INSUFFICIENT_DATA,
                    metrics=EvaluationMetrics(0, 0, 0, 0, 0, 1),
                    thresholds=thresh,
                    recommendations=["Increase data size or reduce analysis complexity"],
                    warnings=[f"Only {len(data)} samples available, need at least {thresh['min_samples']}"],
                    confidence=0.0,
                    reasoning="Insufficient data for reliable analysis"
                )
            
            # Check required columns
            if required_columns:
                missing_cols = [col for col in required_columns if col not in data.columns]
                if missing_cols:
                    warnings.append(f"Missing required columns: {missing_cols}")
            
            # Check for missing values
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_pct > 0.1:  # More than 10% missing
                warnings.append(f"High missing data percentage: {missing_pct:.1%}")
                recommendations.append("Consider data imputation or feature selection")
            
            # Check for constant columns
            constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
            if constant_cols:
                warnings.append(f"Constant columns detected: {constant_cols}")
                recommendations.append("Remove constant columns")
            
            # Check for outliers
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outlier_cols = []
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > len(data) * 0.05:  # More than 5% outliers
                    outlier_cols.append(col)
            
            if outlier_cols:
                warnings.append(f"High outlier percentage in columns: {outlier_cols}")
                recommendations.append("Consider outlier treatment")
            
            # Determine result
            if len(warnings) == 0:
                result = EvaluationResult.PASS
                confidence = 0.9
                reasoning = "Data quality is good for analysis"
            elif len(warnings) <= 2:
                result = EvaluationResult.WARNING
                confidence = 0.7
                reasoning = "Data quality is acceptable with minor issues"
            else:
                result = EvaluationResult.FAIL
                confidence = 0.4
                reasoning = "Data quality issues may affect analysis reliability"
            
            # Create dummy metrics for data quality evaluation
            metrics = EvaluationMetrics(
                mae=0, mse=0, rmse=0, r2=1, mae_naive=0, rel_performance=0
            )
            
            return EvaluationReport(
                result=result,
                metrics=metrics,
                thresholds=thresh,
                recommendations=recommendations,
                warnings=warnings,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            return EvaluationReport(
                result=EvaluationResult.FAIL,
                metrics=EvaluationMetrics(0, 0, 0, 0, 0, 1),
                thresholds=thresh or self.default_thresholds,
                recommendations=[],
                warnings=[f"Data quality evaluation error: {str(e)}"],
                confidence=0.0,
                reasoning=f"Failed to evaluate data quality: {str(e)}"
            )
    
    def evaluate_prediction_confidence(self, y_pred: np.ndarray, 
                                     model_uncertainty: np.ndarray = None,
                                     market_conditions: Dict[str, Any] = None) -> EvaluationReport:
        """
        Evaluate confidence in predictions
        
        Args:
            y_pred: Predictions
            model_uncertainty: Model uncertainty estimates
            market_conditions: Current market conditions
        
        Returns:
            EvaluationReport for prediction confidence
        """
        try:
            thresh = self.default_thresholds
            recommendations = []
            warnings = []
            
            # Calculate prediction statistics
            pred_std = np.std(y_pred)
            pred_range = np.max(y_pred) - np.min(y_pred)
            
            # High volatility warning
            if pred_std > 0.1:  # More than 10% standard deviation
                warnings.append("High prediction volatility detected")
                recommendations.append("Consider ensemble methods or uncertainty quantification")
            
            # Extreme predictions warning
            extreme_predictions = np.sum(np.abs(y_pred) > 0.2)  # More than 20% change
            if extreme_predictions > len(y_pred) * 0.1:  # More than 10% extreme predictions
                warnings.append("High number of extreme predictions")
                recommendations.append("Review model calibration and feature scaling")
            
            # Model uncertainty analysis
            if model_uncertainty is not None:
                avg_uncertainty = np.mean(model_uncertainty)
                if avg_uncertainty > 0.1:  # High uncertainty
                    warnings.append("High model uncertainty detected")
                    recommendations.append("Consider additional data or model ensemble")
            
            # Market conditions analysis
            if market_conditions:
                if market_conditions.get('volatility', 0) > 0.3:  # High market volatility
                    warnings.append("High market volatility may affect prediction reliability")
                    recommendations.append("Use shorter prediction horizons or risk management")
                
                if market_conditions.get('news_sentiment', 0) < -0.5:  # Very negative sentiment
                    warnings.append("Negative market sentiment detected")
                    recommendations.append("Consider sentiment-aware models")
            
            # Calculate confidence score
            confidence = 1.0
            confidence -= min(len(warnings) * 0.1, 0.5)  # Reduce confidence for warnings
            confidence -= min(pred_std * 2, 0.3)  # Reduce confidence for high volatility
            confidence = max(confidence, 0.0)
            
            # Determine result
            if confidence >= thresh['confidence_min']:
                result = EvaluationResult.PASS
                reasoning = "Predictions have sufficient confidence"
            elif confidence >= thresh['confidence_min'] * 0.7:
                result = EvaluationResult.WARNING
                reasoning = "Predictions have moderate confidence with some concerns"
            else:
                result = EvaluationResult.FAIL
                reasoning = "Predictions have low confidence, consider alternative approaches"
            
            # Create dummy metrics for confidence evaluation
            metrics = EvaluationMetrics(
                mae=pred_std, mse=pred_std**2, rmse=pred_std, r2=confidence, 
                mae_naive=0, rel_performance=1
            )
            
            return EvaluationReport(
                result=result,
                metrics=metrics,
                thresholds=thresh,
                recommendations=recommendations,
                warnings=warnings,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            return EvaluationReport(
                result=EvaluationResult.FAIL,
                metrics=EvaluationMetrics(0, 0, 0, 0, 0, 1),
                thresholds=thresh or self.default_thresholds,
                recommendations=[],
                warnings=[f"Confidence evaluation error: {str(e)}"],
                confidence=0.0,
                reasoning=f"Failed to evaluate prediction confidence: {str(e)}"
            )
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> Optional[float]:
        """Calculate Sharpe ratio"""
        if len(returns) < 2 or np.std(returns) == 0:
            return None
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))  # Annualized
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> Optional[float]:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return None
        
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return float(np.min(drawdown))
    
    def _evaluate_against_thresholds(self, metrics: EvaluationMetrics, 
                                   thresholds: Dict[str, float]) -> Tuple[EvaluationResult, List[str], List[str], float, str]:
        """
        Evaluate metrics against thresholds and generate recommendations
        
        Returns:
            Tuple of (result, recommendations, warnings, confidence, reasoning)
        """
        recommendations = []
        warnings = []
        failed_checks = 0
        total_checks = 0
        
        # Check relative performance
        total_checks += 1
        if metrics.rel_performance > thresholds['rel_performance_max']:
            failed_checks += 1
            recommendations.append("Model performs worse than naive baseline - consider feature engineering or different model")
            warnings.append(f"Relative performance {metrics.rel_performance:.3f} exceeds threshold {thresholds['rel_performance_max']}")
        
        # Check R² score
        total_checks += 1
        if metrics.r2 < thresholds['r2_min']:
            failed_checks += 1
            recommendations.append("Low R² score - model explains little variance")
            warnings.append(f"R² score {metrics.r2:.3f} below threshold {thresholds['r2_min']}")
        
        # Check MAE
        total_checks += 1
        if metrics.mae > thresholds['mae_max']:
            failed_checks += 1
            recommendations.append("High MAE - consider model tuning or additional features")
            warnings.append(f"MAE {metrics.mae:.3f} exceeds threshold {thresholds['mae_max']}")
        
        # Check directional accuracy
        if metrics.directional_accuracy is not None:
            total_checks += 1
            if metrics.directional_accuracy < thresholds['directional_accuracy_min']:
                failed_checks += 1
                recommendations.append("Low directional accuracy - model struggles with trend prediction")
                warnings.append(f"Directional accuracy {metrics.directional_accuracy:.3f} below threshold {thresholds['directional_accuracy_min']}")
        
        # Check Sharpe ratio
        if metrics.sharpe_ratio is not None:
            total_checks += 1
            if metrics.sharpe_ratio < thresholds['sharpe_min']:
                failed_checks += 1
                recommendations.append("Low Sharpe ratio - risk-adjusted returns are poor")
                warnings.append(f"Sharpe ratio {metrics.sharpe_ratio:.3f} below threshold {thresholds['sharpe_min']}")
        
        # Check max drawdown
        if metrics.max_drawdown is not None:
            total_checks += 1
            if abs(metrics.max_drawdown) > thresholds['max_drawdown_max']:
                failed_checks += 1
                recommendations.append("High maximum drawdown - consider risk management")
                warnings.append(f"Max drawdown {metrics.max_drawdown:.3f} exceeds threshold {thresholds['max_drawdown_max']}")
        
        # Calculate confidence
        confidence = max(0, 1 - (failed_checks / total_checks))
        
        # Determine result
        if failed_checks == 0:
            result = EvaluationResult.PASS
            reasoning = "All evaluation criteria passed"
        elif failed_checks <= total_checks * 0.3:  # Less than 30% failed
            result = EvaluationResult.WARNING
            reasoning = "Most evaluation criteria passed with minor issues"
        else:
            result = EvaluationResult.FAIL
            reasoning = "Multiple evaluation criteria failed"
        
        return result, recommendations, warnings, confidence, reasoning
    
    def get_evaluation_summary(self, reports: List[EvaluationReport]) -> Dict[str, Any]:
        """
        Get summary of multiple evaluation reports
        
        Args:
            reports: List of evaluation reports
        
        Returns:
            Summary dictionary
        """
        if not reports:
            return {"error": "No reports provided"}
        
        # Count results
        result_counts = {}
        for report in reports:
            result = report.result.value
            result_counts[result] = result_counts.get(result, 0) + 1
        
        # Calculate average confidence
        avg_confidence = np.mean([r.confidence for r in reports])
        
        # Collect all recommendations and warnings
        all_recommendations = []
        all_warnings = []
        for report in reports:
            all_recommendations.extend(report.recommendations)
            all_warnings.extend(report.warnings)
        
        # Get most common recommendations
        from collections import Counter
        common_recommendations = Counter(all_recommendations).most_common(5)
        common_warnings = Counter(all_warnings).most_common(5)
        
        return {
            "total_reports": len(reports),
            "result_distribution": result_counts,
            "average_confidence": float(avg_confidence),
            "most_common_recommendations": common_recommendations,
            "most_common_warnings": common_warnings,
            "overall_status": "PASS" if result_counts.get("pass", 0) > len(reports) * 0.7 else "NEEDS_ATTENTION"
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Evaluation System...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.normal(0, 0.05, n_samples)
    y_pred = y_true + np.random.normal(0, 0.02, n_samples)  # Good predictions
    y_naive = np.roll(y_true, 1)
    y_naive[0] = y_true[0]
    
    # Test evaluator
    evaluator = Evaluator()
    
    # Test model performance evaluation
    report = evaluator.evaluate_model_performance(y_true, y_pred, y_naive)
    print(f"✅ Model Performance: {report.result.value}")
    print(f"   Confidence: {report.confidence:.2f}")
    print(f"   Reasoning: {report.reasoning}")
    
    # Test data quality evaluation
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'target': y_true
    })
    
    data_report = evaluator.evaluate_data_quality(sample_data)
    print(f"✅ Data Quality: {data_report.result.value}")
    print(f"   Confidence: {data_report.confidence:.2f}")
    
    # Test prediction confidence evaluation
    conf_report = evaluator.evaluate_prediction_confidence(y_pred)
    print(f"✅ Prediction Confidence: {conf_report.result.value}")
    print(f"   Confidence: {conf_report.confidence:.2f}")
    
    print("Evaluation system testing completed!")
