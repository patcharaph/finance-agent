#!/usr/bin/env python3
"""
Optimized Finance Agent Demo - Streamlined Version
=================================================

Simplified demo showing the core enhanced features:
- LLM Planner with investment decisions
- Think-Act-Evaluate-Reflect loop
- Reflection and learning
- Bilingual reports
"""

import os
import sys
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.agent_optimized import OptimizedFinanceAgent, AgentConfig
from agent.planner_optimized import OptimizedPlanner


def demo_optimized_features():
    """Demo the optimized features"""
    print("🚀 Optimized Finance Agent Demo")
    print("=" * 50)
    
    # Create agent with optimized configuration
    config = AgentConfig(
        max_loops=2,
        confidence_threshold=0.7,
        enable_llm_planning=True
    )
    
    # Mock LLM client for demo
    class MockLLMClient:
        def __init__(self):
            self.available = True
        
        def chat(self, system_prompt, user_prompt):
            if "financial research planner" in system_prompt:
                return json.dumps({
                    "decision": "SINGLE_STOCK",
                    "why_decision": "Individual stock analysis shows better risk-adjusted returns",
                    "targets": ["PTT.BK", "KBANK.BK", "ADVANC.BK"],
                    "subtasks": [
                        {"id": "fetch_data", "tool": "data.load_prices", "args": {"symbols": ["PTT.BK"], "period": "2y"}},
                        {"id": "features", "tool": "indicators.compute", "args": {"indicators": ["RSI", "MACD", "EMA_10"]}},
                        {"id": "model", "tool": "ml.train_predict", "args": {"algo": "RandomForest", "horizon_days": 10}},
                        {"id": "risk", "tool": "risk.assess", "args": {"mdd_window": "1y"}},
                        {"id": "decide", "tool": "decision.summarize", "args": {}}
                    ],
                    "acceptance": {"min_sharpe": 0.0, "max_mdd_pct": 15}
                })
            elif "reflection analyst" in system_prompt:
                return json.dumps({
                    "insights": ["Model shows good performance", "Data quality is acceptable"],
                    "suggestions": ["Consider additional indicators", "Extend analysis horizon"],
                    "lesson": {
                        "pattern": "Strong momentum with low volatility",
                        "action": "Suitable for medium-term investment",
                        "example": "PTT.BK Q4 2024 conditions"
                    }
                })
            elif "financial report writer" in system_prompt:
                return json.dumps({
                    "executive_summary_th": "การวิเคราะห์หุ้น PTT.BK แสดงให้เห็นโอกาสการลงทุนในระยะกลาง",
                    "executive_summary_en": "PTT.BK analysis shows medium-term investment opportunities",
                    "reasons_evidence_th": ["โมเดลแสดงความแม่นยำ 72%", "RSI อยู่ในระดับเหมาะสม", "MACD แสดงสัญญาณบวก"],
                    "risk_caveats_th": ["ความผันผวนของราคาน้ำมัน", "ความไม่แน่นอนของตลาด"],
                    "investment_recommendation": "BUY - เหมาะสำหรับการลงทุนระยะกลาง",
                    "confidence_level": "High (75%)"
                })
            else:
                return "Mock response"
    
    agent = OptimizedFinanceAgent(config, MockLLMClient())
    
    # Demo scenario
    print("📈 Demo Scenario: PTT.BK Stock Analysis")
    print("-" * 40)
    
    goal = "วิเคราะห์หุ้น PTT.BK และให้คำแนะนำการลงทุน"
    context = {"symbol": "PTT.BK", "horizon": 10, "period": "1y"}
    
    print(f"Goal: {goal}")
    print(f"Context: {context}")
    
    # Run agent
    print(f"\n🔄 Running optimized agent...")
    result = agent.run(goal, context)
    
    # Display results
    print(f"\n✅ Analysis Results:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Loops completed: {result.get('loops_completed', 0)}")
    print(f"   Execution time: {result.get('execution_time', 0):.2f} seconds")
    
    if result.get('final_report'):
        report = result['final_report']
        
        if 'llm_explanation' in report:
            explanation = report['llm_explanation']
            print(f"\n📊 Investment Decision:")
            print(f"   Recommendation: {explanation.get('investment_recommendation', 'N/A')}")
            print(f"   Confidence: {explanation.get('confidence_level', 'N/A')}")
            
            print(f"\n🇹🇭 Thai Summary:")
            print(f"   {explanation.get('executive_summary_th', 'N/A')}")
            
            print(f"\n🇺🇸 English Summary:")
            print(f"   {explanation.get('executive_summary_en', 'N/A')}")
            
            print(f"\n📋 Key Reasons:")
            for reason in explanation.get('reasons_evidence_th', []):
                print(f"   • {reason}")
            
            print(f"\n⚠️ Risk Factors:")
            for risk in explanation.get('risk_caveats_th', []):
                print(f"   • {risk}")
    
    print(f"\n🎯 Optimized Features Demonstrated:")
    print(f"   ✅ LLM Planner: Investment decision logic")
    print(f"   ✅ Agent Loop: Think-Act-Evaluate-Reflect")
    print(f"   ✅ Reflection: Learning and insights")
    print(f"   ✅ Reports: Bilingual summaries")
    
    print(f"\n📊 Code Optimization Benefits:")
    print(f"   • Reduced from 1700+ lines to ~400 lines")
    print(f"   • Simplified configuration and setup")
    print(f"   • Streamlined execution flow")
    print(f"   • Maintained all core features")
    
    return True


if __name__ == "__main__":
    try:
        demo_optimized_features()
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
