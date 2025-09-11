#!/usr/bin/env python3
"""
Demo script for the enhanced Finance Agent features
Shows how to use the 4 new features in practice
"""

import os
import sys
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.agent import FinanceAgent, AgentConfig

def demo_enhanced_features():
    """Demo the enhanced features with real examples"""
    print("🚀 Finance Agent Enhanced Features Demo")
    print("=" * 60)
    
    # Create agent with enhanced configuration
    config = AgentConfig(
        max_loops=3,
        confidence_threshold=0.7,
        enable_llm_planning=True,
        enable_learning=True,
        enable_reflection=True
    )
    
    # Mock LLM client for demo (replace with real LLM client)
    class DemoLLMClient:
        def __init__(self):
            self.available = True
        
        def chat(self, system_prompt, user_prompt):
            if "financial research planner" in system_prompt:
                # Example 1: INDEX_SET decision
                if "SET" in user_prompt or "ดัชนี" in user_prompt:
                    return json.dumps({
                        "decision": "INDEX_SET",
                        "why_decision": "SET Index shows strong momentum with low volatility, suitable for medium-term investment",
                        "targets": ["^SETI"],
                        "subtasks": [
                            {"id": "fetch_data", "tool": "data.load_prices", "args": {"symbols": ["^SETI"], "period": "2y", "interval": "1d"}},
                            {"id": "features", "tool": "indicators.compute", "args": {"indicators": ["RSI", "MACD", "EMA_10", "EMA_20", "BBANDS"]}},
                            {"id": "model", "tool": "ml.train_predict", "args": {"algo": "RandomForest", "horizon_days": 20}},
                            {"id": "risk", "tool": "risk.assess", "args": {"mdd_window": "1y"}},
                            {"id": "mini_backtest", "tool": "backtest.quick", "args": {"metric": ["Sharpe", "MDD"]}},
                            {"id": "decide", "tool": "decision.summarize", "args": {}}
                        ],
                        "acceptance": {"min_sharpe": 0.0, "max_mdd_pct": 20},
                        "report_needs": ["executive_summary", "rationale_th_en", "disclaimer"]
                    })
                # Example 2: SINGLE_STOCK decision
                else:
                    return json.dumps({
                        "decision": "SINGLE_STOCK",
                        "why_decision": "Individual stock analysis shows better risk-adjusted returns for this time horizon",
                        "targets": ["PTT.BK", "KBANK.BK", "ADVANC.BK"],
                        "subtasks": [
                            {"id": "fetch_data", "tool": "data.load_prices", "args": {"symbols": ["PTT.BK"], "period": "2y"}},
                            {"id": "features", "tool": "indicators.compute", "args": {"indicators": ["RSI", "MACD", "EMA_10", "EMA_20", "BBANDS"]}},
                            {"id": "model", "tool": "ml.train_predict", "args": {"algo": "RandomForest", "horizon_days": 10}},
                            {"id": "risk", "tool": "risk.assess", "args": {"mdd_window": "1y"}},
                            {"id": "mini_backtest", "tool": "backtest.quick", "args": {"metric": ["Sharpe", "MDD"]}},
                            {"id": "decide", "tool": "decision.summarize", "args": {}}
                        ],
                        "acceptance": {"min_sharpe": 0.0, "max_mdd_pct": 15},
                        "report_needs": ["executive_summary", "rationale_th_en", "disclaimer"]
                    })
            
            elif "reflection analyst" in system_prompt:
                return json.dumps({
                    "root_causes": ["Insufficient data quality", "Model overfitting to recent trends"],
                    "suggestions": [
                        {"change": "features", "detail": "add EMA_gap, ATR, and Stochastic indicators"},
                        {"change": "horizon", "detail": "extend to 30 days for trend stability"},
                        {"change": "model", "detail": "try GradientBoosting as alternative"}
                    ],
                    "lesson": {
                        "pattern": "Sideways market + low volume conditions",
                        "trigger": "RSI in 45-55 range for >30 sessions, ATR below average",
                        "action": "Prefer INDEX_SET with mean-reversion bands; avoid breakout models",
                        "example": "SETI 2024-2025 Q2 conditions"
                    }
                })
            
            elif "financial report writer" in system_prompt:
                if "INDEX_SET" in user_prompt:
                    return json.dumps({
                        "executive_summary_th": "การวิเคราะห์ดัชนี SET แสดงให้เห็นโอกาสการลงทุนในระยะกลาง โดยมีความเสี่ยงอยู่ในระดับปานกลาง",
                        "executive_summary_en": "SET Index analysis shows medium-term investment opportunities with moderate risk levels",
                        "reasons_evidence_th": [
                            "โมเดลแสดงความแม่นยำ 68% ในการทำนายทิศทาง",
                            "RSI อยู่ในระดับ 52 แสดงความสมดุล",
                            "MACD แสดงสัญญาณบวก",
                            "ความผันผวนอยู่ในระดับที่ยอมรับได้"
                        ],
                        "risk_caveats_th": [
                            "ความผันผวนของตลาดหุ้นโดยรวม",
                            "ปัจจัยเศรษฐกิจมหภาคที่อาจส่งผลกระทบ",
                            "ความไม่แน่นอนของตลาดโลก"
                        ],
                        "watch_next_th": [
                            "ติดตามดัชนีเศรษฐกิจมหภาค",
                            "ดูสัญญาณจากตลาดหุ้นต่างประเทศ",
                            "ติดตามข่าวสารสำคัญของบริษัทใหญ่"
                        ],
                        "disclaimer_th": "ข้อมูลนี้ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน กรุณาศึกษาและปรึกษาผู้เชี่ยวชาญก่อนตัดสินใจลงทุน",
                        "investment_recommendation": "HOLD - เหมาะสำหรับการลงทุนระยะกลาง",
                        "confidence_level": "Medium-High (70%)",
                        "top_pick": "^SETI",
                        "alternatives": ["^SET50", "^SET100"]
                    })
                else:
                    return json.dumps({
                        "executive_summary_th": "การวิเคราะห์หุ้น PTT.BK แสดงให้เห็นโอกาสการลงทุนในระยะสั้นถึงกลาง โดยมีปัจจัยสนับสนุนจากราคาน้ำมัน",
                        "executive_summary_en": "PTT.BK analysis shows short to medium-term investment opportunities supported by oil price factors",
                        "reasons_evidence_th": [
                            "โมเดลแสดงความแม่นยำ 72% ในการทำนายราคา",
                            "RSI อยู่ในระดับ 48 แสดงโอกาสซื้อ",
                            "MACD แสดงสัญญาณกลับตัว",
                            "ราคาน้ำมันดิบมีแนวโน้มดีขึ้น"
                        ],
                        "risk_caveats_th": [
                            "ความผันผวนของราคาน้ำมันดิบ",
                            "ความไม่แน่นอนของตลาดพลังงาน",
                            "ปัจจัยการเมืองและเศรษฐกิจ"
                        ],
                        "watch_next_th": [
                            "ติดตามราคาน้ำมันดิบ WTI และ Brent",
                            "ดูผลประกอบการไตรมาส",
                            "ติดตามข่าวสารการลงทุนใหม่"
                        ],
                        "disclaimer_th": "ข้อมูลนี้ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน กรุณาศึกษาและปรึกษาผู้เชี่ยวชาญก่อนตัดสินใจลงทุน",
                        "investment_recommendation": "BUY - เหมาะสำหรับการลงทุนระยะสั้นถึงกลาง",
                        "confidence_level": "High (75%)",
                        "top_pick": "PTT.BK",
                        "alternatives": ["KBANK.BK", "ADVANC.BK"]
                    })
            
            else:
                return "Mock response"
    
    agent = FinanceAgent(config, DemoLLMClient())
    
    # Demo scenarios
    scenarios = [
        {
            "title": "📈 Scenario 1: SET Index Analysis",
            "goal": "ช่วยประเมินว่าควรลงทุนใน SET หรือเลือกหุ้นรายตัวใน 1-3 เดือนข้างหน้า",
            "context": {"budget_thb": 200000, "risk_tolerance": "medium", "time_horizon": "1-3m"}
        },
        {
            "title": "🏢 Scenario 2: Individual Stock Analysis",
            "goal": "วิเคราะห์หุ้น PTT.BK และให้คำแนะนำการลงทุนระยะสั้น",
            "context": {"symbol": "PTT.BK", "horizon": 10, "period": "1y", "risk_tolerance": "medium"}
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{scenario['title']}")
        print("-" * 50)
        print(f"Goal: {scenario['goal']}")
        print(f"Context: {scenario['context']}")
        
        # Run agent
        result = agent.run(scenario['goal'], scenario['context'])
        
        # Display results
        print(f"\n✅ Analysis Results:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Loops completed: {result.get('loops_completed', 0)}")
        print(f"   Execution time: {result.get('execution_time', 0):.2f} seconds")
        
        if result.get('final_report'):
            report = result['final_report']
            
            # Show plan decision
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
                
                print(f"\n👀 What to Watch:")
                for watch in explanation.get('watch_next_th', []):
                    print(f"   • {watch}")
                
                print(f"\n📝 Disclaimer:")
                print(f"   {explanation.get('disclaimer_th', 'N/A')}")
        
        print("\n" + "=" * 60)
    
    # Show memory statistics
    print(f"\n💾 Memory Statistics:")
    memory_stats = agent.get_agent_status().get('memory_stats', {})
    long_term_stats = memory_stats.get('long_term', {})
    print(f"   Long-term lessons: {long_term_stats.get('lessons_count', 0)}")
    print(f"   Long-term strategies: {long_term_stats.get('strategies_count', 0)}")
    
    print(f"\n🎯 Enhanced Features Demonstrated:")
    print(f"   1. ✅ LLM Planner: INDEX_SET vs SINGLE_STOCK vs NO_TRADE decisions")
    print(f"   2. ✅ Agent Loop: Think-Act-Evaluate-Reflect with acceptance criteria")
    print(f"   3. ✅ Reflect System: Root cause analysis and lesson learning")
    print(f"   4. ✅ Report & Explain: TH/EN summaries with human-readable output")
    
    return True

if __name__ == "__main__":
    try:
        demo_enhanced_features()
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
