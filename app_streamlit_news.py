
# app_streamlit.py
# Streamlit one-page app: LLM-Agentic Finance Analyst with real-time reasoning log
# Author: ChatGPT for Pae
# How to run:
#   pip install streamlit pandas numpy scikit-learn yfinance openai
#   export OPENAI_API_KEY="sk-..."  (or setx on Windows PowerShell)
#   streamlit run app_streamlit.py

import os
import time
import json
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import streamlit as st

# ---------- LLM client (supports OpenRouter or OpenAI, with auto-detect) ----------
class LLMClient:
    """
    เลือก provider อัตโนมัติ:
      - ถ้ามี OPENROUTER_API_KEY -> ใช้ OpenRouter (REST)
      - elif มี OPENAI_API_KEY -> ใช้ OpenAI (ChatCompletion legacy)
      - else -> fallback (ไม่เรียก API; ให้ชั้นบนจัดการ)
    ตั้งโมเดล:
      - ใช้ OPENROUTER_MODEL หรือ LLM_MODEL ถ้ามี
      - ดีฟอลต์: openrouter/auto (OpenRouter) หรือ gpt-4o-mini (OpenAI)
    """
    def __init__(self, model: str = None):
        # auto detect provider
        self.provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else ("openai" if os.getenv("OPENAI_API_KEY") else "none")
        # choose model
        env_model = os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL")
        if self.provider == "openrouter":
            self.model = model or env_model or "openrouter/auto"
            self._api_key = os.getenv("OPENROUTER_API_KEY")
            self._base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            # Header ที่ OpenRouter แนะนำ (ช่วยเรื่อง rate-limit/observability)
            self._site_url = os.getenv("OPENROUTER_SITE_URL", "https://example.com")
            self._app_name = os.getenv("OPENROUTER_APP_NAME", "Finance Agent Demo")
            self.available = True if self._api_key else False
        elif self.provider == "openai":
            self.model = model or env_model or "gpt-4o-mini"
            try:
                import openai  # legacy-compatible
                self._openai = openai
                self._openai.api_key = os.getenv("OPENAI_API_KEY")
                self.available = True
            except Exception:
                self._openai = None
                self.available = False
        else:
            self.model = model or env_model or ""
            self.available = False

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if not self.available:
            return None

        if self.provider == "openrouter":
            import requests
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                # 2 header ด้านล่างช่วยให้โปร่งใส/บางทีได้สิทธิ์ใช้งานดีขึ้น
                "HTTP-Referer": self._site_url,
                "X-Title": self._app_name,
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
            try:
                r = requests.post(f"{self._base_url}/chat/completions", headers=headers, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                return None

        # OpenAI legacy ChatCompletion path
        try:
            resp = self._openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return None


# ---------------------- Prompts ----------------------
SYSTEM_AGENT = """You are a senior finance analyst agent for Thai equities/indices.
Operate with planning, tool-use, evaluation, and reflection. Be risk-aware, concise, and structured.
"""

DEVELOPER_RULES = """
Success criteria:
- Forecast: rel = MAE/MAE_naive < 0.98  (pass)
- Optional backtest: Sharpe > 0.2 and MaxDD < 20%

Reflect policy (priority):
1) Horizon sweep: 5d ↔ 10d ↔ 20d
2) Feature enrich/toggle: RSI, VOLZ (volume z-score)
3) RF tweak: n_estimators (300↔500), min_samples_leaf (1↔2)

Guardrails:
- Max loops = 3
- Avoid leakage/overfit, explain every change briefly
- If data is insufficient: answer WAIT with data_gaps
"""

# ---------------------- Tools: Data/Features/Model ----------------------
def fetch_price(symbol: str, period="2y", interval="1d") -> pd.DataFrame:
    """Fetch price data using test.py method"""
    
    print(f"🔄 กำลังดึงข้อมูล {symbol}...")
    
    try:
        # Convert period to start/end dates
        from datetime import datetime, timedelta
        end_date = datetime.now()
        if period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "3y":
            start_date = end_date - timedelta(days=1095)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:
            start_date = end_date - timedelta(days=730)
        
        # Use test.py method - try real data first
        df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), 
                        end=end_date.strftime("%Y-%m-%d"), 
                        interval=interval, auto_adjust=True, progress=False)
        
        if df is not None and not df.empty:
            print(f"✅ ดึงข้อมูล {symbol} สำเร็จ! ({len(df)} วัน)")
            # Convert timezone to Bangkok time
            try:
                df = df.tz_localize("UTC").tz_convert("Asia/Bangkok")
            except:
                print("⚠️ ไม่สามารถแปลง timezone ได้ ใช้เวลาปกติแทน")
            return df
        else:
            raise ValueError("ข้อมูลว่างเปล่า")
            
    except Exception as e:
        print(f"❌ ไม่สามารถดึงข้อมูล {symbol} ได้: {e}")
        print("🔄 สร้างข้อมูลจำลองแทน...")
        
        # Create realistic dummy data using test.py method
        from datetime import datetime, timedelta
        end_date = datetime.now()
        if period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "3y":
            start_date = end_date - timedelta(days=1095)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:
            start_date = end_date - timedelta(days=730)
            
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Set initial price based on symbol (from test.py)
        if "PTT" in symbol:
            initial_price = 35.0  # PTT ราคาประมาณ 35 บาท
        elif "AAPL" in symbol:
            initial_price = 180.0  # Apple ราคาประมาณ $180
        elif "MSFT" in symbol:
            initial_price = 400.0
        elif "GOOGL" in symbol:
            initial_price = 140.0
        else:
            initial_price = 100.0
        
        # Generate realistic price movement (from test.py)
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% daily return, 2% volatility
        prices = [initial_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))  # ป้องกันราคาติดลบ
        
        # Create OHLCV data (from test.py)
        dummy_df = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        # Adjust High/Low to be realistic (from test.py)
        dummy_df['High'] = dummy_df[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, n_days))
        dummy_df['Low'] = dummy_df[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, n_days))
        
        print(f"✅ สร้างข้อมูลจำลอง {symbol} สำเร็จ! ({len(dummy_df)} วัน)")
        return dummy_df

def build_features(price: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=price.index)
    
    # Handle MultiIndex columns from Yahoo Finance
    if isinstance(price.columns, pd.MultiIndex):
        close_col = price.columns[0]  # First column is usually Close
        volume_col = price.columns[4] # Fifth column is usually Volume
    else:
        close_col = "Close" if "Close" in price.columns else "close"
        volume_col = "Volume" if "Volume" in price.columns else "volume"
    
    feat["ret_1d"] = price[close_col].pct_change()
    feat["ret_5d"] = price[close_col].pct_change(5)
    vol_ma = price[volume_col].rolling(20).mean()
    vol_sd = price[volume_col].rolling(20).std()
    feat["vol_z"] = (price[volume_col] - vol_ma) / (vol_sd + 1e-9)
    delta = price[close_col].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    feat["rsi14"] = 100 - (100 / (1 + rs))
    return feat.dropna()

def make_target(price: pd.DataFrame, horizon_days: int, align_index: pd.Index) -> pd.Series:
    # Handle MultiIndex columns from Yahoo Finance
    if isinstance(price.columns, pd.MultiIndex):
        close_col = price.columns[0]  # First column is usually Close
    else:
        close_col = "Close" if "Close" in price.columns else "close"
    
    close = price[close_col]
    y = close.pct_change(horizon_days).shift(-horizon_days)
    y = y.reindex(align_index).dropna()
    return y

def split_train_test(X, y, test_ratio=0.2, seed=42):
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(n*(1-test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

def train_rf(Xtr, ytr, n_estimators=300, min_samples_leaf=1, random_state=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model.fit(Xtr, ytr)
    return model

# ---------------------- Evaluator ----------------------
@dataclass
class EvalReport:
    passed: bool
    mae: float
    mae_naive: float
    rel: float
    reason: str

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray, thresh=0.98) -> EvalReport:
    mae = mean_absolute_error(y_true, y_pred)
    mae_naive = mean_absolute_error(y_true, y_naive)
    rel = mae / (mae_naive + 1e-9)
    passed = rel < thresh
    return EvalReport(passed, mae, mae_naive, rel, "ok" if passed else "worse-than-threshold")

# ---------------------- LLM layers ----------------------
def plan_via_llm(llm: LLMClient, goal: dict, available_tools: dict) -> List[Dict[str, Any]]:
    if not llm.available:
        return [{"step":"fetch_data"},{"step":"build_xy"},{"step":"train_eval"},{"step":"summarize"}]

    sys_p = SYSTEM_AGENT + "\n" + DEVELOPER_RULES
    user_p = f"""
Goal: {goal}
Available tools (I/O brief): {json.dumps(available_tools, ensure_ascii=False)}

Requirements:
- Output ONLY a valid JSON array of steps (no extra text).
- Steps can include: "fetch_data", "build_xy", "train_eval", "tune_params", "add_features", "summarize".
- Keep it minimal and executable. No prose.
"""
    out = llm.chat(sys_p, user_p)
    try:
        plan = json.loads(out)
        assert isinstance(plan, list)
        return plan
    except Exception:
        return [{"step":"fetch_data"},{"step":"build_xy"},{"step":"train_eval"},{"step":"summarize"}]

def reflect_via_llm(llm: LLMClient, eval_report: EvalReport, short_state: dict) -> dict:
    if not llm.available:
        horizon_cycle = [5,10,20]
        cur = short_state.get("horizon",5)
        nxt = horizon_cycle[(horizon_cycle.index(cur)+1)%len(horizon_cycle)]
        return {
            "changes": {
                "horizon": nxt,
                "use_rsi": not short_state.get("use_rsi", True),
                "use_vol": not short_state.get("use_vol", True),
                "n_estimators": 500 if short_state.get("n_estimators",300)==300 else 300,
                "min_samples_leaf": 2 if short_state.get("min_samples_leaf",1)==1 else 1
            },
            "reason": "Fallback reflect: cycle horizon, toggle features, tweak RF."
        }

    sys_p = SYSTEM_AGENT + "\n" + DEVELOPER_RULES
    snapshot = {
        "horizon": short_state.get("horizon"),
        "use_rsi": short_state.get("use_rsi"),
        "use_vol": short_state.get("use_vol"),
        "n_estimators": short_state.get("n_estimators"),
        "min_samples_leaf": short_state.get("min_samples_leaf"),
        "feature_names": short_state.get("feature_names", [])
    }
    user_p = f"""
Evaluation result: {eval_report.__dict__}
Current settings: {json.dumps(snapshot, ensure_ascii=False)}

Task:
- If passed==False, propose minimal changes to improve rel (<0.98) next run.
- Respect the Reflect policy priority.
- Return ONLY JSON with fields:
  {{
    "changes": {{
      "horizon": <int or null>,
      "use_rsi": <true/false or null>,
      "use_vol": <true/false or null>,
      "n_estimators": <int or null>,
      "min_samples_leaf": <int or null>
    }},
    "reason": "<one-paragraph concise explanation>"
  }}
No extra text.
"""
    out = llm.chat(sys_p, user_p)
    try:
        plan = json.loads(out)
        assert isinstance(plan, dict) and "changes" in plan
        return plan
    except Exception:
        return {"changes": {}, "reason": "LLM reflect parse error; skip changes."}

def rationale_via_llm(llm: LLMClient, decision: str, metrics: dict, context: dict) -> str:
    base = f"Decision={decision}, Metrics={metrics}, Context={json.dumps(context, ensure_ascii=False)}"
    if not llm.available:
        return (f"สรุป: ตัดสินใจ {decision}. rel={metrics.get('rel'):.3f}."
                " โมเดลเทียบกับ naive ดีขึ้นเล็กน้อยหรือยังไม่ผ่านเกณฑ์ตามกรณี."
                " ควรติดตามความผันผวน/ปริมาณการซื้อขาย และปัจจัยข่าวสารเพิ่มเติม.")
    sys_p = SYSTEM_AGENT + "\nWrite like a sell-side market analyst. Be clear and risk-aware."
    user_p = f"""
{base}

Write a short report-style rationale (5-7 bullet lines), Thai language preferred, with:
- ภาพรวมสั้น ๆ
- มุมมองเชิงสัญญาณ/ตัวชี้วัด (RSI, volume z-score)
- ข้อจำกัดของแบบจำลอง/ข้อมูล
- มุมมองเชิงกลยุทธ์ (BUY/HOLD/WAIT) พร้อมเงื่อนไขที่ควรติดตาม
"""
    out = llm.chat(sys_p, user_p)
    return out or "ไม่สามารถสร้างถ้อยคำ rationale จาก LLM ได้"

# ---------------------- Agent with logging callbacks ----------------------
class FinanceAgentLLM:
    def __init__(self, max_loops=3, llm_model="gpt-4o-mini", logger: Callable[[str, Dict[str, Any]], None]=None):
        self.max_loops = max_loops
        self.short: Dict[str, Any] = {}
        self.long: Dict[str, Any] = {}
        self.llm = LLMClient(model=llm_model)
        self.logger = logger or (lambda evt, data: None)

    def log(self, event: str, data: Dict[str, Any]):
        self.logger(event, data)

    def plan(self, goal):
        available_tools = {
            "fetch_price": "symbol->df[open,high,low,close,volume]",
            "build_features": "price_df->feat_df[ret_1d,ret_5d,vol_z,rsi14]",
            "make_target": "price+horizon->forward return",
            "train_rf": "Xtr,ytr->model",
            "evaluate_regression": "y_true,y_pred,naive->mae,mae_naive,rel"
        }
        plan = plan_via_llm(self.llm, goal, available_tools)
        allowed = {"fetch_data","build_xy","train_eval","tune_params","add_features","summarize"}
        plan = [s for s in plan if isinstance(s, dict) and s.get("step") in allowed]
        if not plan:
            plan = [{"step":"fetch_data"},{"step":"build_xy"},{"step":"train_eval"},{"step":"summarize"}]
        self.log("plan", {"steps": plan})
        return plan

    def act(self, step, goal):
        symbol = goal.get("symbol", "PTT.BK")
        horizon = self.short.get("horizon", goal.get("horizon", 5))

        if step["step"] == "fetch_data":
            self.log("step_start", {"step": "fetch_data", "symbol": symbol})
            price = fetch_price(symbol, period=goal.get("period","2y"), interval="1d")
            feats = build_features(price)
            self.short["price"] = price
            self.short["features"] = feats
            self.log("step_done", {"step":"fetch_data", "rows": len(price), "last_date": str(price.index[-1].date())})

        elif step["step"] == "build_xy":
            self.log("step_start", {"step": "build_xy", "horizon": horizon})
            price = self.short["price"]; feats = self.short["features"]
            y = make_target(price, horizon, feats.index)
            X = feats.loc[y.index].copy()
            use_rsi = self.short.get("use_rsi", True)
            use_vol = self.short.get("use_vol", True)
            keep = ["ret_1d","ret_5d"] + (["rsi14"] if use_rsi else []) + (["vol_z"] if use_vol else [])
            X = X[keep]
            self.short["feature_names"] = keep
            self.short["Xy"] = (X, y)
            self.log("step_done", {"step":"build_xy", "samples": len(X), "features": keep})

        elif step["step"] == "tune_params":
            self.log("step_start", {"step": "tune_params"})
            self.short["n_estimators"] = 500 if self.short.get("n_estimators",300)==300 else 300
            self.short["min_samples_leaf"] = 2 if self.short.get("min_samples_leaf",1)==1 else 1
            self.log("step_done", {"step":"tune_params", "n_estimators": self.short["n_estimators"], "min_samples_leaf": self.short["min_samples_leaf"]})

        elif step["step"] == "add_features":
            self.log("step_start", {"step": "add_features"})
            self.short["use_rsi"] = not self.short.get("use_rsi", True)
            self.short["use_vol"] = not self.short.get("use_vol", True)
            self.log("step_done", {"step":"add_features", "use_rsi": self.short["use_rsi"], "use_vol": self.short["use_vol"]})

        elif step["step"] == "train_eval":
            self.log("step_start", {"step": "train_eval"})
            X, y = self.short["Xy"]
            if len(X) < 200:
                raise ValueError("Samples too few (<200).")
            Xtr, Xte, ytr, yte = split_train_test(X, y, test_ratio=0.2, seed=42)
            model = train_rf(
                Xtr, ytr,
                n_estimators=self.short.get("n_estimators", 300),
                min_samples_leaf=self.short.get("min_samples_leaf", 1),
                random_state=42
            )
            yhat = model.predict(Xte)
            naive = np.roll(yte.values, 1); naive[0] = 0.0
            report = evaluate_regression(yte.values, yhat, naive, thresh=goal.get("rel_thresh", 0.98))
            self.short["model"] = model
            self.short["eval"] = report
            self.log("eval", {"mae": report.mae, "mae_naive": report.mae_naive, "rel": report.rel, "passed": report.passed})

        elif step["step"] == "summarize":
            r = self.short["eval"]
            decision = "HOLD"
            if r.passed and r.rel < 0.92:
                decision = "BUY"
            elif not r.passed:
                decision = "WAIT"
            metrics = {"mae": r.mae, "mae_naive": r.mae_naive, "rel": r.rel}
            context = {
                "symbol": goal.get("symbol"),
                "horizon_days": self.short.get("horizon"),
                "features": self.short.get("feature_names", []),
                "notes": "Minimal demo; no sentiment/news yet."
            }
            rationale = rationale_via_llm(LLMClient(), decision, metrics, context)
            self.short["summary"] = {
                "metrics": metrics,
                "decision": decision,
                "rationale_text": rationale,
                "data_gaps": []
            }
            self.log("summary", self.short["summary"])

    def reflect(self):
        r: EvalReport = self.short.get("eval")
        if r and not r.passed:
            suggestion = reflect_via_llm(LLMClient(), r, self.short)
            changes = suggestion.get("changes", {})
            for k, v in changes.items():
                if v is not None:
                    self.short[k] = v
            if "horizon" not in changes or changes["horizon"] is None:
                horizon_cycle = [5,10,20]
                cur = self.short.get("horizon",5)
                self.short["horizon"] = horizon_cycle[(horizon_cycle.index(cur)+1)%len(horizon_cycle)]
            self.short["reflect_reason"] = suggestion.get("reason","")
            self.log("reflect", {"applied_changes": changes, "reason": self.short["reflect_reason"]})

    def run(self, goal):
        self.short.setdefault("horizon", goal.get("horizon", 5))
        self.short.setdefault("use_rsi", True)
        self.short.setdefault("use_vol", True)
        self.short.setdefault("n_estimators", 300)
        self.short.setdefault("min_samples_leaf", 1)

        for loop in range(self.max_loops):
            self.log("loop", {"loop": loop+1})
            plan = self.plan(goal)
            for step in plan:
                self.act(step, goal)
                time.sleep(0.2)  # tiny pause so UI updates look "live"
            if self.short["eval"].passed:
                break
            self.reflect()
            time.sleep(0.2)
        return self.short.get("summary", {})

# ---------------------- Streamlit UI ----------------------
st.set_page_config(
    page_title="🚀 Space Finance Agent", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for space theme
st.markdown("""
<style>
    /* Space Theme CSS */
    .main > div {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    .stApp > header {
        background-color: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 2px solid #00d4ff;
    }
    
    .stSidebar .stSelectbox > div > div {
        background-color: rgba(0, 212, 255, 0.1);
        border: 1px solid #00d4ff;
        color: #ffffff;
    }
    
    .stSidebar .stTextInput > div > div > input {
        background-color: rgba(0, 212, 255, 0.1);
        border: 1px solid #00d4ff;
        color: #ffffff;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #00d4ff, #0099cc);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
    }
    
    .stCheckbox > div > div {
        background-color: rgba(0, 212, 255, 0.1);
        border: 1px solid #00d4ff;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #0099cc);
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(45deg, #00d4ff, #ffffff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Container styling */
    .stContainer {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Chart container */
    .element-container {
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* JSON styling */
    .stJson {
        background: rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid #00d4ff;
        border-radius: 10px;
    }
    
    .stError {
        background: rgba(255, 0, 0, 0.1);
        border: 1px solid #ff0000;
        border-radius: 10px;
    }
    
    /* Sidebar header */
    .stSidebar h2 {
        color: #00d4ff;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        margin: 2rem 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d4ff, #0099cc);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #0099cc, #00d4ff);
    }
    
    /* Animated background stars */
    body::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #00d4ff, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(0, 212, 255, 0.5), transparent),
            radial-gradient(1px 1px at 90px 40px, #ffffff, transparent),
            radial-gradient(1px 1px at 130px 80px, rgba(255, 255, 255, 0.5), transparent),
            radial-gradient(2px 2px at 160px 30px, #00d4ff, transparent);
        background-repeat: repeat;
        background-size: 200px 100px;
        animation: sparkle 20s linear infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes sparkle {
        from { transform: translateY(0px); }
        to { transform: translateY(-100px); }
    }
    
    /* Glowing effects */
    .glow {
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        animation: pulse 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        from { box-shadow: 0 0 20px rgba(0, 212, 255, 0.5); }
        to { box-shadow: 0 0 30px rgba(0, 212, 255, 0.8); }
    }
    
    /* Enhanced button hover effects */
    .stButton > button:hover {
        background: linear-gradient(45deg, #00d4ff, #ffffff, #00d4ff);
        background-size: 200% 200%;
        animation: gradientShift 1s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Code blocks styling */
    code {
        background: rgba(0, 0, 0, 0.5);
        color: #00d4ff;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Enhanced markdown styling */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    /* Loading animation */
    .stSpinner {
        border: 3px solid rgba(0, 212, 255, 0.3);
        border-top: 3px solid #00d4ff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Metrics styling */
    .stMetric {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .stMetric > div {
        color: #ffffff;
    }
    
    .stMetric > div > div > div {
        color: #00d4ff;
        font-weight: bold;
    }
    
    .stMetric > div > div > div[data-testid="metric-delta"] {
        color: #00d4ff;
    }
    
    /* Price display enhancements */
    .price-display {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 0, 0, 0.3));
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    .price-positive {
        color: #00ff00;
        text-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
    }
    
    .price-negative {
        color: #ff0000;
        text-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Main title with space theme
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1>🚀 Space Finance Agent</h1>
    <p style="color: #00d4ff; font-size: 1.2rem; margin-top: -1rem;">
        🌌 Real-time AI reasoning • Planning → Acting → Evaluating → Reflecting 🌌
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>🚀 Mission Control</h2>
        <p style="color: #00d4ff; font-size: 0.9rem;">Configure your space mission</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🌟 Target Selection")
    
    # Predefined symbols that work well
    symbol_options = {
        "🌍 Global Stocks": {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corp.",
            "GOOGL": "Google (Alphabet)",
            "TSLA": "Tesla Inc.",
            "AMZN": "Amazon.com Inc."
        },
        "🇹🇭 Thai Market": {
            "^SETI": "SET Index",
            "^SET50": "SET50 Index",
            "PTT.BK": "PTT Public Company",
            "DELTA.BK": "Delta Electronics Thailand"
        },
        "🏦 Financial": {
            "JPM": "JPMorgan Chase",
            "BAC": "Bank of America",
            "WFC": "Wells Fargo"
        }
    }
    
    # Create selectbox for symbol selection
    symbol_choice = st.selectbox(
        "🎯 Choose Symbol",
        options=list(symbol_options.keys()),
        help="Select a category to see available symbols"
    )
    
    if symbol_choice:
        selected_symbol = st.selectbox(
            "📊 Select Specific Symbol",
            options=list(symbol_options[symbol_choice].keys()),
            format_func=lambda x: f"{x} - {symbol_options[symbol_choice][x]}",
            help="Choose a specific symbol from the selected category"
        )
        symbol = selected_symbol
    else:
        symbol = st.text_input("🎯 Custom Symbol", value="AAPL", help="Enter any valid Yahoo Finance symbol")
    
    st.markdown("### ⏰ Time Parameters")
    horizon = st.selectbox("🔮 Horizon (days forward return)", options=[5,10,20], index=0)
    period = st.selectbox("📅 Download period", options=["2y","3y","5y"], index=0)
    
    st.markdown("### 🎛️ Mission Settings")
    rel_thresh = st.slider("🎯 rel threshold (MAE/MAE_naive)", min_value=0.90, max_value=1.10, value=0.98, step=0.01)
    max_loops = st.slider("🔄 Max loops", 1, 5, 3)
    
    st.markdown("### 🔧 Technical Indicators")
    use_rsi = st.checkbox("📊 Use RSI", value=True)
    use_vol = st.checkbox("📈 Use Volume Z-score", value=True)
    
    st.markdown("---")
    
    # Quick Price Preview
    st.markdown("### 💰 Quick Price Preview")
    try:
        # Try to get real data first
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        try:
            # Try real Yahoo Finance data
            real_df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), 
                                end=end_date.strftime("%Y-%m-%d"), 
                                interval="1d", auto_adjust=True, progress=False)
            
            if real_df is not None and not real_df.empty:
                # Don't rename columns - keep original Yahoo Finance format
                real_df = real_df.dropna()
                
                if len(real_df) > 0:
                    # Handle MultiIndex columns from Yahoo Finance
                    if isinstance(real_df.columns, pd.MultiIndex):
                        close_col = real_df.columns[0]  # First column is usually Close
                    else:
                        close_col = "Close" if "Close" in real_df.columns else "close"
                    
                    latest_price = float(real_df[close_col].iloc[-1])
                    prev_price = float(real_df[close_col].iloc[-2]) if len(real_df) > 1 else latest_price
                    price_change = latest_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
                    
                    change_color = "#00ff00" if price_change >= 0 else "#ff0000"
                    change_icon = "📈" if price_change >= 0 else "📉"
                    
                    st.markdown(f"""
                    <div style="background: rgba(0, 212, 255, 0.1); border: 1px solid #00d4ff; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;">
                        <h4 style="color: #00d4ff; margin: 0 0 0.5rem 0;">{symbol} (Real Data)</h4>
                        <h3 style="color: #ffffff; margin: 0 0 0.5rem 0;">{latest_price:.2f}</h3>
                        <p style="color: {change_color}; margin: 0; font-weight: bold;">
                            {change_icon} {price_change:+.2f} ({price_change_pct:+.2f}%)
                        </p>
                        <p style="color: #00d4ff; margin: 0; font-size: 0.8rem;">✅ Live Data</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    raise ValueError("No real data")
            else:
                raise ValueError("No real data")
                
        except Exception:
            # Fallback to dummy data with warning
            preview_df = fetch_price(symbol, period="5d", interval="1d")
            if preview_df is not None and not preview_df.empty:
                # Handle MultiIndex columns from Yahoo Finance
                if isinstance(preview_df.columns, pd.MultiIndex):
                    close_col = preview_df.columns[0]  # First column is usually Close
                else:
                    close_col = "Close" if "Close" in preview_df.columns else "close"
                
                latest_price = float(preview_df[close_col].iloc[-1])
                prev_price = float(preview_df[close_col].iloc[-2]) if len(preview_df) > 1 else latest_price
                price_change = latest_price - prev_price
                price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
                
                change_color = "#ffa500"  # Orange for demo data
                change_icon = "📈" if price_change >= 0 else "📉"
                
                st.markdown(f"""
                <div style="background: rgba(255, 165, 0, 0.1); border: 1px solid #ffa500; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;">
                    <h4 style="color: #ffa500; margin: 0 0 0.5rem 0;">{symbol} (Demo Data)</h4>
                    <h3 style="color: #ffffff; margin: 0 0 0.5rem 0;">{latest_price:.2f}</h3>
                    <p style="color: {change_color}; margin: 0; font-weight: bold;">
                        {change_icon} {price_change:+.2f} ({price_change_pct:+.2f}%)
                    </p>
                    <p style="color: #ffa500; margin: 0; font-size: 0.8rem;">⚠️ Demo Data (API Offline)</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(255, 0, 0, 0.1); border: 1px solid #ff0000; border-radius: 10px; padding: 0.5rem; margin: 0.5rem 0;">
                    <p style="color: #ff0000; margin: 0; font-size: 0.9rem;">❌ No price data available</p>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.markdown("""
        <div style="background: rgba(255, 0, 0, 0.1); border: 1px solid #ff0000; border-radius: 10px; padding: 0.5rem; margin: 0.5rem 0;">
            <p style="color: #ff0000; margin: 0; font-size: 0.9rem;">❌ Price preview unavailable</p>
        </div>
        """, unsafe_allow_html=True)
    
    # API Status Check
    st.markdown("### 🔗 API Status")
    try:
        # Quick test of Yahoo Finance API
        test_df = yf.download("AAPL", period="5d", progress=False)
        if test_df is not None and not test_df.empty:
            st.markdown("""
            <div style="background: rgba(0, 255, 0, 0.1); border: 1px solid #00ff00; border-radius: 10px; padding: 0.5rem; margin: 0.5rem 0;">
                <p style="color: #00ff00; margin: 0; font-size: 0.9rem;">✅ Yahoo Finance API: Connected</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(255, 165, 0, 0.1); border: 1px solid #ffa500; border-radius: 10px; padding: 0.5rem; margin: 0.5rem 0;">
                <p style="color: #ffa500; margin: 0; font-size: 0.9rem;">⚠️ Yahoo Finance API: Limited</p>
            </div>
            """, unsafe_allow_html=True)
    except:
        st.markdown("""
        <div style="background: rgba(255, 0, 0, 0.1); border: 1px solid #ff0000; border-radius: 10px; padding: 0.5rem; margin: 0.5rem 0;">
            <p style="color: #ff0000; margin: 0; font-size: 0.9rem;">❌ Yahoo Finance API: Offline</p>
        </div>
        """, unsafe_allow_html=True)
    
    run_btn = st.button("🚀 Launch Mission", use_container_width=True, key="launch_mission")
    
    # Add glow effect to the button
    if run_btn:
        st.markdown("""
        <style>
        .stButton > button[key="launch_mission"] {
            animation: pulse 1s ease-in-out infinite alternate;
        }
        </style>
        """, unsafe_allow_html=True)

# Main content layout with space theme
st.markdown("""
<div style="background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 15px; padding: 1rem; margin: 1rem 0; backdrop-filter: blur(10px);">
    <h3 style="color: #00d4ff; text-align: center; margin-bottom: 1rem;">🌌 Mission Dashboard</h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("""
    <div style="background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 15px; padding: 1rem; margin: 1rem 0; backdrop-filter: blur(10px);">
        <h4 style="color: #00d4ff; text-align: center;">📡 Real-time Mission Log</h4>
    </div>
    """, unsafe_allow_html=True)
    log_container = st.container()

with col2:
    st.markdown("""
    <div style="background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 15px; padding: 1rem; margin: 1rem 0; backdrop-filter: blur(10px);">
        <h4 style="color: #00d4ff; text-align: center;">🎯 Mission Summary</h4>
    </div>
    """, unsafe_allow_html=True)
    summary_container = st.container()

st.markdown("""
<div style="background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 15px; padding: 1rem; margin: 1rem 0; backdrop-filter: blur(10px);">
    <h4 style="color: #00d4ff; text-align: center;">📊 Market Data Visualization</h4>
</div>
""", unsafe_allow_html=True)
price_chart_container = st.container()

st.markdown("""
<hr style="border: none; height: 2px; background: linear-gradient(90deg, transparent, #00d4ff, transparent); margin: 2rem 0;">
""", unsafe_allow_html=True)

# Session logs
if "logs" not in st.session_state:
    st.session_state.logs: List[Dict[str, Any]] = []

def logger(event: str, data: Dict[str, Any]):
    st.session_state.logs.append({"event": event, "data": data})
    # Live update rendering with space theme
    with log_container:
        st.markdown(f"""
        <div style="background: rgba(0, 212, 255, 0.1); border: 1px solid #00d4ff; border-radius: 10px; padding: 0.5rem; margin: 0.5rem 0;">
            <strong style="color: #00d4ff;">📡 Event:</strong> <code style="color: #ffffff;">{event}</code>
        </div>
        """, unsafe_allow_html=True)
        st.json(data)

if run_btn:
    # reset logs
    st.session_state.logs = []
    # Run agent
    agent = FinanceAgentLLM(max_loops=max_loops, logger=logger)
    # Initialize defaults that user toggled
    agent.short["use_rsi"] = use_rsi
    agent.short["use_vol"] = use_vol
    goal = {"symbol": symbol, "horizon": horizon, "period": period, "rel_thresh": float(rel_thresh)}

    try:
        # Pre-fetch just for chart
        with st.spinner("🔄 Fetching market data..."):
            price_df = fetch_price(symbol, period=period, interval="1d")
        
        with price_chart_container:
            st.markdown("### 📊 Price Chart (Close)")
            
            # Try to get real data first
            try:
                from datetime import datetime, timedelta
                end_date = datetime.now()
                if period == "2y":
                    start_date = end_date - timedelta(days=730)
                elif period == "3y":
                    start_date = end_date - timedelta(days=1095)
                elif period == "5y":
                    start_date = end_date - timedelta(days=1825)
                else:
                    start_date = end_date - timedelta(days=730)
                
                # Try real Yahoo Finance data
                real_df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), 
                                    end=end_date.strftime("%Y-%m-%d"), 
                                    interval="1d", auto_adjust=True, progress=False)
                
                if real_df is not None and not real_df.empty:
                    # Don't rename columns - keep original Yahoo Finance format
                    real_df = real_df.dropna()
                    
                    if len(real_df) > 0:
                        # Use real data
                        price_df = real_df
                        data_type = "Real Data"
                        data_color = "#00ff00"
                        data_icon = "✅"
                    else:
                        raise ValueError("No real data")
                else:
                    raise ValueError("No real data")
                    
            except Exception:
                # Use dummy data
                data_type = "Demo Data"
                data_color = "#ffa500"
                data_icon = "⚠️"
            
            # Calculate price statistics - handle MultiIndex columns from Yahoo Finance
            if isinstance(price_df.columns, pd.MultiIndex):
                # Yahoo Finance format: [('Close', 'SYMBOL'), ('High', 'SYMBOL'), ...]
                close_col = price_df.columns[0]  # First column is usually Close
                high_col = price_df.columns[1]   # Second column is usually High
                low_col = price_df.columns[2]    # Third column is usually Low
                volume_col = price_df.columns[4] # Fifth column is usually Volume
            else:
                # Regular format: ['Close', 'High', 'Low', ...]
                close_col = "Close" if "Close" in price_df.columns else "close"
                high_col = "High" if "High" in price_df.columns else "high"
                low_col = "Low" if "Low" in price_df.columns else "low"
                volume_col = "Volume" if "Volume" in price_df.columns else "volume"
            
            latest_price = float(price_df[close_col].iloc[-1])
            prev_price = float(price_df[close_col].iloc[-2]) if len(price_df) > 1 else latest_price
            price_change = latest_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            high_52w = float(price_df[high_col].max())
            low_52w = float(price_df[low_col].min())
            volume_avg = float(price_df[volume_col].mean())
            
            # Show current price with change
            change_color = "#00ff00" if price_change >= 0 else "#ff0000"
            change_icon = "📈" if price_change >= 0 else "📉"
            
            st.markdown(f"""
            <div style="background: rgba(0, 212, 255, 0.1); border: 1px solid #00d4ff; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                <h5 style="color: #00d4ff;">💰 Current Price</h5>
                <div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0;">
                    <div style="flex: 1;">
                        <h2 style="color: #ffffff; margin: 0; font-size: 2.5rem;">{latest_price:.2f}</h2>
                        <p style="color: #00d4ff; margin: 0; font-size: 1.1rem;">{symbol}</p>
                    </div>
                    <div style="flex: 1; text-align: right;">
                        <p style="color: {change_color}; margin: 0; font-size: 1.2rem; font-weight: bold;">
                            {change_icon} {price_change:+.2f} ({price_change_pct:+.2f}%)
                        </p>
                        <p style="color: #ffffff; margin: 0; font-size: 0.9rem;">
                            vs Previous Close
                        </p>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 0.5rem;">
                    <p style="color: {data_color}; margin: 0; font-size: 0.9rem; font-weight: bold;">
                        {data_icon} {data_type}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show detailed statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="📊 52-Week High",
                    value=f"{high_52w:.2f}",
                    delta=f"{((latest_price - high_52w) / high_52w * 100):.1f}% from high"
                )
            
            with col2:
                st.metric(
                    label="📉 52-Week Low", 
                    value=f"{low_52w:.2f}",
                    delta=f"{((latest_price - low_52w) / low_52w * 100):.1f}% from low"
                )
            
            with col3:
                st.metric(
                    label="📈 Avg Volume",
                    value=f"{volume_avg:,.0f}",
                    delta=f"{((float(price_df[volume_col].iloc[-1]) - volume_avg) / volume_avg * 100):+.1f}%"
                )
            
            # Show data info
            st.markdown(f"""
            <div style="background: rgba(0, 212, 255, 0.1); border: 1px solid #00d4ff; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                <h5 style="color: #00d4ff;">📈 Data Information</h5>
                <p style="color: #ffffff; margin: 0.5rem 0;">
                    <strong>Symbol:</strong> {symbol}<br>
                    <strong>Period:</strong> {period}<br>
                    <strong>Data Points:</strong> {len(price_df)} days<br>
                    <strong>Date Range:</strong> {price_df.index[0].strftime('%Y-%m-%d')} to {price_df.index[-1].strftime('%Y-%m-%d')}<br>
                    <strong>Last Update:</strong> {price_df.index[-1].strftime('%Y-%m-%d %H:%M')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Handle MultiIndex columns for chart
            if isinstance(price_df.columns, pd.MultiIndex):
                close_col = price_df.columns[0]  # First column is usually Close
            else:
                close_col = "Close" if "Close" in price_df.columns else "close"
            
            st.line_chart(price_df[[close_col]].rename(columns={close_col: symbol}))

        summary = agent.run(goal)

        with summary_container:
            st.markdown("### 🎯 Mission Results")
            if summary:
                st.markdown("""
                <div style="background: rgba(0, 212, 255, 0.1); border: 1px solid #00d4ff; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                    <h5 style="color: #00d4ff;">📋 Mission Data</h5>
                </div>
                """, unsafe_allow_html=True)
                st.json(summary)
                
                st.markdown("""
                <div style="background: rgba(0, 212, 255, 0.1); border: 1px solid #00d4ff; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                    <h5 style="color: #00d4ff;">🧠 AI Analysis Report</h5>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(summary.get("rationale_text", ""))
            else:
                st.markdown("""
                <div style="background: rgba(255, 165, 0, 0.1); border: 1px solid #ffa500; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                    <p style="color: #ffa500;">⚠️ No mission summary produced.</p>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div style="background: rgba(255, 0, 0, 0.1); border: 1px solid #ff0000; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
            <h5 style="color: #ff0000;">🚨 Mission Error</h5>
            <p style="color: #ff0000;">{e}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border: 1px solid #00d4ff; border-radius: 15px; padding: 2rem; margin: 2rem 0; text-align: center;">
        <h4 style="color: #00d4ff; margin-bottom: 1rem;">🚀 Ready for Launch</h4>
        <p style="color: #ffffff; font-size: 1.1rem;">
            Configure your mission parameters in the sidebar, then click <strong style="color: #00d4ff;">🚀 Launch Mission</strong> to begin real-time AI analysis!
        </p>
        <p style="color: #00d4ff; font-size: 0.9rem; margin-top: 1rem;">
            🌌 Watch the AI agent plan, act, evaluate, and reflect in real-time 🌌
        </p>
    </div>
    """, unsafe_allow_html=True)

