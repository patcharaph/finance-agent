
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
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data for symbol: {symbol}. Try a different ticker (e.g., PTT.BK).")
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    return df.dropna()

def build_features(price: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=price.index)
    feat["ret_1d"] = price["close"].pct_change()
    feat["ret_5d"] = price["close"].pct_change(5)
    vol_ma = price["volume"].rolling(20).mean()
    vol_sd = price["volume"].rolling(20).std()
    feat["vol_z"] = (price["volume"] - vol_ma) / (vol_sd + 1e-9)
    delta = price["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    feat["rsi14"] = 100 - (100 / (1 + rs))
    return feat.dropna()

def make_target(price: pd.DataFrame, horizon_days: int, align_index: pd.Index) -> pd.Series:
    close = price["close"]
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
st.set_page_config(page_title="LLM-Agentic Finance Analyst", page_icon="📈", layout="wide")

st.title("📈 LLM-Agentic Finance Analyst")
st.caption("Real-time reasoning log • Planning → Acting → Evaluating → Reflecting")

with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.text_input("Symbol", value="PTT.BK", help="ตัวอย่างหุ้นไทย: PTT.BK, DELTA.BK | ดัชนี: ^SETI")
    horizon = st.selectbox("Horizon (days forward return)", options=[5,10,20], index=0)
    period = st.selectbox("Download period", options=["2y","3y","5y"], index=0)
    rel_thresh = st.slider("rel threshold (MAE/MAE_naive)", min_value=0.90, max_value=1.10, value=0.98, step=0.01)
    max_loops = st.slider("Max loops", 1, 5, 3)
    use_rsi = st.checkbox("Use RSI", value=True)
    use_vol = st.checkbox("Use Volume Z-score", value=True)
    run_btn = st.button("▶️ Run Agent")

# Placeholders
col1, col2 = st.columns([2,1])
log_container = col1.container()
summary_container = col2.container()

price_chart_container = st.container()
st.divider()

# Session logs
if "logs" not in st.session_state:
    st.session_state.logs: List[Dict[str, Any]] = []

def logger(event: str, data: Dict[str, Any]):
    st.session_state.logs.append({"event": event, "data": data})
    # Live update rendering
    with log_container:
        st.markdown(f"**Event:** `{event}`")
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
        price_df = fetch_price(symbol, period=period, interval="1d")
        with price_chart_container:
            st.subheader("Price (Close)")
            st.line_chart(price_df[["close"]].rename(columns={"close": symbol}))

        summary = agent.run(goal)

        with summary_container:
            st.subheader("📌 Summary")
            if summary:
                st.json(summary)
                st.markdown("**Rationale (Analyst-style):**")
                st.markdown(summary.get("rationale_text", ""))
            else:
                st.info("No summary produced.")
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("ใส่สัญลักษณ์ เลือกพารามิเตอร์ แล้วกด ▶️ Run Agent เพื่อเริ่มแสดง reasoning log แบบ real-time")

