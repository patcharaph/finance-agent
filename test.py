import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date="2024-01-01", end_date="2025-01-01"):
    """ดึงข้อมูลหุ้นพร้อม fallback เป็นข้อมูลจำลอง"""
    
    print(f"🔄 กำลังดึงข้อมูล {ticker}...")
    
    try:
        # ลองดึงข้อมูลจริง
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
        
        if df is not None and not df.empty:
            print(f"✅ ดึงข้อมูล {ticker} สำเร็จ! ({len(df)} วัน)")
            return df
        else:
            raise ValueError("ข้อมูลว่างเปล่า")
            
    except Exception as e:
        print(f"❌ ไม่สามารถดึงข้อมูล {ticker} ได้: {e}")
        print("🔄 สร้างข้อมูลจำลองแทน...")
        
        # สร้างข้อมูลจำลอง
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # สร้างข้อมูลราคาที่สมจริง
        np.random.seed(42)
        n_days = len(dates)
        
        # ราคาเริ่มต้นตามสัญลักษณ์
        if "PTT" in ticker:
            initial_price = 35.0  # PTT ราคาประมาณ 35 บาท
        elif "AAPL" in ticker:
            initial_price = 180.0  # Apple ราคาประมาณ $180
        else:
            initial_price = 100.0
        
        # สร้างราคาที่มี trend และ volatility
        returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% daily return, 2% volatility
        prices = [initial_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))  # ป้องกันราคาติดลบ
        
        # สร้าง OHLCV data
        df_sim = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        # ปรับ High/Low ให้สมเหตุสมผล
        df_sim['High'] = df_sim[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, n_days))
        df_sim['Low'] = df_sim[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, n_days))
        
        print(f"✅ สร้างข้อมูลจำลอง {ticker} สำเร็จ! ({len(df_sim)} วัน)")
        return df_sim

# ทดสอบดึงข้อมูล
ticker = "PTT.BK"
df = fetch_stock_data(ticker, "2024-01-01", "2025-01-01")

# แปลง timezone เป็นเวลาไทย (Bangkok)
try:
    df = df.tz_localize("UTC").tz_convert("Asia/Bangkok")
except:
    print("⚠️ ไม่สามารถแปลง timezone ได้ ใช้เวลาปกติแทน")

print("\n📊 ข้อมูล 5 วันแรก:")
print(df.head())

print(f"\n📈 สถิติข้อมูล:")
print(f"จำนวนวัน: {len(df)}")
print(f"ราคาเริ่มต้น: {float(df['Close'].iloc[0]):.2f}")
print(f"ราคาสุดท้าย: {float(df['Close'].iloc[-1]):.2f}")
print(f"ราคาสูงสุด: {float(df['High'].max()):.2f}")
print(f"ราคาต่ำสุด: {float(df['Low'].min()):.2f}")
print(f"Volume เฉลี่ย: {float(df['Volume'].mean()):,.0f}")

# เซฟเป็น CSV
filename = f"{ticker.replace('.', '_')}_daily.csv"
df.to_csv(filename, encoding="utf-8-sig")
print(f"\n💾 บันทึกไฟล์: {filename}")
