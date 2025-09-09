# คู่มือการใช้งาน Finance Agent Tools

## คำแนะนำเชิงปฏิบัติ

### 1. การใช้สัญลักษณ์หุ้น

#### หุ้นไทย (Thai Stocks)
ใช้ suffix `.BK` หลังชื่อหุ้น:
```python
# ตัวอย่างหุ้นไทย
symbols = [
    "PTT.BK",      # ปตท.
    "KBANK.BK",    # ธนาคารกสิกรไทย
    "SCB.BK",      # ธนาคารไทยพาณิชย์
    "CPALL.BK",    # ซีพี ออลล์
    "ADVANC.BK",   # เอไอเอส
    "TRUE.BK"      # ทรู คอร์ปอเรชั่น
]
```

#### หุ้นอเมริกัน (US Stocks)
ใช้สัญลักษณ์ตรง ๆ:
```python
# ตัวอย่างหุ้นอเมริกัน
symbols = [
    "AAPL",        # Apple Inc.
    "MSFT",        # Microsoft Corporation
    "TSLA",        # Tesla Inc.
    "GOOGL",       # Alphabet Inc. (Google)
    "AMZN",        # Amazon.com Inc.
    "META"         # Meta Platforms Inc.
]
```

#### ดัชนี (Indices)
```python
# ตัวอย่างดัชนี
indices = [
    "^SETI",       # SET Index (ไทย)
    "^GSPC",       # S&P 500 (อเมริกา)
    "^IXIC",       # NASDAQ Composite
    "^DJI"         # Dow Jones Industrial Average
]
```

### 2. การดึงข้อมูลล่าสุด

สำหรับข้อมูลวันล่าสุด ให้ใช้ `period="5d"` แล้วใช้ `tail(1)` เพื่อหลีกเลี่ยงวันหยุด:

```python
from agent.tools import DataLoader

loader = DataLoader()

# ดึงข้อมูลล่าสุด (หลีกเลี่ยงวันหยุด)
result = loader.fetch_price_data("PTT.BK", period="5d")
if result.success:
    # ระบบจะใช้ tail(1) อัตโนมัติ
    latest_data = result.data.iloc[-1]
    print(f"ราคาปิดล่าสุด: {latest_data['close']:.2f}")
```

### 3. การจัดการข้อผิดพลาดและ Logging

ระบบมีการ logging ข้อผิดพลาดที่ละเอียดขึ้น:

```python
# ตัวอย่างการจัดการข้อผิดพลาด
result = loader.fetch_price_data("INVALID_SYMBOL")
if not result.success:
    print(f"ข้อผิดพลาด: {result.error}")
    # ระบบจะแสดงข้อมูลเพิ่มเติม:
    # - Error Type
    # - Error Message  
    # - Rate limit detection
    # - Network timeout detection
```

### 4. การใช้ Proxy

สำหรับสภาพแวดล้อมที่เน็ตถูกกรอง/บล็อก:

```python
# ตั้งค่า proxy
proxy_config = {
    "http": "http://proxy.company.com:8080",
    "https": "https://proxy.company.com:8080"
}

# สร้าง DataLoader พร้อม proxy
loader = DataLoader(proxy_config=proxy_config)

# หรือเปลี่ยน proxy ภายหลัง
loader.set_proxy({
    "http": "http://new-proxy:3128",
    "https": "https://new-proxy:3128"
})

# ลบ proxy
loader.clear_proxy()
```

### 5. ตัวอย่างการใช้งานครบถ้วน

```python
#!/usr/bin/env python3
from agent.tools import DataLoader, IndicatorCalculator

def analyze_stock(symbol, period="6mo"):
    """วิเคราะห์หุ้นแบบครบถ้วน"""
    
    # สร้าง tools
    loader = DataLoader()
    calc = IndicatorCalculator()
    
    # ดึงข้อมูลราคา
    print(f"📊 กำลังดึงข้อมูล {symbol}...")
    price_result = loader.fetch_price_data(symbol, period=period)
    
    if not price_result.success:
        print(f"❌ ไม่สามารถดึงข้อมูลได้: {price_result.error}")
        return None
    
    print(f"✅ ได้ข้อมูล {len(price_result.data)} แถว")
    
    # คำนวณ technical indicators
    print("📈 กำลังคำนวณ technical indicators...")
    indicators_result = calc.calculate_indicators(
        price_result.data,
        indicators=['rsi', 'sma', 'macd', 'bollinger']
    )
    
    if not indicators_result.success:
        print(f"❌ ไม่สามารถคำนวณ indicators ได้: {indicators_result.error}")
        return None
    
    # แสดงผลลัพธ์
    latest = indicators_result.data.iloc[-1]
    print(f"\n📊 ผลการวิเคราะห์ {symbol}:")
    print(f"   ราคาปิดล่าสุด: {latest['close']:.2f}")
    print(f"   RSI(14): {latest['rsi_14']:.2f}")
    print(f"   SMA(20): {latest['sma_20']:.2f}")
    print(f"   MACD: {latest['macd']:.4f}")
    
    return indicators_result.data

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # วิเคราะห์หุ้นไทย
    analyze_stock("PTT.BK")
    
    # วิเคราะห์หุ้นอเมริกัน
    analyze_stock("AAPL")
```

### 6. การจัดการ Network Issues

หากพบปัญหา network:

1. **Rate Limiting**: ระบบจะตรวจจับและรออัตโนมัติ
2. **Network Timeout**: ระบบจะ retry 3 ครั้ง
3. **Proxy Required**: ใช้ `DataLoader(proxy_config=...)`
4. **Mock Data**: ระบบจะสร้าง mock data หาก API ไม่พร้อมใช้งาน

### 7. Best Practices

1. **ใช้ Cache**: ระบบมี cache 5 นาที เพื่อลด API calls
2. **Error Handling**: ตรวจสอบ `result.success` เสมอ
3. **Period Selection**: ใช้ `"5d"` สำหรับข้อมูลล่าสุด
4. **Proxy Setup**: ตั้งค่า proxy หากอยู่ใน corporate network
5. **Logging**: ดู log messages เพื่อ debug ปัญหา

### 8. Troubleshooting

#### ปัญหา: ไม่สามารถดึงข้อมูลได้
```python
# ตรวจสอบสัญลักษณ์
symbol = "PTT.BK"  # ใช้ .BK สำหรับหุ้นไทย

# ตรวจสอบ network
result = loader.fetch_price_data(symbol)
if not result.success:
    print(f"Error: {result.error}")
    # อาจต้องใช้ proxy หรือเปลี่ยน network
```

#### ปัญหา: Rate Limiting
```python
# ระบบจะจัดการอัตโนมัติ แต่สามารถเพิ่ม delay ได้
import time
time.sleep(1)  # รอ 1 วินาทีระหว่าง requests
```

#### ปัญหา: Proxy Issues
```python
# ทดสอบ proxy
proxy_config = {"http": "http://proxy:port"}
loader = DataLoader(proxy_config=proxy_config)

# หรือใช้ environment variables
import os
os.environ['HTTP_PROXY'] = 'http://proxy:port'
os.environ['HTTPS_PROXY'] = 'https://proxy:port'
```

## สรุป

Finance Agent Tools ได้รับการปรับปรุงให้รองรับ:
- ✅ สัญลักษณ์หุ้นไทย (.BK) และอเมริกัน
- ✅ การดึงข้อมูลล่าสุดด้วย period="5d" + tail(1)
- ✅ Error logging ที่ละเอียดขึ้น
- ✅ รองรับ proxy สำหรับสภาพแวดล้อมที่เน็ตถูกกรอง
- ✅ Mock data fallback เมื่อ API ไม่พร้อมใช้งาน
