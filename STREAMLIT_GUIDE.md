# คู่มือการใช้งาน Streamlit Finance Agent

## 🚀 วิธีการรัน Streamlit App

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. ตั้งค่า Environment Variables (ถ้าต้องการใช้ LLM)
```bash
# สำหรับ OpenAI
export OPENAI_API_KEY="sk-your-openai-key"

# หรือสำหรับ OpenRouter
export OPENROUTER_API_KEY="sk-or-your-openrouter-key"
export OPENROUTER_MODEL="openrouter/auto"
```

### 3. รัน Streamlit App
```bash
# รัน app หลัก
streamlit run app_demo.py

# หรือรัน app news
streamlit run app_streamlit_news.py
```

## 📱 ฟีเจอร์ของ Streamlit Apps

### app_demo.py - Finance Agent Demo
- **หน้าจอหลัก**: วิเคราะห์หุ้นแบบครบถ้วน
- **ฟีเจอร์**:
  - เลือกสัญลักษณ์หุ้น (ไทย .BK หรือ US)
  - ตั้งค่า horizon (วันทำนาย)
  - แสดง real-time reasoning log
  - แสดงกราฟ technical indicators
  - แสดงผลการวิเคราะห์

### app_streamlit_news.py - News & Analysis
- **หน้าจอหลัก**: วิเคราะห์ข่าวและหุ้น
- **ฟีเจอร์**:
  - วิเคราะห์ข่าวหุ้น
  - Sentiment analysis
  - การวิเคราะห์แบบ agentic
  - Real-time reasoning

## 🎯 วิธีการใช้งาน

### 1. เปิด Browser
หลังจากรันคำสั่ง `streamlit run` จะเปิด browser อัตโนมัติที่:
```
http://localhost:8501
```

### 2. ใช้งาน Interface
1. **เลือกสัญลักษณ์หุ้น**:
   - หุ้นไทย: `PTT.BK`, `KBANK.BK`, `SCB.BK`
   - หุ้นอเมริกัน: `AAPL`, `MSFT`, `TSLA`

2. **ตั้งค่าพารามิเตอร์**:
   - Horizon: จำนวนวันทำนาย (1-30)
   - Period: ช่วงเวลาข้อมูล (1y, 2y, 5y)

3. **กดปุ่ม "Run Agent"**:
   - ระบบจะแสดง reasoning log แบบ real-time
   - แสดงกราฟ technical indicators
   - แสดงผลการวิเคราะห์

### 3. ดูผลลัพธ์
- **Real-time Log**: เห็นขั้นตอนการวิเคราะห์
- **Charts**: กราฟราคาและ indicators
- **Analysis**: ผลการวิเคราะห์และคำแนะนำ

## 🔧 การตั้งค่าขั้นสูง

### Environment Variables
```bash
# OpenAI
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini

# OpenRouter
OPENROUTER_API_KEY=sk-or-your-key
OPENROUTER_MODEL=openrouter/auto
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# App Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### Custom Configuration
```python
# ในไฟล์ app
st.set_page_config(
    page_title="Finance Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

## 📊 ตัวอย่างการใช้งาน

### 1. วิเคราะห์หุ้นไทย
```
Symbol: PTT.BK
Horizon: 5
Period: 1y
```

### 2. วิเคราะห์หุ้นอเมริกัน
```
Symbol: AAPL
Horizon: 10
Period: 2y
```

### 3. วิเคราะห์ดัชนี
```
Symbol: ^SETI
Horizon: 7
Period: 6mo
```

## 🛠️ Troubleshooting

### ปัญหา: ไม่สามารถรันได้
```bash
# ตรวจสอบ dependencies
pip list | grep streamlit

# ติดตั้งใหม่
pip install --upgrade streamlit
```

### ปัญหา: ไม่มี API Key
- ระบบจะใช้ mock data อัตโนมัติ
- ไม่จำเป็นต้องมี API key สำหรับการทดสอบ

### ปัญหา: Port ถูกใช้งาน
```bash
# ใช้ port อื่น
streamlit run app_demo.py --server.port 8502
```

### ปัญหา: ไม่สามารถเข้าถึง Yahoo Finance
- ระบบจะใช้ mock data อัตโนมัติ
- ข้อมูลจะสมจริงและใช้ได้

## 🎨 Customization

### เปลี่ยน Theme
```python
# ในไฟล์ .streamlit/config.toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### เพิ่มฟีเจอร์ใหม่
```python
# เพิ่มใน app
if st.button("New Feature"):
    # Your code here
    pass
```

## 📱 Mobile Support
- Streamlit รองรับ mobile browser
- Interface ปรับขนาดอัตโนมัติ
- Touch-friendly controls

## 🔒 Security
- ไม่เก็บ API keys ใน code
- ใช้ environment variables
- Session state management

## 📈 Performance Tips
- ใช้ caching สำหรับข้อมูลที่ดึงบ่อย
- จำกัดจำนวน requests
- ใช้ mock data สำหรับการทดสอบ

## 🎯 Best Practices
1. **ใช้สัญลักษณ์ที่ถูกต้อง**: `.BK` สำหรับหุ้นไทย
2. **ตั้งค่า horizon ที่เหมาะสม**: 1-30 วัน
3. **ดู real-time log**: เพื่อเข้าใจการวิเคราะห์
4. **ใช้ mock data**: สำหรับการทดสอบ
5. **บันทึกผลลัพธ์**: screenshot หรือ export

## 🚀 Quick Start
```bash
# 1. ติดตั้ง
pip install -r requirements.txt

# 2. รัน
streamlit run app_demo.py

# 3. เปิด browser
# http://localhost:8501

# 4. เลือกหุ้นและวิเคราะห์!
```

**🎉 พร้อมใช้งาน Finance Agent ผ่าน Streamlit แล้ว!**
