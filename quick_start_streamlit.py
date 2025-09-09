#!/usr/bin/env python3
"""
Quick Start Streamlit App for Finance Agent
แอปพลิเคชัน Streamlit แบบง่ายสำหรับเริ่มต้นใช้งาน Finance Agent
"""

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.tools import DataLoader, IndicatorCalculator

# Page config
st.set_page_config(
    page_title="Finance Agent - Quick Start",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Finance Agent - Quick Start")
st.markdown("แอปพลิเคชันวิเคราะห์หุ้นแบบง่าย ๆ")

# Sidebar
st.sidebar.header("⚙️ การตั้งค่า")

# Symbol input
symbol = st.sidebar.text_input(
    "สัญลักษณ์หุ้น",
    value="PTT.BK",
    help="หุ้นไทย: PTT.BK, KBANK.BK, SCB.BK\nหุ้นอเมริกัน: AAPL, MSFT, TSLA"
)

# Period selection
period = st.sidebar.selectbox(
    "ช่วงเวลา",
    options=["6mo", "1y", "2y"],
    index=1,
    help="เลือกช่วงเวลาของข้อมูล"
)

# Analysis button
analyze_button = st.sidebar.button("🔍 วิเคราะห์", type="primary")

# Main content
if analyze_button:
    st.header(f"📈 การวิเคราะห์ {symbol}")
    
    # Create containers
    with st.spinner("กำลังดึงข้อมูล..."):
        # Initialize tools
        loader = DataLoader()
        calc = IndicatorCalculator()
        
        # Fetch price data
        result = loader.fetch_price_data(symbol, period=period)
        
        if result.success:
            st.success(f"✅ ได้ข้อมูล {len(result.data)} แถว")
            
            # Show metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("วันที่เริ่มต้น", result.metadata['start_date'])
            with col2:
                st.metric("วันที่สิ้นสุด", result.metadata['end_date'])
            with col3:
                st.metric("Mock Data", "ใช่" if result.metadata.get('mock_data', False) else "ไม่")
            
            # Show latest price
            latest = result.data.iloc[-1]
            st.subheader("💰 ข้อมูลล่าสุด")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ราคาเปิด", f"{latest['open']:.2f}")
            with col2:
                st.metric("ราคาสูงสุด", f"{latest['high']:.2f}")
            with col3:
                st.metric("ราคาต่ำสุด", f"{latest['low']:.2f}")
            with col4:
                st.metric("ราคาปิด", f"{latest['close']:.2f}")
            
            # Calculate technical indicators
            with st.spinner("กำลังคำนวณ technical indicators..."):
                indicators_result = calc.calculate_indicators(
                    result.data,
                    indicators=['rsi', 'sma', 'macd']
                )
                
                if indicators_result.success:
                    st.success("✅ คำนวณ technical indicators สำเร็จ")
                    
                    # Show latest indicators
                    latest_indicators = indicators_result.data.iloc[-1]
                    st.subheader("📊 Technical Indicators")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rsi = latest_indicators['rsi_14']
                        if rsi > 70:
                            rsi_color = "🔴"
                            rsi_status = "Overbought"
                        elif rsi < 30:
                            rsi_color = "🟢"
                            rsi_status = "Oversold"
                        else:
                            rsi_color = "🟡"
                            rsi_status = "Neutral"
                        
                        st.metric(
                            f"{rsi_color} RSI(14)",
                            f"{rsi:.2f}",
                            help=f"Status: {rsi_status}"
                        )
                    
                    with col2:
                        sma = latest_indicators['sma_20']
                        price = latest['close']
                        if price > sma:
                            sma_status = "Above SMA - Bullish"
                            sma_color = "🟢"
                        else:
                            sma_status = "Below SMA - Bearish"
                            sma_color = "🔴"
                        
                        st.metric(
                            f"{sma_color} SMA(20)",
                            f"{sma:.2f}",
                            help=f"Status: {sma_status}"
                        )
                    
                    with col3:
                        macd = latest_indicators['macd']
                        if macd > 0:
                            macd_status = "Positive - Bullish"
                            macd_color = "🟢"
                        else:
                            macd_status = "Negative - Bearish"
                            macd_color = "🔴"
                        
                        st.metric(
                            f"{macd_color} MACD",
                            f"{macd:.4f}",
                            help=f"Status: {macd_status}"
                        )
                    
                    # Show price chart
                    st.subheader("📈 กราฟราคา")
                    st.line_chart(result.data['close'])
                    
                    # Show technical analysis summary
                    st.subheader("🎯 สรุปการวิเคราะห์")
                    
                    # RSI Analysis
                    if rsi > 70:
                        rsi_analysis = "RSI อยู่ในระดับ Overbought (>70) - อาจเป็นสัญญาณขาย"
                    elif rsi < 30:
                        rsi_analysis = "RSI อยู่ในระดับ Oversold (<30) - อาจเป็นสัญญาณซื้อ"
                    else:
                        rsi_analysis = "RSI อยู่ในระดับปกติ (30-70) - ไม่มีสัญญาณชัดเจน"
                    
                    # Price vs SMA Analysis
                    if price > sma:
                        sma_analysis = f"ราคา ({price:.2f}) อยู่เหนือ SMA(20) ({sma:.2f}) - สัญญาณ Bullish"
                    else:
                        sma_analysis = f"ราคา ({price:.2f}) อยู่ใต้ SMA(20) ({sma:.2f}) - สัญญาณ Bearish"
                    
                    # MACD Analysis
                    if macd > 0:
                        macd_analysis = "MACD เป็นบวก - สัญญาณ Bullish"
                    else:
                        macd_analysis = "MACD เป็นลบ - สัญญาณ Bearish"
                    
                    # Display analysis
                    st.info(f"**RSI Analysis:** {rsi_analysis}")
                    st.info(f"**Price vs SMA:** {sma_analysis}")
                    st.info(f"**MACD Analysis:** {macd_analysis}")
                    
                    # Overall recommendation
                    st.subheader("💡 คำแนะนำ")
                    
                    bullish_signals = 0
                    bearish_signals = 0
                    
                    if rsi < 30:
                        bullish_signals += 1
                    elif rsi > 70:
                        bearish_signals += 1
                    
                    if price > sma:
                        bullish_signals += 1
                    else:
                        bearish_signals += 1
                    
                    if macd > 0:
                        bullish_signals += 1
                    else:
                        bearish_signals += 1
                    
                    if bullish_signals > bearish_signals:
                        st.success("🟢 **สัญญาณโดยรวม: Bullish** - มีแนวโน้มขาขึ้น")
                    elif bearish_signals > bullish_signals:
                        st.error("🔴 **สัญญาณโดยรวม: Bearish** - มีแนวโน้มขาลง")
                    else:
                        st.warning("🟡 **สัญญาณโดยรวม: Neutral** - ไม่มีทิศทางชัดเจน")
                    
                else:
                    st.error(f"❌ คำนวณ indicators ล้มเหลว: {indicators_result.error}")
        else:
            st.error(f"❌ ไม่สามารถดึงข้อมูลได้: {result.error}")

else:
    # Show instructions
    st.info("👆 เลือกสัญลักษณ์หุ้นและกดปุ่ม 'วิเคราะห์' เพื่อเริ่มต้น")
    
    # Show examples
    st.subheader("📝 ตัวอย่างสัญลักษณ์")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**หุ้นไทย:**")
        st.markdown("- PTT.BK (ปตท.)")
        st.markdown("- KBANK.BK (ธนาคารกสิกรไทย)")
        st.markdown("- SCB.BK (ธนาคารไทยพาณิชย์)")
        st.markdown("- CPALL.BK (ซีพี ออลล์)")
        st.markdown("- ADVANC.BK (เอไอเอส)")
        st.markdown("- TRUE.BK (ทรู คอร์ปอเรชั่น)")
    
    with col2:
        st.markdown("**หุ้นอเมริกัน:**")
        st.markdown("- AAPL (Apple Inc.)")
        st.markdown("- MSFT (Microsoft Corporation)")
        st.markdown("- TSLA (Tesla Inc.)")
        st.markdown("- GOOGL (Alphabet Inc.)")
        st.markdown("- AMZN (Amazon.com Inc.)")
        st.markdown("- META (Meta Platforms Inc.)")
    
    # Show features
    st.subheader("✨ ฟีเจอร์")
    st.markdown("- 📊 การวิเคราะห์ technical indicators")
    st.markdown("- 📈 กราฟราคาแบบ real-time")
    st.markdown("- 🎯 สัญญาณซื้อ-ขาย")
    st.markdown("- 💡 คำแนะนำการลงทุน")
    st.markdown("- 🔄 ข้อมูล mock data เมื่อ API ไม่พร้อมใช้งาน")

# Footer
st.markdown("---")
st.markdown("**Finance Agent** - ระบบวิเคราะห์หุ้นอัจฉริยะ")
st.markdown("💡 *ใช้ mock data สำหรับการทดสอบ*")
