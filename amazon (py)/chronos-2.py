import torch
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from chronos import Chronos2Pipeline

# ==========================================
# ⚙️ ส่วนตั้งค่า (CONFIGURATION) - ปรับแก้ตรงนี้ได้เลย
# ==========================================
SYMBOL = "SNDK"       # ชื่อหุ้น (เช่น AAPL, TSLA, BTC-USD, PTT.BK)
PERIOD = "2y"         # ย้อนหลัง: 1d, 5d, 1mo, 6mo, 1y, 2y, 5y, 10y, max
INTERVAL = "1d"       # ความละเอียด: 1m, 2m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
PREDICTION_STEPS = 30 # จำนวนจุดที่จะทำนายล่วงหน้า (ระวัง: ต้องสัมพันธ์กับ Interval)

# ==========================================
# 1. โหลดโมเดล
# ==========================================

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda", 
    torch_dtype=torch.bfloat16,
)

# ==========================================
# 2. ฟังก์ชันแปลง Interval เป็น Frequency
# ==========================================
def get_pandas_freq(interval):
    """แปลง yfinance interval เป็น pandas offset string"""
    mapping = {
        "1m": "1min", "2m": "2min", "5m": "5min",
        "15m": "15min", "30m": "30min", "60m": "60min", "90m": "90min",
        "1h": "1h", 
        "1d": "D", "5d": "5D", 
        "1wk": "W", "1mo": "ME" # Pandas ใหม่ใช้ 'ME' แทน 'M'
    }
    return mapping.get(interval, "D")

# ==========================================
# 3. ดึงและเตรียมข้อมูล (Robust Data Pipeline)
# ==========================================
print(f"🔄 กำลังดึงข้อมูล {SYMBOL} (Period: {PERIOD}, Interval: {INTERVAL})...")
df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)

# 3.1 แก้ไข MultiIndex (สำหรับ yfinance เวอร์ชันใหม่)
if isinstance(df.columns, pd.MultiIndex):
    try:
        df = df.xs(SYMBOL, axis=1, level=1)
    except KeyError:
        pass

# 3.2 หาคอลัมน์ราคา (Close/Adj Close)
target_col = "Close"
if "Close" not in df.columns and "Adj Close" in df.columns:
    target_col = "Adj Close"

if df.empty:
    raise ValueError("❌ ไม่พบข้อมูลหุ้น หรือช่วงเวลาที่ระบุไม่มีข้อมูล")

# 3.3 จัดการ Index และ Timezone
df.index = pd.to_datetime(df.index)
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# 3.4 ตั้งชื่อ Index และ Resample
df.index.name = "timestamp"
pandas_freq = get_pandas_freq(INTERVAL)
print(f"ℹ️ กำลังจัดระเบียบข้อมูลด้วยความถี่: {pandas_freq}")

# Resample และเติมค่าว่าง
df = df.resample(pandas_freq).last()
df = df.ffill() 
df = df.dropna()
df = df.reset_index()

# เช็คจำนวนข้อมูล
if len(df) < 20:
    raise ValueError(f"❌ ข้อมูลน้อยเกินไป ({len(df)} แถว) ลองเพิ่ม Period หรือลด Interval")

# ==========================================
# 4. เตรียม Context และ พยากรณ์
# ==========================================
context_df = pd.DataFrame({
    "id": [SYMBOL] * len(df),
    "timestamp": df["timestamp"],
    "target": df[target_col]
})

print(f"🔮 กำลังพยากรณ์ล่วงหน้า {PREDICTION_STEPS} จุดข้อมูล...")

# ใช้ method predict_df ซึ่งรองรับการจัดการ DataFrame โดยตรง
forecast = pipeline.predict_df(
    context_df,
    prediction_length=PREDICTION_STEPS,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target"
)

# ==========================================
# 5. Visualization (กราฟ)
# ==========================================
# เลือกดูข้อมูลย้อนหลังให้เหมาะสมกับกราฟ (เช่น 4 เท่าของช่วงทำนาย)
lookback = min(len(context_df), PREDICTION_STEPS * 4) 
actual_past = context_df.tail(lookback)

plt.figure(figsize=(12, 6))

# พล็อตราคาจริง
plt.plot(actual_past["timestamp"], actual_past["target"], label="Actual Price", color="black", linewidth=1.5)

# พล็อตพยากรณ์
plt.plot(forecast["timestamp"], forecast["0.5"], label="Forecast (Median)", color="#007acc", linestyle="--", linewidth=2)

# พล็อตช่วงความเชื่อมั่น
plt.fill_between(
    forecast["timestamp"], 
    forecast["0.1"], 
    forecast["0.9"], 
    color="#007acc", alpha=0.3, label="Confidence Interval (10%-90%)"
)

# ตั้งชื่อกราฟ
time_unit = "Days" if "d" in INTERVAL else ("Weeks" if "wk" in INTERVAL else "Steps")
plt.title(f"Forecast for {SYMBOL}\n(Interval: {INTERVAL} | Next {PREDICTION_STEPS} steps)")
plt.xlabel("Date/Time")
plt.ylabel(f"Price ({target_col})")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# สรุปผล
last_val = actual_past["target"].iloc[-1]
pred_val = forecast["0.5"].iloc[-1]
diff = ((pred_val - last_val) / last_val) * 100
print(f"\n--- 📊 Analysis Result ---")
print(f"ราคาล่าสุด: {last_val:.2f}")
print(f"ราคาคาดการณ์ ({PREDICTION_STEPS} steps): {pred_val:.2f}")
print(f"แนวโน้ม: {'บวก 📈' if diff > 0 else 'ลบ 📉'} ({diff:.2f}%)")