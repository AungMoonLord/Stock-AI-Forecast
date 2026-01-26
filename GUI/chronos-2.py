import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import torch
import pandas as pd
import yfinance as yf
from chronos import Chronos2Pipeline

# ตั้งค่า Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class StockForecastApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ตั้งค่าหน้าต่างโปรแกรม
        self.title("AI Stock Forecaster (Chronos-2)")
        self.geometry("1100x750")

        # ตัวแปรเก็บ Model
        self.pipeline = None
        
        # --- Layout แบ่งซ้ายขวา ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar (เมนูซ้าย)
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Stock AI Forecast", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Input: Symbol
        self.lbl_symbol = ctk.CTkLabel(self.sidebar_frame, text="Symbol (ชื่อหุ้น):", anchor="w")
        self.lbl_symbol.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.entry_symbol = ctk.CTkEntry(self.sidebar_frame, placeholder_text="e.g. AAPL")
        self.entry_symbol.insert(0, "SNDK")
        self.entry_symbol.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Input: Period (อัปเดตตามคำขอ)
        self.lbl_period = ctk.CTkLabel(self.sidebar_frame, text="Period (ย้อนหลัง):", anchor="w")
        self.lbl_period.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        
        # รายการ Period ตามที่คุณต้องการ
        period_options = ["1d", "5d", "1mo", "6mo", "1y", "2y", "5y", "10y", "max"]
        self.combo_period = ctk.CTkComboBox(self.sidebar_frame, values=period_options)
        self.combo_period.set("6mo") # ค่าเริ่มต้น
        self.combo_period.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Input: Interval (อัปเดตตามคำขอ)
        self.lbl_interval = ctk.CTkLabel(self.sidebar_frame, text="Interval (ความละเอียด):", anchor="w")
        self.lbl_interval.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        
        # รายการ Interval ตามที่คุณต้องการ
        interval_options = ["1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
        self.combo_interval = ctk.CTkComboBox(self.sidebar_frame, values=interval_options)
        self.combo_interval.set("1h") # ค่าเริ่มต้น
        self.combo_interval.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Input: Steps
        self.lbl_steps = ctk.CTkLabel(self.sidebar_frame, text="Prediction Steps:", anchor="w")
        self.lbl_steps.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        self.entry_steps = ctk.CTkEntry(self.sidebar_frame)
        self.entry_steps.insert(0, "10")
        self.entry_steps.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="ew")

        # ปุ่ม Run
        self.btn_run = ctk.CTkButton(self.sidebar_frame, text="Run Forecast", command=self.start_forecast_thread)
        self.btn_run.grid(row=9, column=0, padx=20, pady=20, sticky="ew")

        # สถานะการทำงาน
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Ready", text_color="gray", wraplength=200)
        self.status_label.grid(row=10, column=0, padx=20, pady=10)

        # ผลลัพธ์ตัวเลข
        self.result_text = ctk.CTkTextbox(self.sidebar_frame, height=150)
        self.result_text.grid(row=11, column=0, padx=20, pady=10, sticky="nsew")

        # 2. Main Area (พื้นที่กราฟขวา)
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        
        self.canvas = None

    def log(self, message):
        """ฟังก์ชันสำหรับแสดงข้อความสถานะ"""
        self.status_label.configure(text=message)
        print(message)

    def load_model_if_needed(self):
        if self.pipeline is None:
            self.log("⏳ Loading AI Model... (Please wait)")
            # ใช้ model tiny/small เพื่อความรวดเร็วในการทดสอบ (เปลี่ยน path ได้ตามต้องการ)
            self.pipeline = Chronos2Pipeline.from_pretrained(
                "amazon/chronos-2", 
                device_map="auto", 
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            self.log("✅ Model Loaded")

    def start_forecast_thread(self):
        # รันใน Thread แยก เพื่อไม่ให้ GUI ค้าง
        self.btn_run.configure(state="disabled")
        threading.Thread(target=self.run_process).start()

    def run_process(self):
        try:
            self.load_model_if_needed()
            
            # รับค่าจาก GUI
            symbol = self.entry_symbol.get()
            period = self.combo_period.get()
            interval = self.combo_interval.get()
            steps = int(self.entry_steps.get())

            self.log(f"🔄 Downloading {symbol} ({period}/{interval})...")
            
            # --- ดึงข้อมูล ---
            try:
                df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
            except Exception as e:
                self.log(f"❌ Download Error: {e}")
                self.btn_run.configure(state="normal")
                return

            # แก้ไข MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(symbol, axis=1, level=1)
                except KeyError: pass
            
            # หาคอลัมน์ราคา
            target_col = "Close"
            if "Close" not in df.columns and "Adj Close" in df.columns:
                target_col = "Adj Close"
            
            if df.empty:
                # แจ้งเตือนกรณีจับคู่ Period/Interval ไม่ได้ (เช่น 1m ย้อนหลัง 1y เป็นไปไม่ได้)
                err_msg = "❌ No Data Found.\nNote: High frequency data (1m, 2m) is only available for last 7 days."
                self.log(err_msg)
                self.btn_run.configure(state="normal")
                return

            # Prepare Data
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df.index.name = "timestamp"
            
            # --- Mapping ให้ครบทุกหน่วยตามที่ขอ ---
            mapping = {
                "1m": "1min", "2m": "2min", "5m": "5min",
                "15m": "15min", "30m": "30min",
                "1h": "1h", 
                "1d": "D", 
                "1wk": "W", 
                "1mo": "ME"
            }
            # Fallback ถ้าหาไม่เจอให้ใช้ 'D'
            pandas_freq = mapping.get(interval, "D")
            
            self.log(f"ℹ️ Resampling data to {pandas_freq}...")
            
            # Resample และ Clean Data
            df = df.resample(pandas_freq).last().ffill().dropna().reset_index()

            if len(df) < 10:
                self.log(f"❌ Not enough data ({len(df)} rows).")
                self.btn_run.configure(state="normal")
                return

            context_df = pd.DataFrame({
                "id": [symbol] * len(df),
                "timestamp": df["timestamp"],
                "target": df[target_col]
            })

            self.log(f"🔮 Forecasting {steps} steps...")
            
            forecast = self.pipeline.predict_df(
                context_df,
                prediction_length=steps,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target"
            )

            # ส่งข้อมูลไปวาดกราฟที่ Main Thread
            self.after(0, lambda: self.update_plot(context_df, forecast, steps, target_col, symbol, interval))
            
        except Exception as e:
            self.log(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_run.configure(state="normal")

    def update_plot(self, context_df, forecast, steps, target_col, symbol, interval):
        self.log("✅ Analysis Complete")
        
        # 1. Update Text Result
        lookback = min(len(context_df), steps * 4)
        actual_past = context_df.tail(lookback)
        
        last_val = actual_past["target"].iloc[-1]
        pred_val = forecast["0.5"].iloc[-1]
        diff = ((pred_val - last_val) / last_val) * 100
        
        res_msg = f"Last Price: {last_val:.2f}\n"
        res_msg += f"Forecast: {pred_val:.2f}\n"
        res_msg += f"Trend: {diff:+.2f}% {'📈' if diff>0 else '📉'}"
        
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", res_msg)

        # 2. Update Graph
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        # ตั้งค่าสีให้เข้ากับ Dark Mode
        bg_color = '#2b2b2b'
        text_color = 'white'
        
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.tick_params(axis='x', colors=text_color, rotation=45)
        ax.tick_params(axis='y', colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)

        # Plot Data
        ax.plot(actual_past["timestamp"], actual_past["target"], label="Actual", color="#00ff00", linewidth=1.5)
        ax.plot(forecast["timestamp"], forecast["0.5"], label="Forecast", color="#00ccff", linestyle="--", linewidth=2)
        ax.fill_between(
            forecast["timestamp"], 
            forecast["0.1"], 
            forecast["0.9"], 
            color="#00ccff", alpha=0.2
        )
        
        ax.set_title(f"Forecast: {symbol} (Interval: {interval})")
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.legend(facecolor=bg_color, labelcolor=text_color)
        plt.tight_layout()

        # แสดงกราฟ
        self.canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

if __name__ == "__main__":
    app = StockForecastApp()
    app.mainloop()
