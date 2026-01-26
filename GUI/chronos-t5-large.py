import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import torch
import pandas as pd
import yfinance as yf
# นำเข้าตามโค้ดของคุณ
try:
    from chronos import ChronosPipeline
except ImportError:
    # เผื่อกรณี Library version ต่างกัน
    from chronos import Chronos2Pipeline as ChronosPipeline

# ==========================================
# ตั้งค่า Theme ของ GUI
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class StockForecastApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ตั้งค่าหน้าต่าง
        self.title("AI Stock Forecaster (Chronos Logic)")
        self.geometry("1200x800")

        self.pipeline = None
        self.is_loading = False

        # --- Layout Grid ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # 1. Sidebar (เมนูซ้าย)
        # ==========================================
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.logo = ctk.CTkLabel(self.sidebar, text="🤖 Chronos Forecast", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Input: Symbol
        ctk.CTkLabel(self.sidebar, text="Symbol (ชื่อหุ้น):", anchor="w").grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.entry_symbol = ctk.CTkEntry(self.sidebar)
        self.entry_symbol.insert(0, "SNDK")
        self.entry_symbol.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Input: Period
        ctk.CTkLabel(self.sidebar, text="Period (ย้อนหลัง):", anchor="w").grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.combo_period = ctk.CTkComboBox(self.sidebar, values=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])
        self.combo_period.set("2y")
        self.combo_period.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Input: Interval
        ctk.CTkLabel(self.sidebar, text="Interval (ความละเอียด):", anchor="w").grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.combo_interval = ctk.CTkComboBox(self.sidebar, values=["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"])
        self.combo_interval.set("1d")
        self.combo_interval.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Input: Steps
        ctk.CTkLabel(self.sidebar, text="Prediction Steps:", anchor="w").grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        self.entry_steps = ctk.CTkEntry(self.sidebar)
        self.entry_steps.insert(0, "30")
        self.entry_steps.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Button: Run
        self.btn_run = ctk.CTkButton(self.sidebar, text="RUN FORECAST", fg_color="#007acc", hover_color="#005da3", command=self.start_thread)
        self.btn_run.grid(row=9, column=0, padx=20, pady=20, sticky="ew")

        # Log Status
        self.status_box = ctk.CTkTextbox(self.sidebar, height=150)
        self.status_box.grid(row=10, column=0, padx=20, pady=10, sticky="nsew")
        self.status_box.insert("1.0", "Ready to start...\n")

        # Result Text
        self.result_lbl = ctk.CTkLabel(self.sidebar, text="--- Result ---", font=ctk.CTkFont(size=14, weight="bold"))
        self.result_lbl.grid(row=11, column=0, padx=20, pady=10)
        
        self.result_val = ctk.CTkLabel(self.sidebar, text="-", text_color="#00ccff")
        self.result_val.grid(row=12, column=0, padx=20, pady=0)

        # ==========================================
        # 2. Main Area (กราฟขวา)
        # ==========================================
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.canvas = None

    def log(self, msg):
        self.status_box.insert("end", f"{msg}\n")
        self.status_box.see("end")
        print(msg)

    # --- ฟังก์ชันช่วย (ตามโค้ดของคุณ) ---
    def get_pandas_freq(self, interval):
        mapping = {
            "1m": "1min", "2m": "2min", "5m": "5min",
            "15m": "15min", "30m": "30min", "60m": "60min", "90m": "90min",
            "1h": "1h", 
            "1d": "D", "5d": "5D", 
            "1wk": "W", "1mo": "ME"
        }
        return mapping.get(interval, "D")

    # --- ส่วนทำงาน Thread แยก (เพื่อไม่ให้ GUI ค้าง) ---
    def start_thread(self):
        self.btn_run.configure(state="disabled")
        threading.Thread(target=self.run_process).start()

    def load_model_if_needed(self):
        if self.pipeline is None:
            self.log("⏳ Loading Model... (First time takes 30s+)")
            try:
                # โค้ดส่วนโหลดโมเดลของคุณ
                try:
                    self.pipeline = ChronosPipeline.from_pretrained(
                        "amazon/chronos-t5-large",
                        device_map="cuda", 
                        torch_dtype=torch.bfloat16,
                    )
                    self.log("✅ Loaded Large Model (CUDA)")
                except Exception as e:
                    self.log(f"⚠️ GPU Issue: {e}")
                    self.log("🔄 Switching to CPU/Small model...")
                    self.pipeline = ChronosPipeline.from_pretrained(
                        "amazon/chronos-t5-small",
                        device_map="cpu", 
                        torch_dtype=torch.float32,
                    )
                    self.log("✅ Loaded Small Model (CPU)")
            except Exception as e:
                self.log(f"❌ Critical Error loading model: {e}")
                return False
        return True

    def run_process(self):
        try:
            if not self.load_model_if_needed():
                self.btn_run.configure(state="normal")
                return

            # รับค่าจาก GUI
            symbol = self.entry_symbol.get()
            period = self.combo_period.get()
            interval = self.combo_interval.get()
            steps = int(self.entry_steps.get())

            # ==========================================
            # 3. ดึงและเตรียมข้อมูล (Logic ของคุณ)
            # ==========================================
            self.log(f"🔄 Downloading {symbol} ({period}, {interval})...")
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)

            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(symbol, axis=1, level=1)
                except KeyError: pass

            target_col = "Close"
            if "Close" not in df.columns and "Adj Close" in df.columns:
                target_col = "Adj Close"
            
            if df.empty:
                self.log("❌ Data not found/Empty")
                self.btn_run.configure(state="normal")
                return

            # จัดการ Index/Timezone
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index.name = "timestamp"

            # Resample
            pandas_freq = self.get_pandas_freq(interval)
            self.log(f"ℹ️ Frequency: {pandas_freq}")
            df = df.resample(pandas_freq).last()
            df = df.ffill()
            df = df.dropna()
            df = df.reset_index()

            if len(df) < 20:
                self.log(f"❌ Data too short ({len(df)} rows)")
                self.btn_run.configure(state="normal")
                return

            # ==========================================
            # 4. พยากรณ์
            # ==========================================
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

            # ส่งข้อมูลไป Plot
            self.after(0, lambda: self.update_ui(context_df, forecast, steps, symbol, interval))

        except Exception as e:
            self.log(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_run.configure(state="normal")

    def update_ui(self, context_df, forecast, steps, symbol, interval):
        self.log("✅ Analysis Complete!")

        # 1. คำนวณตัวเลข
        actual_past = context_df.tail(min(len(context_df), 100))
        last_val = actual_past["target"].iloc[-1]
        pred_val = forecast["0.5"].iloc[-1]
        diff = ((pred_val - last_val) / last_val) * 100
        
        # อัปเดตตัวเลขหน้าจอ
        res_str = f"Last: {last_val:.2f}\nPred: {pred_val:.2f}\nDiff: {diff:+.2f}%"
        self.result_val.configure(text=res_str, text_color="#00ff00" if diff > 0 else "#ff5555")

        # 2. วาดกราฟ
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        # สร้าง Figure ใหม่
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        
        # ปรับสีให้เข้ากับ Dark Theme
        bg = '#2b2b2b'
        fg = 'white'
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg, rotation=45)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values(): spine.set_color(fg)

        # Plot Data
        ax.plot(actual_past["timestamp"], actual_past["target"], label="Actual", color="#ffff00", linewidth=1.5)
        ax.plot(forecast["timestamp"], forecast["0.5"], label="Forecast", color="#00ccff", linestyle="--", linewidth=2)
        ax.fill_between(
            forecast["timestamp"], 
            forecast["0.1"], 
            forecast["0.9"], 
            color="#00ccff", alpha=0.2
        )

        time_unit = "Days" if "d" in interval else ("Steps")
        ax.set_title(f"{symbol} Forecast ({interval}) - Next {steps} {time_unit}")
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.legend(facecolor=bg, labelcolor=fg)
        plt.tight_layout()

        # แสดงผลบน GUI
        self.canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

if __name__ == "__main__":
    app = StockForecastApp()
    app.mainloop()