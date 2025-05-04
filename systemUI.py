"""
    Author:  Eusebiu Tamas
    Purpose: To develop a front-end graphcal user interface for backtest purposes initially
             and as future work to use it for manual and automated tradings based on the best model predictions
"""
import pickle
from typing import Union, Callable
import matplotlib
import numpy as np
import pandas as pd
from keras.src.metrics import MeanSquaredError
from keras.src.optimizers import Adam, RMSprop
from matplotlib import animation
from matplotlib.dates import DateFormatter, HourLocator, MinuteLocator
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import cnn_lstm_model
from cnn_lstm_model import CNN_LSTM
from genetic_optimizer import GeneticAlgorithm

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
import customtkinter as ctk
from tensorflow.keras.models import load_model
import tensorflow as tf

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")
style.use('seaborn-v0_8')

IST_huge_font = ("Fira Code", 14)
IST_large_font = ("Fira Code", 12)
IST_medium_font = ("Fira Code", 10)
IST_tiny_font = ("Fira Code", 8)
IST_custom_font = ("Fira Code", 15, "bold")

fig = Figure(facecolor='#eeeeee')
a = fig.add_subplot(111, facecolor='#eeeeee')
plot_type = 'candlestick'

current_index = 0
exchange = "AAPL"
programName = "aapl"

resampleSize = "1Min"
DataPace = "1Min"
candleWidth = 0.0005
DatCounter = 9000
zoom_factor = 1
animation_running = True
simulation_active = False

features = ['minute_1_return', 'minute_5_return', 'minute_10_return', 'minute_15_return',
            'rup', 'rdown', 'MFI', 'CCI', 'EMA_5', 'EMA_20', 'MACD_HIST', 'STOCHRSI',
            'WILLR', 'slowk', 'minute_5_forward', 'minute_10_forward']

def prepare_features(df):
    X = df[features].values
    y = df['minute_1_forward'].values
    return X, y

def changeTimeFrame(tf):
    global DataPace, DatCounter
    if tf == "7d" and resampleSize == "1Min":
        popupmsg("Not enough data to display!")
    else:
        DataPace = tf
        DatCounter = 9000
        animate(0)

def changeSampleSize(size, width):
    global resampleSize, DatCounter, candleWidth
    if DataPace == "7d" and size == "1Min":
        popupmsg("Not enough data to display!")
    elif DataPace == "tick":
        popupmsg("Not enough data to display!")
    else:
        resampleSize = size
        DatCounter = 9000
        candleWidth = width
        animate(0)

def changeExchange(toWhat):
    global exchange, programName
    exchange = toWhat
    programName = toWhat.lower()
    animate(0)

def popupmsg(msg):
    popup = ctk.CTk()
    popup.wm_title("!")
    label = ctk.CTkLabel(popup, text=msg, font=IST_medium_font)
    label.pack(side="top", fill="x", pady=10)
    B1 = ctk.CTkButton(popup, text="OK", command=popup.destroy)
    B1.pack()
    popup.mainloop()

def resample_data(df, interval):
    df_resampled = (df.resample(interval)
                    .agg({'first': 'first', 'high': 'max', 'low': 'min', 'last': 'last'})
                    .dropna())
    return df_resampled

def zoom_in():
    global zoom_factor
    zoom_factor *= 0.8
    animate(0)

def zoom_out():
    global zoom_factor
    zoom_factor *= 1.2
    animate(0)

def pause_animation():
    global animation_running
    if animation_running:
        ani.event_source.stop()
        animation_running = False
        print("Trading simulation paused.")

def start_animation():
    global animation_running
    if not animation_running:
        ani.event_source.start()
        animation_running = True
        print("Trading simulation resumed.")

def toggle_plot_type():
    global plot_type
    plot_type = 'line' if plot_type == 'candlestick' else 'candlestick'
    animate(0)

def animate(i):
    global current_index, exchange, zoom_factor, df_resampled, simulation_active
    pickle_path = '2017_model_data.pkl'
    with open(pickle_path, 'rb') as file:
        df = pickle.load(file)
    try:
        df = df.xs(exchange, level='ticker')
    except KeyError:
        popupmsg(f"Stock {exchange} not found in dataset!")
        return
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # This is using the 1st of June 2017 as test data
    try:
        df = df.loc['2017-01-01 10:01':]
    except KeyError:
        popupmsg(f"Data for {exchange} does not contain the specified date range!")
        return
    df_resampled = resample_data(df, resampleSize)
    df_current = df_resampled.iloc[:current_index + 1]
    if len(df_current) > 0:
        last_time = df_current.index[-1]
        start_time = last_time - pd.Timedelta(hours=1 * zoom_factor)
        df_current = df_current.loc[df_current.index >= start_time]
    a.clear()
    a.set_facecolor('#e0e0e0')
    a.tick_params(axis='x', colors='black')
    a.tick_params(axis='y', colors='black')
    a.spines['bottom'].set_color('black')
    a.spines['left'].set_color('black')
    a.xaxis.label.set_color('black')
    a.yaxis.label.set_color('black')
    a.title.set_color('black')

    # This is plotting the time series data in a candlestick format
    if plot_type == 'line':
        a.plot(df_current.index, df_current['last'], label=f'{exchange} Last Price', color='deepskyblue')
        a.set_title(f'IST Application - {exchange} Stock (Line)\nTime: {df_current.index[-1]}')
    elif plot_type == 'candlestick':
        df_ohlc = df_current[['first', 'high', 'low', 'last']].copy()
        df_ohlc.reset_index(inplace=True)
        df_ohlc['date_time'] = df_ohlc['date_time'].map(matplotlib.dates.date2num)
        candlestick_ohlc(a, df_ohlc.values, width=candleWidth, colorup='limegreen', colordown='tomato', alpha=0.8)
        a.set_title(f'IST Application - {exchange} Stock (Candles)\nTime: {df_current.index[-1]}')

        # This is loading the processed data.
        if simulation_active:
            try:
                with open("processed_2017_data.pkl", "rb") as f:
                    X, y = pickle.load(f)

                # Those are for debugginh purposes
                #print("Shape X:", X.shape)
                #print("Shape y:", y.shape)

                # This is loading the model with custom pre-defined metrics
                @tf.keras.utils.register_keras_serializable()
                def mse(y_true, y_pred):
                    return MeanSquaredError()(y_true, y_pred)

                model = load_model('hybrid_model.keras', custom_objects={'mse': mse})

                # This is using the last predictions
                y_pred_scaled = model.predict(X[-len(df_current):])
                y_pred = y_pred_scaled.flatten()

                # Those print statements are for debugging purposes
                # print("Predictii în unități originale:\n", y_pred[:5])
                # print("\n=== PREDICTII FINALE ===")
                # print("Ultimele 5 valori reale:", y[-5:])
                # print("Ultimele 5 predicții:", y_pred[-5:])

                # This will generate signals for building the strategy
                signals = []

                for i in range(1, len(y_pred)):
                    current_candle_time = df_current.index[i]
                    current_price = df_current['last'].iloc[i]
                    predicted_price = y_pred[i]

                    # This compares the previous predictions to determine if it is a Buy or a Sell signal
                    if y_pred[i] > y_pred[i - 1]:
                        signals.append(('BUY', current_candle_time, current_price, f"{predicted_price:.2f}"))
                    elif y_pred[i] < y_pred[i - 1]:
                        signals.append(('SELL', current_candle_time, current_price, f"{predicted_price:.2f}"))
                    else:
                        signals.append(('HOLD', current_candle_time, current_price, f"{predicted_price:.2f}"))

                # This simulate a simple vectorized trend following strategy with an initial ballance of $100,000
                initial_balance = 100000
                balance = initial_balance
                positions = 0

                current_prices = df_current['last'].tolist()

                for i in range(1, len(signals)):
                    signal = signals[i - 1][0]
                    price = current_prices[i]

                    if signal == 'BUY' and balance > 0:
                        positions += balance / price
                        balance = 0
                    elif signal == 'SELL' and positions > 0:
                        balance += positions * price
                        positions = 0

                # This calculate the final amount
                final_balance = balance + positions * current_prices[-1]
                profit = final_balance - initial_balance

                #print(f"Final Balance: £{final_balance:.2f}")
                #print(f"Profit: £{profit:.2f}")

                # Now this will help visualise the signals on the graph
                buy_signals = [s for s in signals if s[0] == 'BUY']
                sell_signals = [s for s in signals if s[0] == 'SELL']

                # This will add the buy signals on the graph
                if buy_signals:
                    a.scatter(
                        [s[1] for s in buy_signals[-60:]],
                        [s[2] for s in buy_signals[-60:]],
                        marker='^', color='lime', s=100,
                        edgecolors='black', linewidths=0.5,
                        label=f'BUY ({len(buy_signals)})',
                        zorder=5
                    )

                # This is adding the sell signals on the graph
                if sell_signals:
                    a.scatter(
                        [s[1] for s in sell_signals[-60:]],
                        [s[2] for s in sell_signals[-60:]],
                        marker='v', color='red', s=100,
                        edgecolors='black', linewidths=0.5,
                        label=f'SELL ({len(sell_signals)})',
                        zorder=5
                    )

                # This will append the history with the signals found on the history text box
                stats_text = f""":
                 BUY Signals: {len(buy_signals)}
                 SELL Signals: {len(sell_signals)}
                 Final Balance: £{final_balance:.2f}
                 Profit: £{profit:.2f}"""

                current_text = app.history_label.cget("text").split("-")[0].strip()
                history_update = f"{current_text}-\n{stats_text}"
                app.history_label.configure(text=history_update.strip())

                # Those print statements were added for debugging
                #print(f"\n{'=' * 40}\nDebug Info:")
                #print(f"Last 3 predictions: {y_pred[-3:]}")
                #print(f"last 3 signals: {signals[-3:] if signals else 'N/A'}")
                #print(f"Signals Distribution: BUY {len(buy_signals)} | SELL {len(sell_signals)}")
                #print(f"Exemple trade results:", trade_results[-2:] if trade_results else 'N/A')

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                current_text = app.history_label.cget("text").split("-")[0].strip()
                app.history_label.configure(text=f"{current_text}\n-\n{error_msg}")

    a.set_xlabel('Time Intervals')
    a.set_ylabel('Price')
    a.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    a.xaxis.set_major_locator(MinuteLocator(interval=5))
    if len(df_current) > 0:
        a.set_xlim(left=start_time, right=last_time)
    fig.autofmt_xdate()

    current_index += 1
    if current_index >= len(df_resampled):
        current_index = 0

def map_individual_to_dict(individual):
    gene_names = ['cnn_filters', 'lstm_units', 'dropout_rate', 'kernel_size', 'activation']
    return {gene_names[i]: individual[i] for i in range(len(gene_names))}

class ISTapplication(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iconbitmap(default='icons/stock-market-fin.ico')
        self.title("IST Application")
        self.geometry("800x600")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=1, sticky="sew")
        self.add_plot_to_main_frame()
        self.sidebar = ctk.CTkFrame(self, width=350, corner_radius=2)
        self.sidebar.grid(row=1, column=0, sticky="nsw")
        self.sidebar.grid_propagate(False)
        self.add_sidebar_buttons()
        self.topbar = ctk.CTkFrame(self, width=5, corner_radius=2)
        self.topbar.grid(row=1, column=1, sticky="new")
        self.add_topbar_buttons()
        self.step_size: Union[int, float] = 1
        self.command: Callable[[], None]
        self.configure(fg_color=("#DCDCDC", "#DCDCDC"))
        self.grid_columnconfigure((0, 2), weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.label_amount = ctk.CTkLabel(self, text="Insert Amount:", font=IST_huge_font, fg_color='#DCDCDC',
                                         bg_color="transparent")
        self.label_amount.grid(row=1, column=0, padx=(35, 0), pady=200, sticky="nw")
        self.entry_amount = ctk.CTkEntry(self, width=100, corner_radius=5, validate="key")
        self.entry_amount.configure(validatecommand=(self.register(self.validate_amount), "%P"))
        self.entry_amount.grid(row=1, column=0, padx=(30, 0), pady=230, sticky="nw")
        self.entry_amount.insert(0, "$")
        self.entry_amount.bind("<KeyRelease>", self.enforce_dollar_sign)
        self.subtract_button_stop_loss = ctk.CTkButton(self, text="-", width=26, height=26,
                                                        command=self.subtract_button_stop_loss_callback,
                                                        fg_color=("#A9A9A9", "#A9A9A9"),
                                                        text_color="#000000", hover_color="#808080")
        self.subtract_button_stop_loss.grid(row=1, column=0, pady=260, sticky="sw")
        self.label_stop_loss = ctk.CTkLabel(self, text="Add Stop Loss %:", font=IST_huge_font, fg_color='#DCDCDC',
                                            bg_color="transparent")
        self.label_stop_loss.grid(row=1, column=0, padx=(32, 0), pady=280, sticky="sw")
        self.entry_stop_loss = ctk.CTkEntry(self, width=100, corner_radius=5)
        self.entry_stop_loss.grid(row=1, column=0, padx=(30, 0), pady=260, sticky="sw")
        self.add_button_stop_loss = ctk.CTkButton(self, text="+", width=26, height=26,
                                                  command=self.add_button_stop_loss_callback,
                                                  fg_color=("#A9A9A9", "#A9A9A9"),
                                                  text_color="#000000", hover_color="#808080")
        self.add_button_stop_loss.grid(row=1, column=0, padx=(135, 0), pady=260, sticky="sw")
        self.entry_stop_loss.insert(0, "0.0")
        self.subtract_button_take_profit = ctk.CTkButton(self, text="-", width=26, height=26,
                                                         command=self.subtract_button_take_profit_callback,
                                                         fg_color=("#A9A9A9", "#A9A9A9"),
                                                         text_color="#000000", hover_color="#808080")
        self.subtract_button_take_profit.grid(row=1, column=0, padx=(0, 130), pady=210, sticky="sw")
        self.label_take_profit = ctk.CTkLabel(self, text="Add Take Profit %:", font=IST_huge_font, fg_color='#DCDCDC',
                                              bg_color="transparent")
        self.label_take_profit.grid(row=1, column=0, padx=(32, 0), pady=230, sticky="sw")
        self.entry_take_profit = ctk.CTkEntry(self, width=100, corner_radius=5)
        self.entry_take_profit.grid(row=1, column=0, padx=(30, 0), pady=210, sticky="sw")
        self.add_button_take_profit = ctk.CTkButton(self, text="+", width=26, height=26,
                                                    command=self.add_button_take_profit_callback,
                                                    fg_color=("#A9A9A9", "#A9A9A9"),
                                                    text_color="#000000", hover_color="#808080")
        self.add_button_take_profit.grid(row=1, column=0, padx=(135, 0), pady=210, sticky="sw")
        self.entry_take_profit.insert(0, "0.0")
        self.trade_button = ctk.CTkButton(self, text="Trade", font=IST_custom_font, width=130, height=40,
                                          command=self.trade_button_callback, corner_radius=20, fg_color="#3CB371",
                                          hover_color="#2E8B57")
        self.trade_button.grid(row=1, column=0, padx=(18, 10), pady=140, sticky="sw")
        self.predict_button = ctk.CTkButton(self, text="Predict", font=IST_custom_font, width=130, height=40,
                                          command=self.predict_button, corner_radius=20, fg_color="#CD5C5C",
                                            hover_color="#B22222")
        self.predict_button.grid(row=1, column=0, padx=(18, 10), pady=150, sticky="nw")
        self.history_log = []
        self.history_label = ctk.CTkLabel(self, text="History:\n", font=IST_medium_font, anchor="w",
                                          justify="left")
        self.history_label.grid(row=1, column=0, padx=(5, 0), pady=10, sticky="sw")

    def validate_amount(self, new_text):
        return new_text.startswith("$")

    def enforce_dollar_sign(self, event):
        current_text = self.entry_amount.get()
        if not current_text.startswith("$"):
            self.entry_amount.delete(0, "end")
            self.entry_amount.insert(0, "$" + current_text)

    def add_button_stop_loss_callback(self):
        self.update_entry(self.entry_stop_loss, self.step_size)

    def subtract_button_stop_loss_callback(self):
        self.update_entry(self.entry_stop_loss, -self.step_size)

    def add_button_take_profit_callback(self):
        self.update_entry(self.entry_take_profit, self.step_size)

    def subtract_button_take_profit_callback(self):
        self.update_entry(self.entry_take_profit, -self.step_size)

    def update_entry(self, entry, step):
        try:
            current_value = float(entry.get())
            new_value = current_value + step
            if new_value < 0.0:
                new_value = 0.0
            elif new_value > 10.0:
                new_value = 10.0
            entry.delete(0, "end")
            entry.insert(0, str(new_value))
        except ValueError:
            return

    def get_stop_loss(self) -> Union[float, None]:
        try:
            return float(self.entry_stop_loss.get())
        except ValueError:
            return None

    def set_stop_loss(self, value: float):
        if value < 0.0:
            value = 0.0
        elif value > 10.0:
            value = 10.0
        self.entry_stop_loss.delete(0, "end")
        self.entry_stop_loss.insert(0, str(float(value)))

    def get_take_profit(self) -> Union[float, None]:
        try:
            return float(self.entry_take_profit.get())
        except ValueError:
            return None

    def set_take_profit(self, value: float):
        if value < 0.0:
            value = 0.0
        elif value > 10.0:
            value = 10.0
        self.entry_take_profit.delete(0, "end")
        self.entry_take_profit.insert(0, str(float(value)))

    def get_amount(self) -> Union[float, None]:
        try:
            return float(self.entry_amount.get().replace("$", ""))
        except ValueError:
            return None

    def set_amount(self, value: float):
        self.entry_amount.delete(0, "end")
        self.entry_amount.insert(0, f"${value}")

    def trade_button_callback(self):
        amount = self.get_amount()
        stop_loss = self.get_stop_loss()
        take_profit = self.get_take_profit()
        if amount is None or stop_loss is None or take_profit is None:
            print("Please enter valid values for Amount, Stop Loss, and Take Profit.")
            return
        print(f"You are trading with: Amount=${amount}, Stop Loss={stop_loss}%, Take Profit={take_profit}%")

    def predict_button(self):
        global simulation_active
        simulation_active = True

    def add_sidebar_buttons(self):
        search_frame = ctk.CTkFrame(self.sidebar)
        search_frame.pack(side="top", fill="x", padx=10, pady=5)
        search_label = ctk.CTkLabel(search_frame, text="Search Stock:", font=IST_huge_font)
        search_label.pack(side="top")
        search_entry = ctk.CTkEntry(search_frame, width=130, font=IST_medium_font)
        search_entry.pack(side="top", padx=5)
        def on_search():
            ticker = search_entry.get().strip().upper()
            if ticker:
                changeExchange(ticker)
                self.configure(text=f"IST Application - {ticker} Stock")
        search_button = ctk.CTkButton(search_frame, text="Search", command=on_search, corner_radius=8,
                                       fg_color="#A9A9A9", text_color="#000000", hover_color="#808080")
        search_button.pack(side="top", padx=1, pady=1)
        toggle_button = ctk.CTkButton(self.sidebar, text="Change Plot Style", command=toggle_plot_type,
                                      corner_radius=8, fg_color="#A9A9A9", text_color="#000000", hover_color="#808080")
        toggle_button.pack(side="top", padx=1, pady=1)

    def add_topbar_buttons(self):
        intervals = [("1Min", 0.0005), ("5Min", 0.003), ("10Min", 0.005),
                     ("15Min", 0.008), ("30Min", 0.016), ("1h", 0.032)]
        for text, width in intervals:
            button = ctk.CTkButton(self.topbar, text=text, command=lambda t=text, w=width: changeSampleSize(t, w),
                                   corner_radius=8, width=50, fg_color="#DCDCDC",
                                   text_color="#000000", hover_color="#C0C0C0")
            button.pack(side="left", expand=True)
        zoom_in_button = ctk.CTkButton(self.topbar, width=3, text="Zoom In", command=zoom_in, corner_radius=8,
                                       fg_color="#DCDCDC", text_color="#000000", hover_color="#C0C0C0")
        zoom_in_button.pack(side="left", expand=True)
        zoom_out_button = ctk.CTkButton(self.topbar, width=3, text="Zoom Out", command=zoom_out, corner_radius=8,
                                        fg_color="#DCDCDC", text_color="#000000", hover_color="#C0C0C0")
        zoom_out_button.pack(side="left", expand=True)
        pause_button = ctk.CTkButton(self.topbar, width=3, text="Pause", command=pause_animation, corner_radius=8,
                                     fg_color="#DCDCDC", text_color="#000000", hover_color="#C0C0C0")
        pause_button.pack(side="left", expand=True)
        start_button = ctk.CTkButton(self.topbar, width=3, text="Resume", command=start_animation, corner_radius=8,
                                     fg_color="#DCDCDC", text_color="#000000", hover_color="#C0C0C0")
        start_button.pack(side="left", expand=True)

    def add_plot_to_main_frame(self):
        canvas = FigureCanvasTkAgg(fig, self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self.main_frame)
        toolbar.update()
        canvas._tkcanvas.pack(side="top", fill="both", expand=True)

app = ISTapplication()
ani = animation.FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
app.mainloop()

