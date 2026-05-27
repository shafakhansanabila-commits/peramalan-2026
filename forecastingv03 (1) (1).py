import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import io

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False


st.set_page_config(
    page_title="Dashboard Peramalan Colorful",
    page_icon="📈",
    layout="wide"
)

# Suntikan CSS - VIBRANT COLORFUL LIGHT STYLE
st.markdown("""
    <style>
    /* 1. Fondasi Font & Background Utama Cerah */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="st-"], .stMarkdown, p, span, label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Memaksa background utama menjadi gelap elegan agar teks pastel terlihat tajam */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
    }

    /* 2. Sidebar Cerah & Segar */
    [data-testid="stSidebar"] {
        background-color: #1e1e38 !important;
        border-right: 2px solid #2e2d56 !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stWidgetLabel p,
    [data-testid="stSidebar"] p {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f472b6 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-bottom: 2px solid #2e2d56;
        padding-bottom: 8px;
        margin-top: 20px !important;
        letter-spacing: 0.5px;
    }

    /* 3. Layout Uploader Colorful */
    [data-testid="stFileUploader"] {
        background-color: #252448 !important;
        border: 2px dashed #f472b6 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #ec4899 0%, #db2777 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        box-shadow: 0 4px 6px -1px rgba(236, 72, 153, 0.2) !important;
    }
    
    [data-testid="stFileUploader"] button * {
        font-size: 0px !important;
        color: transparent !important;
        display: none !important;
    }
    
    [data-testid="stFileUploader"] button::after {
        content: "Pilih File Kamu" !important;
        color: #ffffff !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        display: block !important;
    }

    [data-testid="stFileUploader"] text {
        fill: #e2e8f0 !important;
    }
    [data-testid="stFileUploader"] div {
        color: #e2e8f0 !important;
        font-weight: 500;
    }

    /* 4. Area Konten Utama */
    .stApp h1, .stApp h2, .stApp h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    
    .stApp p, .stMarkdown p {
        color: #cbd5e1 !important;
        font-weight: 500;
    }

    /* 5. Metric Cards Penuh Warna Pastel */
    [data-testid="stMetricValue"] {
        background: #1e1e38 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        border-radius: 12px !important;
        padding: 15px 20px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
        border-left: 5px solid #ec4899 !important;
        border-top: 1px solid #2e2d56 !important;
        border-right: 1px solid #2e2d56 !important;
        border-bottom: 1px solid #2e2d56 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #f472b6 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        margin-left: 5px !important;
    }

    /* 6. Tombol Utama Pro (Neon/Bright Coral Style) */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%) !important;
        color: #FFFFFF !important;            
        font-weight: 700 !important;          
        font-size: 1rem;
        padding: 0.65rem 1rem;
        border: none !important;
        box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(236, 72, 153, 0.6);
        background: linear-gradient(135deg, #f43f5e 0%, #ec4899 100%) !important;
    }

    /* 7. Desain Tabel Data Grid */
    .stDataFrame, div[data-testid="stDataFrame"] {
        background-color: #1e1e38 !important;
        border: 2px solid #2e2d56 !important;
        border-radius: 12px;
    }

    /* Customisasi Tabs Streamlit agar Colorful */
    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        color: #94a3b8 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f472b6 !important;
        border-bottom-color: #f472b6 !important;
    }

    hr {
        margin: 1.5rem 0 !important;
        border-color: #2e2d56 !important;
    }
    </style>
""", unsafe_allow_html=True)
# --- FUNGSI PROSES DAN PERHITUNGAN DASAR ---

def clean_numeric_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.dropna().astype(float)


def safe_mape(actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100


def calculate_error_table(periods, actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)

    error = actual - forecast
    abs_error = np.abs(error)
    squared_error = error ** 2
    ape = np.where(actual != 0, np.abs(error / actual) * 100, np.nan)

    result_df = pd.DataFrame({
        "Periode": periods,
        "Aktual": actual,
        "Forecast": forecast,
        "Error": error,
        "Absolute Error": abs_error,
        "Squared Error": squared_error,
        "APE (%)": ape
    })

    metrics = {
        "MAD": np.mean(abs_error),
        "MSE": np.mean(squared_error),
        "MAPE": safe_mape(actual, forecast)
    }

    return result_df, metrics


def split_train_test(values: np.ndarray, test_percentage: int):
    n = len(values)
    test_size = max(1, int(round(n * test_percentage / 100)))
    test_size = min(test_size, n - 2)

    train = values[:-test_size]
    test = values[-test_size:]
    return train, test, test_size


def parse_weights(weight_text: str):
    try:
        weights = [float(x.strip()) for x in weight_text.split(",") if x.strip() != ""]
        weights = [w for w in weights if w > 0]
        if len(weights) == 0:
            return [0.2, 0.3, 0.5]
        total = sum(weights)
        return [w / total for w in weights]
    except Exception:
        return [0.2, 0.3, 0.5]


def make_period_labels(df: pd.DataFrame, period_col):
    if period_col is None:
        return [f"Periode {i}" for i in range(1, len(df) + 1)], None

    raw_period = df[period_col]
    parsed = pd.to_datetime(raw_period, errors="coerce")
    valid_ratio = parsed.notna().mean()

    if valid_ratio >= 0.7:
        return parsed.dt.strftime("%Y-%m-%d").fillna(raw_period.astype(str)).tolist(), parsed
    return raw_period.astype(str).tolist(), None


def make_future_labels(period_dates, existing_labels, horizon: int):
    if period_dates is not None and period_dates.notna().sum() >= 2:
        valid_dates = period_dates.dropna().reset_index(drop=True)
        try:
            inferred_freq = pd.infer_freq(valid_dates)
        except Exception:
            inferred_freq = None

        last_date = valid_dates.iloc[-1]
        if inferred_freq is not None:
            future_dates = pd.date_range(
                start=last_date,
                periods=horizon + 1,
                freq=inferred_freq
            )[1:]
            return future_dates.strftime("%Y-%m-%d").tolist()

        delta = valid_dates.iloc[-1] - valid_dates.iloc[-2]
        future_dates = [last_date + (i * delta) for i in range(1, horizon + 1)]
        return [d.strftime("%Y-%m-%d") for d in future_dates]

    return [f"Periode {len(existing_labels) + i}" for i in range(1, horizon + 1)]


# --- FUNGSI GRAFIK PLOTLY (COLORFUL & VIBRANT THESIS STYLE) ---

def plot_actual_forecast(periods, actual, forecast, title):
    fig = go.Figure()
    # Warna Aktual: Indigo Cerah
    fig.add_trace(go.Scatter(x=periods, y=actual, mode="lines+markers", name="Aktual", line=dict(color='#4F46E5', width=3)))
    # Warna Forecast: Pink/Neon Red Cerah
    fig.add_trace(go.Scatter(x=periods, y=forecast, mode="lines+markers", name="Forecast", line=dict(color='#EC4899', width=3)))
    fig.update_layout(title=title, xaxis_title="Periode", yaxis_title="Nilai", hovermode="x unified", template="plotly_white")
    return fig


def plot_future_forecast_with_ci(all_periods, actual_values, future_periods, future_forecast, residual_std=0):
    fig = go.Figure()
    
    # Historis (Vibrant Indigo)
    fig.add_trace(go.Scatter(x=all_periods, y=actual_values, mode="lines+markers", name="Aktual Historis", line=dict(color='#4F46E5', width=3)))
    
    # Interval Keyakinan (Soft Pink Transparent)
    if residual_std > 0:
        upper_bound = future_forecast + (1.96 * residual_std)
        lower_bound = future_forecast - (1.96 * residual_std)
        lower_bound = np.clip(lower_bound, 0, None)
        
        fig.add_trace(go.Scatter(
            x=future_periods + future_periods[::-1],
            y=list(upper_bound) + list(lower_bound[::-1]),
            fill='toself',
            fillcolor='rgba(236, 72, 153, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="Interval Keyakinan (95%)"
        ))

    # Garis Forecast Masa Depan (Vibrant Pink Putus-putus)
    fig.add_trace(go.Scatter(x=future_periods, y=future_forecast, mode="lines+markers", name="Proyeksi Utama", line=dict(color='#EC4899', width=3, dash='dash')))

    fig.update_layout(
        title="Grafik Proyeksi Nilai Masa Depan",
        xaxis_title="Periode",
        yaxis_title="Nilai",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig


# --- ALGORITMA METODE PERAMALAN ---

def forecast_naive(history, horizon, **kwargs):
    if len(history) == 0: return np.zeros(horizon)
    return np.repeat(history[-1], horizon)

def forecast_moving_average(history, horizon, window=3, **kwargs):
    history_list = list(history)
    forecasts = []
    for _ in range(horizon):
        usable_window = min(window, len(history_list))
        pred = np.mean(history_list[-usable_window:])
        forecasts.append(pred)
        history_list.append(pred)
    return np.array(forecasts)

def forecast_weighted_moving_average(history, horizon, weights=None, **kwargs):
    if weights is None: weights = [0.2, 0.3, 0.5]
    history_list = list(history)
    forecasts = []
    for _ in range(horizon):
        usable_window = min(len(weights), len(history_list))
        recent_values = np.array(history_list[-usable_window:], dtype=float)
        recent_weights = np.array(weights[-usable_window:], dtype=float)
        recent_weights = recent_weights / recent_weights.sum()
        pred = np.sum(recent_values * recent_weights)
        forecasts.append(pred)
        history_list.append(pred)
    return np.array(forecasts)

def get_fitted_param(fitted, keys):
    for key in keys:
        value = fitted.params.get(key, None)
        if value is not None: return value
    return None

def format_param(value):
    if value is None or pd.isna(value): return "-"
    try: return f"{float(value):.4f}"
    except Exception: return "-"

def limit_smoothing_param(value, minimum=0.01, maximum=0.99):
    try:
        if value is None or pd.isna(value): return minimum
        value = float(value)
        if value < minimum: return minimum
        value = maximum if value > maximum else value
        return value
    except Exception: return minimum

def forecast_single_exponential_smoothing(history, horizon, optimized=True, alpha=None, **kwargs):
    if not STATSMODELS_AVAILABLE or len(history) < 3:
        return np.array(forecast_naive(history, horizon)), {}
    try:
        model = SimpleExpSmoothing(history, initialization_method="estimated")
        if optimized:
            fitted_auto = model.fit(optimized=True)
            alpha_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_level"]))
            fitted = model.fit(smoothing_level=alpha_used, optimized=False)
        else:
            alpha_used = limit_smoothing_param(alpha)
            fitted = model.fit(smoothing_level=alpha_used, optimized=False)
        return np.array(fitted.forecast(horizon)), {"Alpha": alpha_used, "Beta": None, "Gamma": None}
    except Exception:
        return np.array(forecast_naive(history, horizon)), {}

def forecast_double_exponential_smoothing(history, horizon, optimized=True, alpha=None, beta=None, **kwargs):
    if not STATSMODELS_AVAILABLE or len(history) < 4:
        return np.array(forecast_naive(history, horizon)), {}
    try:
        model = ExponentialSmoothing(history, trend="add", seasonal=None, initialization_method="estimated")
        if optimized:
            fitted_auto = model.fit(optimized=True)
            alpha_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_level"]))
            beta_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_trend", "smoothing_slope"]))
            fitted = model.fit(smoothing_level=alpha_used, smoothing_trend=beta_used, optimized=False)
        else:
            alpha_used = limit_smoothing_param(alpha)
            beta_used = limit_smoothing_param(beta)
            fitted = model.fit(smoothing_level=alpha_used, smoothing_trend=beta_used, optimized=False)
        return np.array(fitted.forecast(horizon)), {"Alpha": alpha_used, "Beta": beta_used, "Gamma": None}
    except Exception:
        return np.array(forecast_naive(history, horizon)), {}

def forecast_triple_exponential_smoothing(history, horizon, seasonal_periods=12, optimized=True, alpha=None, beta=None, gamma=None, **kwargs):
    min_data = max(2 * seasonal_periods, seasonal_periods + 4)
    if not STATSMODELS_AVAILABLE or len(history) < min_data:
        return forecast_double_exponential_smoothing(history, horizon, optimized=optimized, alpha=alpha, beta=beta)
    try:
        model = ExponentialSmoothing(history, trend="add", seasonal="add", seasonal_periods=seasonal_periods, initialization_method="estimated")
        if optimized:
            fitted_auto = model.fit(optimized=True)
            alpha_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_level"]))
            beta_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_trend", "smoothing_slope"]))
            gamma_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_seasonal"]))
            fitted = model.fit(smoothing_level=alpha_used, smoothing_trend=beta_used, smoothing_seasonal=gamma_used, optimized=False)
        else:
            alpha_used = limit_smoothing_param(alpha)
            beta_used = limit_smoothing_param(beta)
            gamma_used = limit_smoothing_param(gamma)
            fitted = model.fit(smoothing_level=alpha_used, smoothing_trend=beta_used, smoothing_seasonal=gamma_used, optimized=False)
        return np.array(fitted.forecast(horizon)), {"Alpha": alpha_used, "Beta": beta_used, "Gamma": gamma_used}
    except Exception:
        return forecast_double_exponential_smoothing(history, horizon, optimized=optimized, alpha=alpha, beta=beta)

def forecast_linear_trend(history, horizon, **kwargs):
    if len(history) < 2: return forecast_naive(history, horizon)
    x = np.arange(1, len(history) + 1)
    y = np.array(history, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(len(history) + 1, len(history) + horizon + 1)
    return np.array(intercept + slope * future_x)

def forecast_least_square_quadratic(history, horizon, **kwargs):
    if len(history) < 3: return forecast_linear_trend(history, horizon)
    x = np.arange(1, len(history) + 1)
    y = np.array(history, dtype=float)
    a, b, c = np.polyfit(x, y, 2)
    future_x = np.arange(len(history) + 1, len(history) + horizon + 1)
    return np.array(a * (future_x ** 2) + b * future_x + c)

def forecast_seasonal_naive(history, horizon, seasonal_periods=12, **kwargs):
    if len(history) < seasonal_periods: return forecast_naive(history, horizon)
    history_list = list(history)
    forecasts = []
    for _ in range(horizon):
        pred = history_list[-seasonal_periods]
        forecasts.append(pred)
        history_list.append(pred)
    return np.array(forecasts)

def forecast_arima(history, horizon, arima_order=(1, 1, 1), **kwargs):
    if not STATSMODELS_AVAILABLE or len(history) < 8: return forecast_naive(history, horizon)
    try:
        model = ARIMA(history, order=arima_order)
        fitted_model = model.fit()
        return np.array(fitted_model.forecast(steps=horizon))
    except Exception:
        return forecast_naive(history, horizon)


FORECAST_METHODS = {
    "Naive Forecast": forecast_naive,
    "Moving Average": forecast_moving_average,
    "Weighted Moving Average": forecast_weighted_moving_average,
    "Single Exponential Smoothing": forecast_single_exponential_smoothing,
    "Double Exponential Smoothing": forecast_double_exponential_smoothing,
    "Triple Exponential Smoothing": forecast_triple_exponential_smoothing,
    "Linear Trend Projection": forecast_linear_trend,
    "Least Square Quadratic Trend": forecast_least_square_quadratic,
    "Seasonal Naive": forecast_seasonal_naive,
    "ARIMA": forecast_arima
}

def run_forecast(method_name, history, horizon, params):
    method_function = FORECAST_METHODS[method_name]
    result = method_function(history, horizon, **params)
    return np.array(result[0] if isinstance(result, tuple) else result, dtype=float)

def run_forecast_with_params(method_name, history, horizon, params):
    method_function = FORECAST_METHODS[method_name]
    result = method_function(history, horizon, **params)
    if isinstance(result, tuple):
        return np.array(result[0], dtype=float), result[1]
    return np.array(result, dtype=float), {}

def evaluate_one_method(method_name, train, test, test_periods, params):
    forecast = run_forecast(method_name, train, len(test), params)
    error_table, metrics = calculate_error_table(test_periods, test, forecast)
    return forecast, error_table, metrics

def evaluate_all_methods(train, test, test_periods, params):
    rows = []
    details = {}
    for method_name in FORECAST_METHODS.keys():
        forecast, error_table, metrics = evaluate_one_method(method_name, train, test, test_periods, params)
        rows.append({
            "Metode": method_name,
            "MAD": metrics["MAD"],
            "MSE": metrics["MSE"],
            "MAPE": metrics["MAPE"]
        })
        details[method_name] = {"forecast": forecast, "error_table": error_table, "metrics": metrics}
    
    comparison_df = pd.DataFrame(rows).sort_values(by=["MAPE", "MAD", "MSE"], ascending=True, na_position="last").reset_index(drop=True)
    return comparison_df, details


def convert_all_to_excel(comparison_df, best_method_name, future_labels, future_forecast):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        comparison_df.to_excel(writer, index=False, sheet_name='Perbandingan_Metode')
        
        best_df = pd.DataFrame({"Periode": future_labels, "Forecast Utama": future_forecast})
        best_df.to_excel(writer, index=False, sheet_name='Proyeksi_Metode_Terbaik')
        
        workbook  = writer.book
        
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#4F46E5', 'font_color': '#FFFFFF', 
            'border': 1, 'align': 'center', 'valign': 'vcenter'
        })
        num_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1, 'align': 'right'})
        text_format = workbook.add_format({'border': 1, 'align': 'left'})
        
        # Format Sheet 1
        ws1 = writer.sheets['Perbandingan_Metode']
        ws1.set_row(0, 24)
        for col_num, value in enumerate(comparison_df.columns.values):
            ws1.write(0, col_num, value, header_format)
            
        for i, col in enumerate(comparison_df.columns):
            max_len = max(comparison_df[col].astype(str).map(len).max(), len(col)) + 4
            if col == "Metode":
                ws1.set_column(i, i, max_len, text_format)
            else:
                ws1.set_column(i, i, max_len, num_format)
                
        # Format Sheet 2
        ws2 = writer.sheets['Proyeksi_Metode_Terbaik']
        ws2.set_row(0, 24)
        for col_num, value in enumerate(best_df.columns.values):
            ws2.write(0, col_num, value, header_format)
            
        for i, col in enumerate(best_df.columns):
            max_len = max(best_df[col].astype(str).map(len).max(), len(col)) + 5
            if col == "Periode":
                ws2.set_column(i, i, max_len, text_format)
            else:
                ws2.set_column(i, i, max_len, num_format)
            
    return output.getvalue()


def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil_Proyeksi')
        
        workbook  = writer.book
        worksheet = writer.sheets['Hasil_Proyeksi']
        worksheet.set_row(0, 24)
        
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#4F46E5', 'font_color': '#FFFFFF', 
            'border': 1, 'align': 'center', 'valign': 'vcenter'
        })
        num_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1, 'align': 'right'})
        text_format = workbook.add_format({'border': 1, 'align': 'left'})
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 4
            if col in ["No", "Periode"]:
                worksheet.set_column(i, i, max_len, text_format)
            else:
                worksheet.set_column(i, i, max_len, num_format)
                
    return output.getvalue()


# --- INTERFACE UTAMA DASHBOARD ---

st.title("✨ Dashboard Peramalan Data Historis")
st.write("Aplikasi analitik interaktif berbasis sains data untuk menghitung peramalan tingkat lanjut.")

if not STATSMODELS_AVAILABLE:
    st.warning("⚠️ Library statsmodels belum tersedia. Metode Exponential Smoothing dan ARIMA memakai fallback Naive Forecast.")

with st.sidebar:
    st.header("🔮 Pengaturan Input")
    uploaded_file = st.file_uploader("Upload data historis (.csv / .xlsx)", type=["csv", "xlsx"])
    st.divider()

    st.header("⚙️ Pengaturan Evaluasi")
    test_percentage = st.slider("Persentase data uji (%)", min_value=10, max_value=50, value=20, step=5)
    future_horizon = st.number_input("Jumlah periode ke depan", min_value=1, max_value=60, value=6, step=1)
    mode = st.radio("Mode Perhitungan", ["Satu metode", "Bandingkan semua metode"])
    selected_method = st.selectbox("Pilih Metode Utama", list(FORECAST_METHODS.keys()))
    st.divider()

    st.header("🛠️ Parameter Tambahan")
    st.write("**Exponential Smoothing Parameters**")
    smoothing_mode = st.radio("Metode Penyetelan", ["Optimasi otomatis", "Input manual"])

    if smoothing_mode == "Input manual":
        alpha_input = st.slider("Alpha (Level)", min_value=0.01, max_value=0.99, value=0.30, step=0.01)
        beta_input = st.slider("Beta (Trend)", min_value=0.01, max_value=0.99, value=0.20, step=0.01)
        gamma_input = st.slider("Gamma (Seasonality)", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    else:
        alpha_input = beta_input = gamma_input = None

    ma_window = st.number_input("Window Moving Average", min_value=2, max_value=24, value=3, step=1)
    wma_weight_text = st.text_input("Bobot WMA (Pisahkan koma)", value="0.2, 0.3, 0.5")
    seasonal_periods = st.number_input("Seasonal Periods (Musiman)", min_value=2, max_value=52, value=12, step=1)

    st.write("**ARIMA Parameters (p, d, q)**")
    arima_p = st.number_input("Order p (AR)", min_value=0, max_value=5, value=1, step=1)
    arima_d = st.number_input("Order d (I)", min_value=0, max_value=2, value=1, step=1)
    arima_q = st.number_input("Order q (MA)", min_value=0, max_value=5, value=1, step=1)

    process_button = st.button("🚀 Jalankan Proses", type="primary")


if uploaded_file is None:
    st.info("💡 Petunjuk: Silakan unggah berkas excel atau csv kamu di panel bagian kiri untuk memulai analisis.")
    st.subheader("📋 Contoh Struktur Tabel Excel/CSV yang Benar")
    sample = pd.DataFrame({
        "Tanggal": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "Penjualan": [120, 135, 128, 140, 150, 160, 155, 170, 180, 175, 190, 200]
    })
    preview_df = sample.copy()
    preview_df.insert(0, "No", range(1, len(preview_df) + 1))
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"File gagal dibaca: {e}"); st.stop()

if df_raw.empty:
    st.error("File tidak memiliki data."); st.stop()

st.subheader("📊 Pratinjau Data Unggahan")
preview_df = df_raw.head(20).copy()
preview_df.insert(0, "No", range(1, len(preview_df) + 1))
st.dataframe(preview_df, use_container_width=True, hide_index=True)

columns = df_raw.columns.tolist()
col1, col2 = st.columns(2)

with col1:
    period_options = ["Tidak ada"] + columns
    default_period_index = period_options.index("Tanggal") if "Tanggal" in period_options else 0
    period_option = st.selectbox("Pilih Kolom Indeks Waktu/Periode", period_options, index=default_period_index)

with col2:
    default_value_index = columns.index("Penjualan") if "Penjualan" in columns else 0
    value_col = st.selectbox("Pilih Kolom Nilai Aktual (Numerik)", columns, index=default_value_index)

period_col = None if period_option == "Tidak ada" else period_option
df = df_raw.copy()

if period_col is not None:
    temp_date = pd.to_datetime(df[period_col], errors="coerce")
    if temp_date.notna().mean() >= 0.7:
        df["_parsed_period"] = temp_date
        df = df.sort_values("_parsed_period").drop(columns=["_parsed_period"])
    else:
        df = df.sort_values(period_col)

values_series = clean_numeric_series(df[value_col])
if len(values_series) < 6:
    st.error("Data numerik terdeteksi terlalu sedikit! Mohon gunakan minimal 6 baris data numerik.")
    st.stop()

df = df.loc[values_series.index].reset_index(drop=True)
values = values_series.reset_index(drop=True).values
period_labels, period_dates = make_period_labels(df, period_col)

params = {
    "window": int(ma_window), "weights": parse_weights(wma_weight_text), "seasonal_periods": int(seasonal_periods),
    "arima_order": (int(arima_p), int(arima_d), int(arima_q)), "optimized": smoothing_mode == "Optimasi otomatis",
    "alpha": alpha_input, "beta": beta_input, "gamma": gamma_input
}

train, test, test_size = split_train_test(values, test_percentage)
train_periods = period_labels[:-test_size]
test_periods = period_labels[-test_size:]

st.subheader("📌 Ringkasan Distribusi Data")
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Total Observasi", len(values))
metric_col2.metric("Dataset Latih (Train)", len(train))
metric_col3.metric("Dataset Uji (Test Validation)", len(test))


# --- AREA IMPLEMENTASI GRID TAB & PROSES UTAMA ---

tab_data, tab_grafik = st.tabs(["🔍 Karakteristik & Tren Data", "📊 Hasil Komputasi Peramalan"])

with tab_data:
    st.write("### Analisis Karakteristik Data Historis")
    c_desc, c_roll = st.columns([1, 2])
    
    with c_desc:
        st.write("**Statistik Deskriptif Utama**")
        desc_df = pd.DataFrame(values, columns=[value_col]).describe()
        st.dataframe(desc_df, use_container_width=True)
        
    with c_roll:
        st.write("**Deteksi Pergerakan Tren (Rolling Mean)**")
        roll_df = pd.DataFrame({"Periode": period_labels, "Aktual": values})
        roll_df["Rolling_Mean"] = roll_df["Aktual"].rolling(window=min(3, len(values)), min_periods=1).mean()
        
        roll_fig = go.Figure()
        roll_fig.add_trace(go.Scatter(x=roll_df["Periode"], y=roll_df["Aktual"], name="Aktual", mode="lines", line=dict(color="#4F46E5")))
        roll_fig.add_trace(go.Scatter(x=roll_df["Periode"], y=roll_df["Rolling_Mean"], name="Tren Isyarat (MA)", line=dict(dash='dot', color='#06B6D4', width=2)))
        roll_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
        st.plotly_chart(roll_fig, use_container_width=True)

with tab_grafik:
    history_fig = go.Figure()
    history_fig.add_trace(go.Scatter(x=period_labels, y=values, mode="lines+markers", name="Nilai Aktual", line=dict(color="#A855F7", width=3)))
    history_fig.update_layout(title="Visualisasi Runtun Waktu Historis", xaxis_title="Periode", yaxis_title="Nilai", template="plotly_white")
    st.plotly_chart(history_fig, use_container_width=True)

    if process_button:
        if mode == "Satu metode":
            forecast_test, error_table, metrics = evaluate_one_method(selected_method, train, test, test_periods, params)
            _, used_params = run_forecast_with_params(selected_method, train, len(test), params)
            future_forecast = run_forecast(selected_method, values, int(future_horizon), params)
            future_labels = make_future_labels(period_dates, period_labels, int(future_horizon))

            residuals = test - forecast_test
            std_error = np.std(residuals)

            st.subheader(f"📊 Hasil Analisis: {selected_method}")
            with st.container(border=True):
                st.write("**🎯 Metrik Validasi Tingkat Akurasi**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Akurasi (MAPE)", f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics['MAPE']) else "N/A")
                m2.metric("Error (MAD)", f"{metrics['MAD']:.4f}")
                m3.metric("Error (MSE)", f"{metrics['MSE']:.4f}")

            if selected_method in ["Single Exponential Smoothing", "Double Exponential Smoothing", "Triple Exponential Smoothing"]:
                with st.container(border=True):
                    st.write("**⚙️ Nilai Parameter Alpha, Beta, Gamma Optimal**")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Alpha (Level)", format_param(used_params.get("Alpha")))
                    p2.metric("Beta (Trend)", format_param(used_params.get("Beta")))
                    p3.metric("Gamma (Seasonality)", format_param(used_params.get("Gamma")))

            st.write("") 
            tab1, tab2 = st.tabs(["📉 Grafik Evaluasi Model", "🔮 Hasil Proyeksi Masa Depan"])

            with tab1:
                st.plotly_chart(plot_actual_forecast(test_periods, test, forecast_test, "Uji Validasi: Data Aktual vs Estimasi Model"), use_container_width=True)
                with st.expander("Lihat Rincian Tabel Komputasi Error"):
                    error_table_view = error_table.copy()
                    error_table_view.insert(0, "No", range(1, len(error_table_view) + 1))
                    st.dataframe(error_table_view, use_container_width=True, hide_index=True)

            with tab2:
                st.write("### 🔮 Proyeksi Nilai Masa Depan")
                st.plotly_chart(plot_future_forecast_with_ci(period_labels, values, future_labels, future_forecast, std_error), use_container_width=True)
                st.divider()

                col_tabel, col_download = st.columns([2, 1])
                with col_tabel:
                    st.write("**Tabel Angka Hasil Prediksi**")
                    f_df = pd.DataFrame({"Periode": future_labels, "Hasil Forecast": future_forecast})
                    f_df_view = f_df.copy()
                    f_df_view.insert(0, "No", range(1, len(f_df_view) + 1))
                    st.dataframe(f_df_view, use_container_width=True, hide_index=True)
                
                with col_download:
                    st.write("**Unduh Hasil**")
                    excel_data = convert_df_to_excel(f_df)
                    st.download_button(
                        label="📥 Download Hasil (.xlsx)",
                        data=excel_data,
                        file_name=f"Hasil_Proyeksi_{selected_method}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        else:
            # Mode: Bandingkan semua metode
            comparison_df, details = evaluate_all_methods(train, test, test_periods, params)
            best_method = comparison_df.iloc[0]["Metode"]
            
            future_forecast = run_forecast(best_method, values, int(future_horizon), params)
            future_labels = make_future_labels(period_dates, period_labels, int(future_horizon))
            
            st.subheader("🏆 Tabel Perbandingan Akurasi Semua Metode")
            st.write("Diurutkan otomatis dari model dengan tingkat error (MAPE) terkecil.")
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.success(f"💡 Rekomendasi Model Terbaik Berdasarkan Uji Akurasi: **{best_method}**")
            
            st.plotly_chart(plot_future_forecast_with_ci(period_labels, values, future_labels, future_forecast, 0), use_container_width=True)
            
            excel_all = convert_all_to_excel(comparison_df, best_method, future_labels, future_forecast)
            st.download_button(
                label="📥 Download Laporan Perbandingan Lengkap (.xlsx)",
                data=excel_all,
                file_name="Laporan_Perbandingan_Metode_Peramalan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
