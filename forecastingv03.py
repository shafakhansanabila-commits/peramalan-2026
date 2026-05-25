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
    page_title="Dashboard Peramalan",
    page_icon="📈",
    layout="wide"
)

# Suntikan CSS untuk mengubah tampilan dasar
st.markdown("""
    <style>
    /* 1. Mengubah Background Utama Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 2px solid #E2E8F0;
    }
    
    /* 2. Membuat Kotak Biru Keren untuk Setiap Widget di Sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%) !important;
        padding: 20px 15px !important;
        border-radius: 14px !important;
        border: 1px solid #BFDBFE !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.05) !important;
        margin-bottom: 15px !important;
    }
    
    [data-testid="stSidebar"] hr {
        margin: 10px 0 !important;
        border-color: #BFDBFE !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] .stWidgetLabel p,
    [data-testid="stSidebar"] p {
        color: #1E3A8A !important;
        font-weight: 700 !important;
    }

    /* 3. Mengubah Font dan Warna Judul Utama (Efek Gradient Biru-Cyan) */
    h1 {
        color: #1E3A8A !important; 
        font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
        font-weight: 800;
        letter-spacing: -0.8px;
        background: linear-gradient(90deg, #1E40AF, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }
    
    h2, h3 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* 4. TAMPILAN TABEL DENGAN SHADING WARNA BIRU (BARU) */
    /* Mewarnai Header Tabel menjadi Biru Navy */
    div[data-testid="stDataFrame"] iframe, 
    div[data-testid="stDataFrame"] data-grid,
    .stDataFrame th {
        background-color: #1E40AF !important;
        color: white !important;
    }
    
    /* Sentuhan custom menggunakan selector internal Streamlit Glide Data Grid */
    div[data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #BFDBFE !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.05) !important;
    }

    /* 5. Membuat "Card" untuk Metrik Ringkasan Data */
    [data-testid="stMetricValue"] {
        background-color: #FFFFFF !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 15px -3px rgba(6, 182, 212, 0.1) !important;
        border: 2px solid #E0F2FE !important;
        color: #2563EB !important; 
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] p {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* 6. Mempercantik Tombol Proses Peramalan (Electric Blue) */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white !important;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.4);
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 6px 20px 0 rgba(29, 78, 216, 0.6);
        transform: translateY(-2px);
        color: white !important;
    }

    /* 7. Styling Kotak Keterangan Alert agar Teks Tetap Jelas */
    .stAlert {
        border-radius: 12px !important;
        border-left: 6px solid !important;
    }
    .stAlert p {
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)

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


def plot_actual_forecast(periods, actual, forecast, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=periods,
        y=actual,
        mode="lines+markers",
        name="Aktual"
    ))

    fig.add_trace(go.Scatter(
        x=periods,
        y=forecast,
        mode="lines+markers",
        name="Forecast"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Periode",
        yaxis_title="Nilai",
        hovermode="x unified",
        template="plotly_white"
    )

    return fig


def plot_future_forecast(all_periods, actual_values, future_periods, future_forecast):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_periods,
        y=actual_values,
        mode="lines+markers",
        name="Aktual Historis"
    ))

    fig.add_trace(go.Scatter(
        x=future_periods,
        y=future_forecast,
        mode="lines+markers",
        name="Forecast Masa Depan"
    ))

    fig.update_layout(
        title="Grafik Forecast Masa Depan",
        xaxis_title="Periode",
        yaxis_title="Nilai",
        hovermode="x unified",
        template="plotly_white"
    )

    return fig


def forecast_naive(history, horizon, **kwargs):
    if len(history) == 0:
        return np.zeros(horizon)

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
    if weights is None:
        weights = [0.2, 0.3, 0.5]

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
        if value is not None:
            return value
    return None


def format_param(value):
    if value is None:
        return "-"

    try:
        if pd.isna(value):
            return "-"

        return f"{float(value):.4f}"

    except Exception:
        return "-"

def limit_smoothing_param(value, minimum=0.01, maximum=0.99):
    try:
        if value is None or pd.isna(value):
            return minimum

        value = float(value)

        if value < minimum:
            return minimum

        if value > maximum:
            return maximum

        return value

    except Exception:
        return minimum

def forecast_single_exponential_smoothing(history, horizon, optimized=True, alpha=None, **kwargs):
    if not STATSMODELS_AVAILABLE or len(history) < 3:
        return np.array(forecast_naive(history, horizon)), {}

    try:
        model = SimpleExpSmoothing(
            history,
            initialization_method="estimated"
        )

        if optimized:
            fitted_auto = model.fit(optimized=True)

            alpha_used = limit_smoothing_param(
                get_fitted_param(fitted_auto, ["smoothing_level"])
            )

            fitted = model.fit(
                smoothing_level=alpha_used,
                optimized=False
            )
        else:
            alpha_used = limit_smoothing_param(alpha)

            fitted = model.fit(
                smoothing_level=alpha_used,
                optimized=False
            )

        used_params = {
            "Alpha": alpha_used,
            "Beta": None,
            "Gamma": None
        }

        return np.array(fitted.forecast(horizon)), used_params

    except Exception:
        return np.array(forecast_naive(history, horizon)), {}

def forecast_double_exponential_smoothing(history, horizon, optimized=True, alpha=None, beta=None, **kwargs):
    if not STATSMODELS_AVAILABLE or len(history) < 4:
        return np.array(forecast_naive(history, horizon)), {}

    try:
        model = ExponentialSmoothing(
            history,
            trend="add",
            seasonal=None,
            initialization_method="estimated"
        )

        if optimized:
            fitted_auto = model.fit(optimized=True)

            alpha_used = limit_smoothing_param(
                get_fitted_param(fitted_auto, ["smoothing_level"])
            )

            beta_used = limit_smoothing_param(
                get_fitted_param(fitted_auto, ["smoothing_trend", "smoothing_slope"])
            )

            fitted = model.fit(
                smoothing_level=alpha_used,
                smoothing_trend=beta_used,
                optimized=False
            )
        else:
            alpha_used = limit_smoothing_param(alpha)
            beta_used = limit_smoothing_param(beta)

            fitted = model.fit(
                smoothing_level=alpha_used,
                smoothing_trend=beta_used,
                optimized=False
            )

        used_params = {
            "Alpha": alpha_used,
            "Beta": beta_used,
            "Gamma": None
        }

        return np.array(fitted.forecast(horizon)), used_params

    except Exception:
        return np.array(forecast_naive(history, horizon)), {}


def forecast_triple_exponential_smoothing(
    history,
    horizon,
    seasonal_periods=12,
    optimized=True,
    alpha=None,
    beta=None,
    gamma=None,
    **kwargs
):
    min_data = max(2 * seasonal_periods, seasonal_periods + 4)

    if not STATSMODELS_AVAILABLE or len(history) < min_data:
        fallback_forecast, fallback_params = forecast_double_exponential_smoothing(
            history,
            horizon,
            optimized=optimized,
            alpha=alpha,
            beta=beta
        )
        return np.array(fallback_forecast), fallback_params

    try:
        model = ExponentialSmoothing(
            history,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        )

        if optimized:
            fitted_auto = model.fit(optimized=True)

            alpha_used = limit_smoothing_param(
                get_fitted_param(fitted_auto, ["smoothing_level"])
            )

            beta_used = limit_smoothing_param(
                get_fitted_param(fitted_auto, ["smoothing_trend", "smoothing_slope"])
            )

            gamma_used = limit_smoothing_param(
                get_fitted_param(fitted_auto, ["smoothing_seasonal"])
            )

            fitted = model.fit(
                smoothing_level=alpha_used,
                smoothing_trend=beta_used,
                smoothing_seasonal=gamma_used,
                optimized=False
            )
        else:
            alpha_used = limit_smoothing_param(alpha)
            beta_used = limit_smoothing_param(beta)
            gamma_used = limit_smoothing_param(gamma)

            fitted = model.fit(
                smoothing_level=alpha_used,
                smoothing_trend=beta_used,
                smoothing_seasonal=gamma_used,
                optimized=False
            )

        used_params = {
            "Alpha": alpha_used,
            "Beta": beta_used,
            "Gamma": gamma_used
        }

        return np.array(fitted.forecast(horizon)), used_params

    except Exception:
        fallback_forecast, fallback_params = forecast_double_exponential_smoothing(
            history,
            horizon,
            optimized=optimized,
            alpha=alpha,
            beta=beta
        )
        return np.array(fallback_forecast), fallback_params

def forecast_linear_trend(history, horizon, **kwargs):
    if len(history) < 2:
        return forecast_naive(history, horizon)

    x = np.arange(1, len(history) + 1)
    y = np.array(history, dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(history) + 1, len(history) + horizon + 1)
    forecast = intercept + slope * future_x

    return np.array(forecast)


def forecast_least_square_quadratic(history, horizon, **kwargs):
    if len(history) < 3:
        return forecast_linear_trend(history, horizon)

    x = np.arange(1, len(history) + 1)
    y = np.array(history, dtype=float)

    a, b, c = np.polyfit(x, y, 2)

    future_x = np.arange(len(history) + 1, len(history) + horizon + 1)
    forecast = a * (future_x ** 2) + b * future_x + c

    return np.array(forecast)


def forecast_seasonal_naive(history, horizon, seasonal_periods=12, **kwargs):
    if len(history) < seasonal_periods:
        return forecast_naive(history, horizon)

    history_list = list(history)
    forecasts = []

    for _ in range(horizon):
        pred = history_list[-seasonal_periods]

        forecasts.append(pred)
        history_list.append(pred)

    return np.array(forecasts)


def forecast_arima(history, horizon, arima_order=(1, 1, 1), **kwargs):
    if not STATSMODELS_AVAILABLE or len(history) < 8:
        return forecast_naive(history, horizon)

    try:
        model = ARIMA(history, order=arima_order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=horizon)

        return np.array(forecast)
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

    if isinstance(result, tuple):
        forecast = result[0]
    else:
        forecast = result

    return np.array(forecast, dtype=float)


def run_forecast_with_params(method_name, history, horizon, params):
    method_function = FORECAST_METHODS[method_name]
    result = method_function(history, horizon, **params)

    if isinstance(result, tuple):
        forecast = result[0]
        used_params = result[1]
    else:
        forecast = result
        used_params = {}

    return np.array(forecast, dtype=float), used_params


def evaluate_one_method(method_name, train, test, test_periods, params):
    forecast = run_forecast(method_name, train, len(test), params)
    error_table, metrics = calculate_error_table(test_periods, test, forecast)

    return forecast, error_table, metrics


def evaluate_all_methods(train, test, test_periods, params):
    rows = []
    details = {}

    for method_name in FORECAST_METHODS.keys():
        forecast, error_table, metrics = evaluate_one_method(
            method_name,
            train,
            test,
            test_periods,
            params
        )

        rows.append({
            "Metode": method_name,
            "MAD": metrics["MAD"],
            "MSE": metrics["MSE"],
            "MAPE": metrics["MAPE"]
        })

        details[method_name] = {
            "forecast": forecast,
            "error_table": error_table,
            "metrics": metrics
        }

    comparison_df = pd.DataFrame(rows)

    comparison_df = comparison_df.sort_values(
        by=["MAPE", "MAD", "MSE"],
        ascending=True,
        na_position="last"
    ).reset_index(drop=True)

    return comparison_df, details

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil_Proyeksi')
        workbook  = writer.book
        worksheet = writer.sheets['Hasil_Proyeksi']
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#1E3A8A', 'font_color': '#FFFFFF',
            'border': 1, 'align': 'center'
        })
        num_format = workbook.add_format({'num_format': '#,##0.0000', 'border': 1})
        text_format = workbook.add_format({'border': 1})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        worksheet.set_column('A:A', 10, text_format)
        worksheet.set_column('B:B', 20, text_format)
        worksheet.set_column('C:C', 25, num_format)
    return output.getvalue()

st.title("Dashboard Peramalan Data Historis")

st.write(
    "Aplikasi ini menghitung peramalan, menampilkan grafik, dan menghitung error "
    "menggunakan MAD, MAPE, dan MSE."
)

if not STATSMODELS_AVAILABLE:
    st.warning(
        "Library statsmodels belum tersedia. Metode Exponential Smoothing dan ARIMA "
        "akan memakai fallback Naive Forecast. Install statsmodels agar semua metode berjalan."
    )

with st.sidebar:
    st.header("Pengaturan Input")

    uploaded_file = st.file_uploader(
        "Upload data historis",
        type=["csv", "xlsx"]
    )

    st.divider()

    st.header("Pengaturan Evaluasi")

    test_percentage = st.slider(
        "Persentase data uji",
        min_value=10,
        max_value=50,
        value=20,
        step=5
    )

    future_horizon = st.number_input(
        "Jumlah periode forecast masa depan",
        min_value=1,
        max_value=60,
        value=6,
        step=1
    )

    mode = st.radio(
        "Mode perhitungan",
        ["Satu metode", "Bandingkan semua metode"]
    )

    selected_method = st.selectbox(
        "Pilih metode",
        list(FORECAST_METHODS.keys())
    )

    st.divider()

    st.header("Parameter Metode")

    st.write("Parameter Exponential Smoothing")

    smoothing_mode = st.radio(
        "Mode parameter smoothing",
        ["Optimasi otomatis", "Input manual"]
    )

    if smoothing_mode == "Input manual":
        alpha_input = st.slider(
            "Alpha",
            min_value=0.01,
            max_value=0.99,
            value=0.30,
            step=0.01
        )

        beta_input = st.slider(
            "Beta",
            min_value=0.01,
            max_value=0.99,
            value=0.20,
            step=0.01
        )

        gamma_input = st.slider(
            "Gamma",
            min_value=0.01,
            max_value=0.99,
            value=0.10,
            step=0.01
        )
    else:
        alpha_input = None
        beta_input = None
        gamma_input = None


    ma_window = st.number_input(
        "Window Moving Average",
        min_value=2,
        max_value=24,
        value=3,
        step=1
    )

    wma_weight_text = st.text_input(
        "Bobot WMA",
        value="0.2, 0.3, 0.5",
        help="Contoh: 0.2, 0.3, 0.5. Bobot otomatis dinormalisasi."
    )

    seasonal_periods = st.number_input(
        "Seasonal periods",
        min_value=2,
        max_value=52,
        value=12,
        step=1
    )

    st.write("Parameter ARIMA")

    arima_p = st.number_input(
        "p",
        min_value=0,
        max_value=5,
        value=1,
        step=1
    )

    arima_d = st.number_input(
        "d",
        min_value=0,
        max_value=2,
        value=1,
        step=1
    )

    arima_q = st.number_input(
        "q",
        min_value=0,
        max_value=5,
        value=1,
        step=1
    )

    process_button = st.button("Proses Peramalan", type="primary")


if uploaded_file is None:
    st.info("Silakan upload file CSV atau Excel terlebih dahulu.")

    st.subheader("Contoh Format Data")

    sample = pd.DataFrame({
        "Tanggal": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "Penjualan": [120, 135, 128, 140, 150, 160, 155, 170, 180, 175, 190, 200]
    })

    preview_df = sample.copy()
    preview_df.insert(0, "No", range(1, len(preview_df) + 1))

    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    st.stop()

try:
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"File gagal dibaca: {e}")
    st.stop()


if df_raw.empty:
    st.error("File tidak memiliki data.")
    st.stop()


st.subheader("Preview Data")

preview_df = df_raw.head(20).copy()
preview_df.insert(0, "No", range(1, len(preview_df) + 1))

st.dataframe(preview_df, use_container_width=True, hide_index=True)

columns = df_raw.columns.tolist()

col1, col2 = st.columns(2)

with col1:
    period_options = ["Tidak ada"] + columns
    default_period_index = period_options.index("Tanggal") if "Tanggal" in period_options else 0

    period_option = st.selectbox(
    "Pilih kolom periode",
    period_options,
    index=default_period_index
    )

with col2:
    default_value_index = columns.index("Penjualan") if "Penjualan" in columns else 0

    value_col = st.selectbox(
        "Pilih kolom nilai aktual",
        columns,
        index=default_value_index
    )

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
    st.error("Data numerik terlalu sedikit. Minimal gunakan 6 baris data historis.")
    st.stop()

df = df.loc[values_series.index].reset_index(drop=True)
values = values_series.reset_index(drop=True).values

period_labels, period_dates = make_period_labels(df, period_col)

params = {
    "window": int(ma_window),
    "weights": parse_weights(wma_weight_text),
    "seasonal_periods": int(seasonal_periods),
    "arima_order": (int(arima_p), int(arima_d), int(arima_q)),
    "optimized": smoothing_mode == "Optimasi otomatis",
    "alpha": alpha_input,
    "beta": beta_input,
    "gamma": gamma_input
}

train, test, test_size = split_train_test(values, test_percentage)

train_periods = period_labels[:-test_size]
test_periods = period_labels[-test_size:]

st.subheader("Ringkasan Data")

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric("Jumlah Data", len(values))
metric_col2.metric("Data Latih", len(train))
metric_col3.metric("Data Uji", len(test))

history_fig = go.Figure()

history_fig.add_trace(go.Scatter(
    x=period_labels,
    y=values,
    mode="lines+markers",
    name="Nilai Aktual"
))

history_fig.update_layout(
    title="Grafik Data Historis",
    xaxis_title="Periode",
    yaxis_title="Nilai",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(history_fig, use_container_width=True)

if process_button:
    if mode == "Satu metode":
        # --- 1. PROSES HITUNG ---
        forecast_test, error_table, metrics = evaluate_one_method(
            selected_method, train, test, test_periods, params
        )
        _, used_params = run_forecast_with_params(
            selected_method, train, len(test), params
        )
        future_forecast = run_forecast(
            selected_method, values, int(future_horizon), params
        )
        future_labels = make_future_labels(
            period_dates, period_labels, int(future_horizon)
        )

        # --- 2. TAMPILAN DASHBOARD ---
        st.subheader(f"📊 Hasil Analisis: {selected_method}")
        
        # Kotak 1: Performa (Output)
        with st.container(border=True):
            st.write("**📈 Performa Model (Error)**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Akurasi (MAPE)", f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics['MAPE']) else "N/A")
            m2.metric("Error (MAD)", f"{metrics['MAD']:.4f}")
            m3.metric("Error (MSE)", f"{metrics['MSE']:.4f}")

        # Kotak 2: Setelan Mesin (Input/Process)
        if selected_method in ["Single Exponential Smoothing", "Double Exponential Smoothing", "Triple Exponential Smoothing"]:
            with st.container(border=True):
                st.write("**⚙️ Konfigurasi Smoothing (Alpha, Beta, Gamma)**")
                p1, p2, p3 = st.columns(3)
                p1.metric("Alpha (Level)", format_param(used_params.get("Alpha")), help="Bobot data terbaru")
                p2.metric("Beta (Trend)", format_param(used_params.get("Beta")), help="Bobot pola tren")
                p3.metric("Gamma (Seasonality)", format_param(used_params.get("Gamma")), help="Bobot pola musiman")
                st.caption(f"Metode optimasi: {smoothing_mode}")

        st.write("") 

        # --- 3. DETAIL VISUAL (TABS) ---
        tab1, tab2 = st.tabs(["📉 Grafik & Validasi", "🔮 Proyeksi Masa Depan"])

        with tab1:
            st.plotly_chart(
                plot_actual_forecast(test_periods, test, forecast_test, "Validasi Model: Aktual vs Prediksi"), 
                use_container_width=True
            )
            with st.expander("Klik untuk cek Tabel Error per Periode"):
                error_table_view = error_table.copy()
                error_table_view.insert(0, "No", range(1, len(error_table_view) + 1))
                st.dataframe(error_table_view, use_container_width=True, hide_index=True)

        with tab2:
            # Kita buat layout satu kolom saja dulu agar grafik terlihat besar dan jelas
            st.write("### 🔮 Proyeksi Tren Masa Depan")
            
            # --- BAGIAN GRAFIK (DI ATAS) ---
            # Kita panggil grafik agar muncul full width dulu supaya "nendang" visualnya
            st.plotly_chart(
                plot_future_forecast(period_labels, values, future_labels, future_forecast),
                use_container_width=True
            )
            
            st.divider()

            # --- BAGIAN DATA (DI BAWAH) ---
            col_tabel, col_download = st.columns([2, 1])
            with col_tabel:
                st.write("**Tabel Angka Proyeksi**")
                f_df = pd.DataFrame({"Periode": future_labels, "Forecast": future_forecast})
                f_df_view = f_df.copy()
                f_df_view.insert(0, "No", range(1, len(f_df_view) + 1))
                st.dataframe(f_df_view, use_container_width=True, hide_index=True)
            
            with col_download:
                st.write("**Opsi Ekspor**")
                excel_data = convert_df_to_excel(f_df_view)
                st.download_button(
                    label="📥 Download Excel (.xlsx)",
                    data=excel_data,
                    file_name=f"Proyeksi_{selected_method.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    else:
        # --- MODE PERBANDINGAN (Agar tidak error lagi) ---
        st.subheader("🏁 Perbandingan Performa Semua Metode")
        
        comparison_df, details = evaluate_all_methods(train, test, test_periods, params)
        
        best_method = comparison_df.iloc[0]["Metode"]
        best_mape = comparison_df.iloc[0]["MAPE"]
        
        # Tampilan banner sukses untuk metode terbaik
        st.success(f"🏆 Metode Terbaik: **{best_method}** dengan MAPE **{best_mape:.2f}%**")

        tab_rank, tab_best = st.tabs(["📊 Tabel Ranking Akurasi", "🏆 Detail Metode Terbaik"])

        with tab_rank:
            st.write("Metode diurutkan dari yang paling akurat (MAPE terkecil):")
            # Memberi warna hijau pada hasil terbaik di tabel
            st.dataframe(comparison_df.style.highlight_min(axis=0, subset=['MAPE'], color='#D1FAE5'), use_container_width=True)

        with tab_best:
            best_detail = details[best_method]
            st.write(f"Grafik validasi untuk metode **{best_method}**:")
            st.plotly_chart(
                plot_actual_forecast(test_periods, test, best_detail["forecast"], f"Performa Terbaik: {best_method}"), 
                use_container_width=True
            )
            
            # Tambahkan juga grafik masa depannya
            f_forecast_best = run_forecast(best_method, values, int(future_horizon), params)
            f_labels_best = make_future_labels(period_dates, period_labels, int(future_horizon))
            st.plotly_chart(
                plot_future_forecast(period_labels, values, f_labels_best, f_forecast_best),
                use_container_width=True
            )    

else:
    st.info("Klik tombol Proses Peramalan untuk menjalankan perhitungan.")
