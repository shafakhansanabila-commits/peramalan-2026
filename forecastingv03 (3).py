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
    page_title="Forecasting Dashboard ELITE 2026",
    page_icon="📈",
    layout="wide"
)

# CSS Injection - VIBRANT COLORFUL LIGHT STYLE
st.markdown("""
    <style>
    /* 1. Font & Main Background Foundation */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #F5F7FF 0%, #FAFAFA 100%) !important;
    }

    /* 2. Bright & Fresh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 2px solid #E0E4FF !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stWidgetLabel p,
    [data-testid="stSidebar"] p {
        color: #4A5568 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #4F46E5 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-bottom: 2px solid #F0F2FF;
        padding-bottom: 8px;
        margin-top: 20px !important;
        letter-spacing: 0.5px;
    }

    /* 3. Colorful File Uploader Layout */
    [data-testid="stFileUploader"] {
        background-color: #F0F4FF !important;
        border: 2px dashed #6366F1 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2) !important;
    }
    
    [data-testid="stFileUploader"] button * {
        font-size: 0px !important;
        color: transparent !important;
        display: none !important;
    }
    
    [data-testid="stFileUploader"] button::after {
        content: "Choose Your File" !important;
        color: #FFFFFF !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        display: block !important;
    }

    [data-testid="stFileUploader"] text {
        fill: #4F46E5 !important;
    }
    [data-testid="stFileUploader"] div {
        color: #4A5568 !important;
        font-weight: 500;
    }

    /* 4. Main Content Area */
    .main h1 {
        background: linear-gradient(135deg, #4F46E5 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.3rem !important;
        margin-bottom: 8px !important;
    }
    
    .main p {
        color: #64748B !important;
        font-weight: 500;
    }

    /* 5. Colorful Pastel Metric Cards */
    [data-testid="stMetricValue"] {
        background: #FFFFFF !important;
        color: #1E1B4B !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        border-radius: 12px !important;
        padding: 15px 20px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05) !important;
        border-left: 5px solid #EC4899 !important;
        border-top: 1px solid #EEF2F6 !important;
        border-right: 1px solid #EEF2F6 !important;
        border-bottom: 1px solid #EEF2F6 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #4F46E5 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        margin-left: 5px !important;
    }

    /* 6. Pro Main Button (Neon/Bright Coral Style) */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%) !important;
        color: #FFFFFF !important;            
        font-weight: 700 !important;          
        font-size: 1rem;
        padding: 0.65rem 1rem;
        border: none !important;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 75, 75, 0.4);
        background: linear-gradient(135deg, #FF6B6B 0%, #FF4B4B 100%) !important;
        color: #FFFFFF !important;            
    }

    /* 7. Data Grid Table Design */
    .stDataFrame {
        background-color: #FFFFFF;
        border: 2px solid #F0F2FF !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);
    }

    /* Colorful Streamlit Tabs Customization */
    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        color: #64748B !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4F46E5 !important;
        border-bottom-color: #4F46E5 !important;
    }

    hr {
        margin: 1.5rem 0 !important;
        border-color: #E2E8F0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE PROCESSING AND CALCULATION FUNCTIONS ---

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
        "Period": periods,
        "Actual": actual,
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
        return [f"Period {i}" for i in range(1, len(df) + 1)], None

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

    return [f"Period {len(existing_labels) + i}" for i in range(1, horizon + 1)]


# --- PLOTLY CHART FUNCTIONS (COLORFUL & VIBRANT THESIS STYLE) ---

def plot_actual_forecast(periods, actual, forecast, title):
    fig = go.Figure()
    # Actual Color: Bright Indigo
    fig.add_trace(go.Scatter(x=periods, y=actual, mode="lines+markers", name="Actual", line=dict(color='#4F46E5', width=3)))
    # Forecast Color: Bright Pink/Neon Red
    fig.add_trace(go.Scatter(x=periods, y=forecast, mode="lines+markers", name="Forecast", line=dict(color='#EC4899', width=3)))
    fig.update_layout(title=title, xaxis_title="Period", yaxis_title="Value", hovermode="x unified", template="plotly_white")
    return fig


def plot_future_forecast_with_ci(all_periods, actual_values, future_periods, future_forecast, residual_std=0):
    fig = go.Figure()
    
    # Historical (Vibrant Indigo)
    fig.add_trace(go.Scatter(x=all_periods, y=actual_values, mode="lines+markers", name="Historical Actual", line=dict(color='#4F46E5', width=3)))
    
    # Confidence Interval (Soft Pink Transparent)
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
            name="Confidence Interval (95%)"
        ))

    # Future Forecast Line (Vibrant Dashed Pink)
    fig.add_trace(go.Scatter(x=future_periods, y=future_forecast, mode="lines+markers", name="Main Projection", line=dict(color='#EC4899', width=3, dash='dash')))

    fig.update_layout(
        title="Future Value Projection Chart",
        xaxis_title="Period",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig


# --- FORECASTING METHOD ALGORITHMS ---

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
            "Method": method_name,
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
        comparison_df.to_excel(writer, index=False, sheet_name='Method_Comparison')
        
        best_df = pd.DataFrame({"Period": future_labels, "Main Forecast": future_forecast})
        best_df.to_excel(writer, index=False, sheet_name='Best_Method_Projection')
        
        workbook  = writer.book
        
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#4F46E5', 'font_color': '#FFFFFF', 
            'border': 1, 'align': 'center', 'valign': 'vcenter'
        })
        num_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1, 'align': 'right'})
        text_format = workbook.add_format({'border': 1, 'align': 'left'})
        
        # Format Sheet 1
        ws1 = writer.sheets['Method_Comparison']
        ws1.set_row(0, 24)
        for col_num, value in enumerate(comparison_df.columns.values):
            ws1.write(0, col_num, value, header_format)
            
        for i, col in enumerate(comparison_df.columns):
            max_len = max(comparison_df[col].astype(str).map(len).max(), len(col)) + 4
            if col == "Method":
                ws1.set_column(i, i, max_len, text_format)
            else:
                ws1.set_column(i, i, max_len, num_format)
                
        # Format Sheet 2
        ws2 = writer.sheets['Best_Method_Projection']
        ws2.set_row(0, 24)
        for col_num, value in enumerate(best_df.columns.values):
            ws2.write(0, col_num, value, header_format)
            
        for i, col in enumerate(best_df.columns):
            max_len = max(best_df[col].astype(str).map(len).max(), len(col)) + 5
            if col == "Period":
                ws2.set_column(i, i, max_len, text_format)
            else:
                ws2.set_column(i, i, max_len, num_format)
            
    return output.getvalue()


def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Projection_Results')
        
        workbook  = writer.book
        worksheet = writer.sheets['Projection_Results']
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
            if col in ["No", "Period"]:
                worksheet.set_column(i, i, max_len, text_format)
            else:
                worksheet.set_column(i, i, max_len, num_format)
                
    return output.getvalue()


# --- MAIN DASHBOARD INTERFACE ---

st.title("✨ Predictive Production Planning for MSMEs")
st.write("An interactive data science–based analytics application for advanced forecasting")

if not STATSMODELS_AVAILABLE:
    st.warning("⚠️ The statsmodels library is not available. Exponential Smoothing and ARIMA methods will fall back to Naive Forecast.")

with st.sidebar:
    st.header("🔮 Input Settings")
    uploaded_file = st.file_uploader("Upload Historical Data (.csv / .xlsx)", type=["csv", "xlsx"])
    st.divider()

    st.header("⚙️ Evaluation Settings")
    test_percentage = st.slider("Test Data Percentage (%)", min_value=10, max_value=50, value=20, step=5)
    future_horizon = st.number_input("Number of Future Periods", min_value=1, max_value=60, value=6, step=1)
    mode = st.radio("Calculation Mode", ["Single Method", "Compare All Methods"])
    selected_method = st.selectbox("Select Primary Method", list(FORECAST_METHODS.keys()))
    st.divider()

    st.header("🛠️ Additional Parameters")
    st.write("**Exponential Smoothing Parameters**")
    smoothing_mode = st.radio("Tuning Method", ["Automatic Optimization", "Manual Input"])

    if smoothing_mode == "Manual Input":
        alpha_input = st.slider("Alpha (Level)", min_value=0.01, max_value=0.99, value=0.30, step=0.01)
        beta_input = st.slider("Beta (Trend)", min_value=0.01, max_value=0.99, value=0.20, step=0.01)
        gamma_input = st.slider("Gamma (Seasonality)", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    else:
        alpha_input = beta_input = gamma_input = None

    ma_window = st.number_input("Window Moving Average", min_value=2, max_value=24, value=3, step=1)
    wma_weight_text = st.text_input("WMA Weights (Comma Separated)", value="0.2, 0.3, 0.5")
    seasonal_periods = st.number_input("Seasonal Periods (Seasonal)", min_value=2, max_value=52, value=12, step=1)

    st.write("**ARIMA Parameters (p, d, q)**")
    arima_p = st.number_input("Order p (AR)", min_value=0, max_value=5, value=1, step=1)
    arima_d = st.number_input("Order d (I)", min_value=0, max_value=2, value=1, step=1)
    arima_q = st.number_input("Order q (MA)", min_value=0, max_value=5, value=1, step=1)

    process_button = st.button("🚀 Run Process", type="primary")


if uploaded_file is None:
    st.info("💡 Instruction: Please upload your Excel or CSV file in the left panel to start the analysis.")
    st.subheader("📋 Example of a Proper Excel/CSV Table Structure")
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #F0F4FF 0%, #EEF2FF 100%);
            border: 1.5px solid #A5B4FC;
            border-radius: 10px;
            padding: 12px 18px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        ">
            <span style="font-size: 1.3rem;">📥</span>
            <span style="color: #4A5568; font-weight: 500; font-size: 0.95rem;">
                Download the data template here: &nbsp;
                <a href="https://docs.google.com/spreadsheets/d/1DVghuZxRV-Xj269UMerp_NLRkCuLys5j/edit?usp=sharing&ouid=104043343223917321673&rtpof=true&sd=true"
                   target="_blank"
                   style="
                       color: #4F46E5;
                       font-weight: 700;
                       text-decoration: none;
                       background: #E0E7FF;
                       padding: 4px 12px;
                       border-radius: 6px;
                       border: 1px solid #C7D2FE;
                   ">
                    📊 Open Template in Google Drive
                </a>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    sample = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "Sales": [120, 135, 128, 140, 150, 160, 155, 170, 180, 175, 190, 200]
    })
    preview_df = sample.copy()
    preview_df.insert(0, "No", range(1, len(preview_df) + 1))
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}"); st.stop()

if df_raw.empty:
    st.error("The file contains no data."); st.stop()

st.subheader("📊 Uploaded Data Preview")

# Show summary row count info
total_rows = len(df_raw)
st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%);
        border-left: 4px solid #4F46E5;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span style="font-size: 1.2rem;">📂</span>
        <span style="color: #3730A3; font-weight: 600; font-size: 0.95rem;">
            File loaded successfully — <b>{total_rows} rows</b> of data detected. All rows are used for analysis.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Show ALL rows in scrollable table (no head() limit)
preview_df = df_raw.copy()
if "No" not in preview_df.columns:
    preview_df.insert(0, "No", range(1, len(preview_df) + 1))
st.dataframe(preview_df, use_container_width=True, hide_index=True, height=min(400, 45 + len(preview_df) * 35))

columns = df_raw.columns.tolist()
col1, col2 = st.columns(2)

with col1:
    period_options = ["None"] + columns
    default_period_index = period_options.index("Date") if "Date" in period_options else 0
    period_option = st.selectbox("Select Time/Period Index Column", period_options, index=default_period_index)

with col2:
    default_value_index = columns.index("Sales") if "Sales" in columns else 0
    value_col = st.selectbox("Select Actual Value Column (Numeric)", columns, index=default_value_index)

period_col = None if period_option == "None" else period_option
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
    st.error("Too few numeric data points detected! Please use at least 6 rows of numeric data.")
    st.stop()

df = df.loc[values_series.index].reset_index(drop=True)
values = values_series.reset_index(drop=True).values
period_labels, period_dates = make_period_labels(df, period_col)

params = {
    "window": int(ma_window), "weights": parse_weights(wma_weight_text), "seasonal_periods": int(seasonal_periods),
    "arima_order": (int(arima_p), int(arima_d), int(arima_q)), "optimized": smoothing_mode == "Automatic Optimization",
    "alpha": alpha_input, "beta": beta_input, "gamma": gamma_input
}

train, test, test_size = split_train_test(values, test_percentage)
train_periods = period_labels[:-test_size]
test_periods = period_labels[-test_size:]

st.subheader("📌 Data Distribution Summary")
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Total Observations", len(values))
metric_col2.metric("Training Dataset", len(train))
metric_col3.metric("Test Dataset (Validation)", len(test))


# --- TAB GRID & MAIN PROCESS IMPLEMENTATION AREA ---

tab_data, tab_grafik = st.tabs(["🔍 Data Characteristics & Trends", "📊 Forecasting Computation Results"])

with tab_data:
    st.write("### Historical Data Characteristics Analysis")
    c_desc, c_roll = st.columns([1, 2])
    
    with c_desc:
        st.write("**Main Descriptive Statistics**")
        desc_df = pd.DataFrame(values, columns=[value_col]).describe()
        st.dataframe(desc_df, use_container_width=True)
        
    with c_roll:
        st.write("**Trend Movement Detection (Rolling Mean)**")
        roll_df = pd.DataFrame({"Period": period_labels, "Actual": values})
        roll_df["Rolling_Mean"] = roll_df["Actual"].rolling(window=min(3, len(values)), min_periods=1).mean()
        
        roll_fig = go.Figure()
        roll_fig.add_trace(go.Scatter(x=roll_df["Period"], y=roll_df["Actual"], name="Actual", mode="lines", line=dict(color="#4F46E5")))
        roll_fig.add_trace(go.Scatter(x=roll_df["Period"], y=roll_df["Rolling_Mean"], name="Trend Signal (MA)", line=dict(dash='dot', color='#06B6D4', width=2)))
        roll_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
        st.plotly_chart(roll_fig, use_container_width=True)

with tab_grafik:
    history_fig = go.Figure()
    history_fig.add_trace(go.Scatter(x=period_labels, y=values, mode="lines+markers", name="Actual Values", line=dict(color="#A855F7", width=3)))
    history_fig.update_layout(title="Historical Time Series Visualization", xaxis_title="Period", yaxis_title="Value", template="plotly_white")
    st.plotly_chart(history_fig, use_container_width=True)

    if process_button:
        if mode == "Single Method":
            forecast_test, error_table, metrics = evaluate_one_method(selected_method, train, test, test_periods, params)
            _, used_params = run_forecast_with_params(selected_method, train, len(test), params)
            future_forecast = run_forecast(selected_method, values, int(future_horizon), params)
            future_labels = make_future_labels(period_dates, period_labels, int(future_horizon))

            residuals = test - forecast_test
            std_error = np.std(residuals)

            st.subheader(f"📊 Analysis Results: {selected_method}")
            with st.container(border=True):
                st.write("**🎯 Accuracy Validation Metrics**")
                m1, m2, m3 = st.columns(3)
                mape_val = f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics['MAPE']) else "N/A"
                m1.metric("Accuracy (MAPE)", mape_val)
                m2.metric("Error (MAD)", f"{metrics['MAD']:.4f}")
                m3.metric("Error (MSE)", f"{metrics['MSE']:.4f}")

            if selected_method in ["Single Exponential Smoothing", "Double Exponential Smoothing", "Triple Exponential Smoothing"]:
                with st.container(border=True):
                    st.write("**⚙️ Optimal Alpha, Beta, Gamma Parameter Values**")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Alpha (Level)", format_param(used_params.get("Alpha")))
                    p2.metric("Beta (Trend)", format_param(used_params.get("Beta")))
                    p3.metric("Gamma (Seasonality)", format_param(used_params.get("Gamma")))

            st.write("") 
            tab1, tab2 = st.tabs(["📉 Model Evaluation Chart", "🔮 Future Projection Results"])

            with tab1:
                st.plotly_chart(plot_actual_forecast(test_periods, test, forecast_test, "Validation Test: Actual Data vs Model Estimate"), use_container_width=True)
                with st.expander("View Error Computation Table Details"):
                    error_table_view = error_table.copy()
                    error_table_view.insert(0, "No", range(1, len(error_table_view) + 1))
                    st.dataframe(error_table_view, use_container_width=True, hide_index=True)

            with tab2:
                st.write("### 🔮 Future Value Projection")
                st.plotly_chart(plot_future_forecast_with_ci(period_labels, values, future_labels, future_forecast, std_error), use_container_width=True)
                st.divider()

                col_tabel, col_download = st.columns([2, 1])
                with col_tabel:
                    st.write("**Forecast Result Table**")
                    f_df = pd.DataFrame({"Period": future_labels, "Forecast Result": future_forecast})
                    f_df_view = f_df.copy()
                    f_df_view.insert(0, "No", range(1, len(f_df_view) + 1))
                    st.dataframe(f_df_view, use_container_width=True, hide_index=True)
                
                with col_download:
                    st.write("**Download Results**")
                    excel_data = convert_df_to_excel(f_df)
                    st.download_button(
                        label="📥 Download Results (.xlsx)",
                        data=excel_data,
                        file_name=f"Projection_Results_{selected_method}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        else:
            # Mode: Compare all methods
            comparison_df, details = evaluate_all_methods(train, test, test_periods, params)
            best_method = comparison_df.iloc[0]["Method"]
            
            future_forecast = run_forecast(best_method, values, int(future_horizon), params)
            future_labels = make_future_labels(period_dates, period_labels, int(future_horizon))
            
            st.subheader("🏆 Accuracy Comparison Table for All Methods")
            st.write("Automatically sorted from the model with the lowest error rate (MAPE).")
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.success(f"💡 Best Model Recommendation Based on Accuracy Test: **{best_method}**")
            
            st.plotly_chart(plot_future_forecast_with_ci(period_labels, values, future_labels, future_forecast, 0), use_container_width=True)
            
            excel_all = convert_all_to_excel(comparison_df, best_method, future_labels, future_forecast)
            st.download_button(
                label="📥 Download Full Comparison Report (.xlsx)",
                data=excel_all,
                file_name="Forecasting_Method_Comparison_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
