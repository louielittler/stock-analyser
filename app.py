"""Stock Analyser + Backtester (Upload + Live) — v2

Run locally:
  pip install streamlit pandas numpy openpyxl yfinance plotly
  streamlit run app.py

Deploy (tablets): Streamlit Community Cloud with app.py + requirements.txt

Features:
- Upload CSV/XLSX OR fetch Live data (yfinance)
- Candlestick chart with EMA + Bollinger overlays
- Signals: trend (EMA50/EMA200) + momentum (MACD) + RSI guardrails + optional ADX regime filter
- Backtest: ATR stop/TP, fees+slippage, gap-aware stops, extra stop slippage, risk-based sizing
- Walk-forward: time-based train/test split + out-of-sample metrics
- Parameter sweep: grid search (stop_atr x take_atr) + heatmap
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# Optional imports for Live mode + better charts
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# -----------------------------
# App config
# -----------------------------

st.set_page_config(page_title="Stock Analyser + Backtester", layout="wide")

# -----------------------------
# Indicators
# -----------------------------


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    return mid, upper, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX) to estimate trend strength."""
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    tr_sm = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (
        pd.Series(plus_dm, index=df.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / tr_sm
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=df.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / tr_sm
    )

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


# -----------------------------
# Data loading
# -----------------------------


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["date", "datetime", "time", "timestamp"]:
            mapping[c] = "date"
        elif lc in ["open", "o"]:
            mapping[c] = "open"
        elif lc in ["high", "h"]:
            mapping[c] = "high"
        elif lc in ["low", "l"]:
            mapping[c] = "low"
        elif lc in ["close", "c", "adj close", "adj_close", "adjclose"]:
            mapping[c] = "close"
        elif lc in ["volume", "vol", "v"]:
            mapping[c] = "volume"
    return df.rename(columns=mapping)


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]
    if "open" not in df.columns:
        df["open"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = np.nan
    return df


def load_any(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file)
    else:
        raise ValueError("Upload a .csv or .xlsx/.xls")

    df = normalize_cols(df)

    if "close" not in df.columns:
        raise ValueError("Missing 'Close' column (or something that maps to it).")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.set_index("date")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).copy()
    df = ensure_ohlcv(df)
    return df


@st.cache_data(show_spinner=False)
def load_live_cached(ticker: str, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
    if yf is None:
        raise ValueError("Live mode needs yfinance. Install: pip install yfinance")

    t = ticker.strip()
    if not t:
        raise ValueError("Enter a ticker (e.g., AAPL, TSLA, MSFT, NVDA, ^GSPC)")

    data = yf.download(
        t,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if data is None or data.empty:
        raise ValueError("No data returned. Try interval 1d + period 1y.")

    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",
            "Volume": "volume",
        }
    )

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in data.columns]
    data = data[cols].dropna(subset=["close"]).copy()
    data = ensure_ohlcv(data)
    return data


# -----------------------------
# Signal pipeline
# -----------------------------


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["ema200"] = ema(close, 200)
    df["rsi14"] = rsi(close, 14)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(close)
    df["bb_mid"], df["bb_up"], df["bb_low"] = bollinger(close, 20, 2.0)
    df["atr14"] = atr(df, 14)
    df["atr_pct"] = (df["atr14"] / df["close"]) * 100
    df["adx14"] = adx(df, 14)
    return df


def add_signal(df: pd.DataFrame, use_adx_filter: bool, adx_threshold: float) -> pd.DataFrame:
    """Per-bar action + target position. 1=long, -1=short, 0=flat."""
    df = df.copy()

    uptrend = (df["close"] > df["ema50"]) & (df["ema50"] > df["ema200"])
    downtrend = (df["close"] < df["ema50"]) & (df["ema50"] < df["ema200"])
    trend_score = np.select([uptrend, downtrend], [2, -2], default=0)

    macd_bull = df["macd"] > df["macd_sig"]
    momentum_score = np.where(macd_bull, 1, -1)

    rsi_overbought = df["rsi14"] >= 70
    rsi_oversold = df["rsi14"] <= 30

    total = trend_score + momentum_score
    target = np.select([total >= 2, total <= -2], [1, -1], default=0)

    # RSI guardrails
    target = np.where(rsi_overbought & (target == 1), 0, target)
    target = np.where(rsi_oversold & (target == -1), 0, target)

    # ADX regime filter: avoid trading when trend strength is low
    if use_adx_filter:
        target = np.where(df["adx14"] < adx_threshold, 0, target)

    df["score"] = total
    df["target_pos"] = target.astype(int)
    df["action"] = np.where(target == 1, "BUY", np.where(target == -1, "SHORT", "HOLD"))

    df["rsi_state"] = np.where(
        rsi_overbought, "Overbought", np.where(rsi_oversold, "Oversold", "Neutral")
    )
    df["trend_state"] = np.where(
        uptrend, "Uptrend", np.where(downtrend, "Downtrend", "Sideways / Transition")
    )
    df["macd_state"] = np.where(macd_bull, "bullish", "bearish")
    return df


# -----------------------------
# Backtest
# -----------------------------


def compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity / peak) - 1.0


def annualization_factor(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 252.0
    deltas = index.to_series().diff().dropna().dt.total_seconds().values
    if len(deltas) == 0:
        return 252.0
    median_sec = float(np.median(deltas))
    if median_sec <= 0:
        return 252.0
    days = median_sec / 86400.0
    return 365.0 / days


def backtest_atr_strategy(
    df: pd.DataFrame,
    initial_cash: float,
    risk_per_trade: float,
    stop_atr: float,
    take_atr: float,
    fee_bps: float,
    slippage_bps: float,
    stop_extra_slip_bps: float,
    allow_short: bool,
    fill_on_gap: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Single-position backtest.

    - Enter: next bar OPEN (fallback next CLOSE)
    - Exit: ATR stop/TP intrabar using H/L, or signal flip at close
    - Conservative: if stop+TP same bar, take stop
    - Gap-aware stops: if open gaps through stop, fill at open
    - Fees+slippage: bps on entry & exit; extra bps on stop exits
    """
    df = df.copy().dropna(subset=["atr14", "target_pos"])
    if len(df) < 10:
        return df, pd.DataFrame(), {"error": "Not enough data after indicators/signals."}

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    entry_px_next = o.shift(-1).fillna(c.shift(-1))

    fee_rate = (fee_bps + slippage_bps) / 10_000.0
    stop_slip_rate = stop_extra_slip_bps / 10_000.0

    cash = float(initial_cash)
    pos = 0
    shares = 0.0
    entry_px = np.nan
    stop_px = np.nan
    tp_px = np.nan
    entry_time = None

    equity_points: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    idx = df.index

    for i in range(0, len(df) - 1):
        ts = idx[i]

        # MTM at close
        mtm = cash
        if pos != 0:
            mtm += shares * (c.iloc[i] - entry_px) * pos
        equity_points.append((ts, mtm))

        # Manage open position
        if pos != 0:
            hit_stop = (l.iloc[i] <= stop_px) if pos == 1 else (h.iloc[i] >= stop_px)
            hit_tp = (h.iloc[i] >= tp_px) if pos == 1 else (l.iloc[i] <= tp_px)

            exit_reason = None
            exit_px = None

            if hit_stop and hit_tp:
                exit_reason = "stop_and_tp_same_bar"
                exit_px = float(stop_px)
            elif hit_stop:
                exit_reason = "stop"
                exit_px = float(stop_px)
            elif hit_tp:
                exit_reason = "take_profit"
                exit_px = float(tp_px)

            # Signal flip exit
            if exit_reason is None:
                target_now = int(df["target_pos"].iloc[i])
                if (pos == 1 and target_now == -1) or (pos == -1 and target_now == 1):
                    exit_reason = "signal_flip"
                    exit_px = float(c.iloc[i])

            if exit_reason is not None:
                # Gap-aware stop fills
                if fill_on_gap and exit_reason.startswith("stop"):
                    if pos == 1 and float(o.iloc[i]) < float(stop_px):
                        exit_px = float(o.iloc[i])
                        exit_reason = exit_reason + "_gap"
                    if pos == -1 and float(o.iloc[i]) > float(stop_px):
                        exit_px = float(o.iloc[i])
                        exit_reason = exit_reason + "_gap"

                notional = abs(shares * float(exit_px))
                fee = notional * fee_rate
                extra_stop_slip = notional * stop_slip_rate if exit_reason.startswith("stop") else 0.0

                pnl = shares * (float(exit_px) - entry_px) * pos - fee - extra_stop_slip
                cash += pnl

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "side": "LONG" if pos == 1 else "SHORT",
                        "entry_px": float(entry_px),
                        "exit_px": float(exit_px),
                        "shares": float(shares),
                        "stop_px": float(stop_px),
                        "tp_px": float(tp_px),
                        "reason": exit_reason,
                        "pnl": float(pnl),
                    }
                )

                pos = 0
                shares = 0.0
                entry_px = np.nan
                stop_px = np.nan
                tp_px = np.nan
                entry_time = None

        # Entries when flat
        if pos == 0:
            target = int(df["target_pos"].iloc[i])
            if target == -1 and not allow_short:
                target = 0

            if target != 0:
                next_px = float(entry_px_next.iloc[i])
                if not np.isfinite(next_px) or next_px <= 0:
                    continue

                atr_now = float(df["atr14"].iloc[i])
                if not np.isfinite(atr_now) or atr_now <= 0:
                    continue

                stop_dist = stop_atr * atr_now
                risk_cash = float(mtm) * float(risk_per_trade)
                qty = risk_cash / max(stop_dist, 1e-9)

                # No leverage in this simple model
                max_qty = cash / max(next_px, 1e-9)
                qty = min(qty, max_qty)

                if qty * next_px < 10:
                    continue

                entry_notional = qty * next_px
                entry_fee = entry_notional * fee_rate
                cash -= entry_fee

                pos = int(target)
                shares = float(qty)
                entry_px = float(next_px)
                entry_time = idx[i + 1]

                if pos == 1:
                    stop_px = entry_px - stop_atr * atr_now
                    tp_px = entry_px + take_atr * atr_now
                else:
                    stop_px = entry_px + stop_atr * atr_now
                    tp_px = entry_px - take_atr * atr_now

    # Final equity
    last_ts = idx[-1]
    last_mtm = cash
    if pos != 0:
        last_mtm += shares * (c.iloc[-1] - entry_px) * pos
    equity_points.append((last_ts, last_mtm))

    equity_df = pd.DataFrame(equity_points, columns=["time", "equity"]).set_index("time")
    trades_df = pd.DataFrame(trades)

    ret_series = equity_df["equity"].pct_change().fillna(0.0)
    dd = compute_drawdown(equity_df["equity"])
    ann = annualization_factor(equity_df.index)

    total_return = float(equity_df["equity"].iloc[-1] / initial_cash - 1.0)

    if isinstance(equity_df.index, pd.DatetimeIndex):
        days = max(1.0, (equity_df.index[-1] - equity_df.index[0]).days)
        years = max(days / 365.0, 1e-9)
        cagr = float((equity_df["equity"].iloc[-1] / initial_cash) ** (1 / years) - 1.0)
    else:
        cagr = np.nan

    vol = float(ret_series.std(ddof=0) * np.sqrt(max(ann, 1e-9)))
    mean_ann = float(ret_series.mean() * ann)
    sharpe = float(mean_ann / max(vol, 1e-9)) if np.isfinite(vol) else np.nan
    max_dd = float(dd.min())

    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = float(len(wins) / len(trades_df))
        gross_profit = float(wins["pnl"].sum()) if len(wins) else 0.0
        gross_loss = float(losses["pnl"].sum()) if len(losses) else 0.0
        profit_factor = float(gross_profit / max(abs(gross_loss), 1e-9))
        expectancy = float(trades_df["pnl"].mean())
    else:
        win_rate = np.nan
        profit_factor = np.nan
        expectancy = np.nan

    exposure = float((df["target_pos"].shift(1).fillna(0).astype(int) != 0).mean())

    metrics = {
        "final_equity": float(equity_df["equity"].iloc[-1]),
        "total_return": total_return,
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "annualized_vol": vol,
        "sharpe_like": sharpe,
        "max_drawdown": max_dd,
        "num_trades": int(len(trades_df)),
        "win_rate": float(win_rate) if np.isfinite(win_rate) else np.nan,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else np.nan,
        "expectancy_per_trade": float(expectancy) if np.isfinite(expectancy) else np.nan,
        "exposure_pct": exposure,
    }

    df["equity"] = equity_df["equity"].reindex(df.index).ffill()
    df["drawdown"] = compute_drawdown(df["equity"]).reindex(df.index).fillna(0.0)

    return df, trades_df, metrics


# -----------------------------
# Walk-forward + sweep
# -----------------------------


def split_train_test(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna().copy()
    n = len(df)
    if n < 50:
        return df, df.iloc[0:0]
    cut = int(np.floor(n * (1 - test_fraction)))
    cut = max(30, min(cut, n - 10))
    return df.iloc[:cut], df.iloc[cut:]


def run_backtest_bundle(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return backtest_atr_strategy(
        df,
        initial_cash=float(params["initial_cash"]),
        risk_per_trade=float(params["risk_per_trade"]),
        stop_atr=float(params["stop_atr"]),
        take_atr=float(params["take_atr"]),
        fee_bps=float(params["fee_bps"]),
        slippage_bps=float(params["slippage_bps"]),
        stop_extra_slip_bps=float(params["stop_extra_slip_bps"]),
        allow_short=bool(params["allow_short"]),
        fill_on_gap=bool(params["fill_on_gap"]),
    )


def sweep_params(df: pd.DataFrame, params: dict, stop_grid: list[float], take_grid: list[float]) -> pd.DataFrame:
    rows = []
    for s in stop_grid:
        for t in take_grid:
            p = dict(params)
            p["stop_atr"] = float(s)
            p["take_atr"] = float(t)
            _, _, m = run_backtest_bundle(df, p)
            if "error" in m:
                continue
            rows.append(
                {
                    "stop_atr": float(s),
                    "take_atr": float(t),
                    "total_return": float(m["total_return"]),
                    "max_drawdown": float(m["max_drawdown"]),
                    "profit_factor": float(m["profit_factor"])
                    if np.isfinite(m.get("profit_factor", np.nan))
                    else np.nan,
                    "win_rate": float(m["win_rate"]) if np.isfinite(m.get("win_rate", np.nan)) else np.nan,
                    "num_trades": int(m["num_trades"]),
                }
            )
    return pd.DataFrame(rows)


# -----------------------------
# Charts
# -----------------------------


def plot_candles(df: pd.DataFrame, overlays: list[str] | None = None, title: str = ""):
    overlays = overlays or []

    if go is None:
        st.line_chart(df[["close"]].dropna())
        return

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
            )
        ]
    )

    for col in overlays:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_heatmap(df_sweep: pd.DataFrame, value_col: str, title: str):
    if df_sweep.empty:
        st.info("No sweep results.")
        return

    if go is None:
        st.dataframe(df_sweep)
        return

    pivot = df_sweep.pivot(index="stop_atr", columns="take_atr", values=value_col)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            colorbar=dict(title=value_col),
        )
    )
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# UI
# -----------------------------

st.title("Stock Analyser + Backtester")

mode_upload, mode_live = st.tabs(["Upload", "Live"])

with st.sidebar:
    st.header("Signal")
    use_adx_filter = st.checkbox("Use ADX filter (avoid chop)", value=True)
    adx_threshold = st.slider("ADX threshold", 5.0, 35.0, 18.0, 0.5)

    st.divider()
    st.header("Backtest")
    initial_cash = st.number_input("Initial cash", min_value=100.0, value=10_000.0, step=100.0)
    risk_per_trade = st.slider("Risk per trade (% equity)", 0.1, 5.0, 1.0, 0.1) / 100.0
    stop_atr = st.slider("Stop (ATR)", 0.5, 5.0, 2.0, 0.1)
    take_atr = st.slider("Take profit (ATR)", 0.5, 10.0, 3.0, 0.1)
    fee_bps = st.slider("Fees (bps/side)", 0.0, 50.0, 2.0, 0.5)
    slippage_bps = st.slider("Slippage (bps/side)", 0.0, 50.0, 1.0, 0.5)
    stop_extra_slip_bps = st.slider("Extra slip on STOP (bps)", 0.0, 100.0, 3.0, 1.0)
    fill_on_gap = st.checkbox("Gap-aware stop fills", value=True)
    allow_short = st.checkbox("Allow shorting", value=True)

    st.divider()
    st.header("Walk-forward")
    test_fraction = st.slider("Test fraction", 0.1, 0.6, 0.3, 0.05)

    st.divider()
    st.header("Extras")
    show_raw = st.checkbox("Show raw/cleaned data", value=False)
    show_table = st.checkbox("Show indicator table", value=False)


def run_all(df0: pd.DataFrame, title: str):
    if len(df0) < 80:
        st.warning("Short dataset (<80 rows). Stats may be unreliable.")

    df = add_indicators(df0)
    df = add_signal(df, use_adx_filter=use_adx_filter, adx_threshold=float(adx_threshold))
    df = df.dropna().copy()

    latest = df.iloc[-1]

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Overview")
        st.write(
            f"Price {float(latest['close']):.2f}. {latest['trend_state']}. "
            f"MACD {latest['macd_state']}. RSI {float(latest['rsi14']):.1f} ({latest['rsi_state']}). "
            f"ADX {float(latest['adx14']):.1f}. ATR {float(latest['atr_pct']):.2f}%."
        )
        st.metric("Signal", str(latest["action"]), f"Score {int(latest['score'])}")

        lookback = min(60, len(df))
        support = float(df["low"].tail(lookback).min())
        resistance = float(df["high"].tail(lookback).max())
        st.write({"Support (60 bars)": support, "Resistance (60 bars)": resistance})

    with right:
        st.subheader("Price chart")
        plot_candles(
            df,
            overlays=["ema20", "ema50", "ema200", "bb_up", "bb_mid", "bb_low"],
            title=title,
        )

    t_over, t_eq, t_wf, t_sweep, t_trades = st.tabs(["Backtest", "Equity", "Walk-forward", "Sweep", "Trades"])

    params = {
        "initial_cash": float(initial_cash),
        "risk_per_trade": float(risk_per_trade),
        "stop_atr": float(stop_atr),
        "take_atr": float(take_atr),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "stop_extra_slip_bps": float(stop_extra_slip_bps),
        "allow_short": bool(allow_short),
        "fill_on_gap": bool(fill_on_gap),
    }

    bt_df, trades_df, metrics = run_backtest_bundle(df, params)

    with t_over:
        st.subheader("Backtest summary")
        if "error" in metrics:
            st.error(metrics["error"])
        else:
            a, b, c1, d = st.columns(4)
            a.metric("Final equity", f"{metrics['final_equity']:.2f}", f"Return {metrics['total_return']*100:.1f}%")
            b.metric("Max drawdown", f"{metrics['max_drawdown']*100:.1f}%")
            c1.metric("Trades", f"{metrics['num_trades']}", f"Win {metrics['win_rate']*100:.1f}%" if np.isfinite(metrics["win_rate"]) else "Win n/a")
            d.metric("Profit factor", f"{metrics['profit_factor']:.2f}" if np.isfinite(metrics["profit_factor"]) else "n/a")

            st.write(
                {
                    "CAGR": metrics["cagr"],
                    "Annualized vol": metrics["annualized_vol"],
                    "Sharpe-like": metrics["sharpe_like"],
                    "Expectancy/trade": metrics["expectancy_per_trade"],
                    "Exposure (proxy)": metrics["exposure_pct"],
                }
            )

    with t_eq:
        st.subheader("Equity curve")
        if "equity" in bt_df.columns:
            st.line_chart(bt_df[["equity"]].dropna())
        st.subheader("Drawdown")
        if "drawdown" in bt_df.columns:
            st.line_chart(bt_df[["drawdown"]].dropna())

        with st.expander("Momentum"):
            st.line_chart(df[["rsi14", "macd_hist", "adx14"]].dropna())

    with t_wf:
        st.subheader("Walk-forward (out-of-sample)")
        train, test = split_train_test(df, test_fraction=float(test_fraction))
        if test.empty:
            st.info("Not enough data to split. Use more rows or lower the test fraction.")
        else:
            _, _, m_train = run_backtest_bundle(train, params)
            _, trades_test, m_test = run_backtest_bundle(test, params)

            colA, colB = st.columns(2)
            with colA:
                st.write("Train metrics")
                st.write({k: m_train.get(k) for k in ["total_return", "max_drawdown", "profit_factor", "win_rate", "num_trades"]})
            with colB:
                st.write("Test metrics (out-of-sample)")
                st.write({k: m_test.get(k) for k in ["total_return", "max_drawdown", "profit_factor", "win_rate", "num_trades"]})

            if trades_test is None or trades_test.empty:
                st.info("No test trades with current settings.")

    with t_sweep:
        st.subheader("Parameter sweep (stability check)")
        st.caption("Grid search on stop_atr × take_atr using full dataset. Use walk-forward for real validation.")

        stop_grid = st.multiselect(
            "Stop ATR grid",
            options=[0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
            default=[1.5, 2.0, 2.5, 3.0],
        )
        take_grid = st.multiselect(
            "Take ATR grid",
            options=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
            default=[2.5, 3.0, 3.5, 4.0],
        )

        if st.button("Run sweep"):
            if len(stop_grid) == 0 or len(take_grid) == 0:
                st.error("Pick at least one stop and one take value.")
            else:
                sweep_df = sweep_params(df, params, stop_grid=sorted(stop_grid), take_grid=sorted(take_grid))
                if sweep_df.empty:
                    st.info("No sweep results.")
                else:
                    st.dataframe(sweep_df.sort_values("total_return", ascending=False).head(20))
                    plot_heatmap(sweep_df, value_col="total_return", title="Total return heatmap")

    with t_trades:
        st.subheader("Trades")
        if trades_df is None or trades_df.empty:
            st.info("No trades with current settings.")
        else:
            st.dataframe(trades_df.sort_values("exit_time", ascending=False))

    if show_table:
        st.subheader("Indicators + signals (tail)")
        st.dataframe(df.tail(250))

    if show_raw:
        st.subheader("Raw/cleaned data (tail)")
        st.dataframe(df0.tail(250))


with mode_upload:
    st.subheader("Upload")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="upload_file")
    st.caption("Best: Date/Open/High/Low/Close/Volume. Minimum: Close.")

    if uploaded is None:
        st.info("Upload a CSV/XLSX to analyse + backtest.")
    else:
        try:
            df0 = load_any(uploaded)
            run_all(df0, title="Uploaded data")
        except Exception as e:
            st.error(f"Could not load/analyse file: {e}")

with mode_live:
    st.subheader("Live")
    if yf is None:
        st.warning("Live mode needs yfinance: pip install yfinance")

    ticker = st.text_input("Ticker", value="AAPL")
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=3)
    with col2:
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo", "1h", "30m", "15m", "5m"], index=0)
    with col3:
        auto_adjust = st.checkbox("Auto-adjust (splits/dividends)", value=True)

    st.caption("If it errors: try interval 1d + period 1y.")

    if st.button("Fetch + Analyse", type="primary"):
        try:
            df0 = load_live_cached(ticker=ticker, period=period, interval=interval, auto_adjust=auto_adjust)
            run_all(df0, title=f"{ticker.upper()} ({interval}, {period})")
        except Exception as e:
            st.error(str(e))
