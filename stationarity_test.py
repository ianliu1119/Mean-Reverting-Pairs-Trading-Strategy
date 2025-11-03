import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import plotly.graph_objs as go
from dateutil.relativedelta import relativedelta

start_date = "2023-10-01"
end_date = "2024-10-01"


def aspread_test(price_series):
    aspread_result = adfuller(price_series)
    p_value = aspread_result[1]
    return p_value < 0.05


def optimal_lag_selection(time_series):
    model = VAR(time_series)
    lag_order_results = model.select_order(maxlags=10)
    optimal_lag = lag_order_results.aic
    return optimal_lag - 1


def johansen_test_summary(data, start_date, end_date):
    time_series = yf.download(data, start=start_date, end=end_date, auto_adjust=True)[
        "Close"
    ].dropna()

    if aspread_test(time_series[data[0]]) or aspread_test(time_series[data[1]]):
        return "Not applicable for the Johansen Test", -1, -1

    johansen_result = coint_johansen(
        time_series, det_order=0, k_ar_diff=optimal_lag_selection(time_series)
    )

    test_stat_trace_r0 = johansen_result.lr1[0]
    critical_value_trace_r0 = johansen_result.cvt[0, 1]
    coint_vector = johansen_result.evec[:, 0]

    return (
        [
            float(test_stat_trace_r0),
            float(critical_value_trace_r0),
        ],
        time_series,
        coint_vector,
    )


asset_pairs = ["GLD", "SLV"]
result, merged_data, coint_vector = johansen_test_summary(
    asset_pairs, start_date, end_date
)
print(result)


def get_hedge_weights(pair_prices):
    coint_vectors = list(coint_johansen(pair_prices, 0, 1).evec[:, 0])
    return pd.Series(coint_vectors, index=pair_prices.columns)


def rolling_hedge_weights(pair_prices, lookback_window):
    all_hedge_ratios = []
    stat = 0

    for index in range(len(pair_prices)):
        start_index = index - lookback_window
        if start_index < 0:
            # not enough data yet
            all_hedge_ratios.append(
                pd.Series([None] * len(pair_prices.columns), index=pair_prices.columns)
            )
            continue

        some_closes = pair_prices.iloc[start_index:index]
        hedge_ratio = get_hedge_weights(some_closes)
        # hedge_ratio = pd.Series(hedge_ratio).to_frame().T
        all_hedge_ratios.append(hedge_ratio)
        stat += 1

    hedge_ratio = pd.DataFrame(all_hedge_ratios, index=pair_prices.index).dropna()
    return hedge_ratio.dropna()


hedge_weights = rolling_hedge_weights(merged_data, 20)
print(hedge_weights)


# trade by spread
def spread_bands(pair_prices, hedge_weights, lookback_window, num_std):
    spreads = (pair_prices * hedge_weights).sum(axis=1)
    means = spreads.rolling(lookback_window).mean()
    stds = spreads.rolling(lookback_window).std()
    upper_band = means + num_std * stds
    lower_band = means - num_std * stds

    return pd.DataFrame(
        {
            "ratio": spreads,
            "mean": means,
            "std": stds,
            "upper_band": upper_band,
            "lower_band": lower_band,
        }
    )


def calculate_ratios(pair_prices):
    if pair_prices.shape[1] != 2:
        raise ValueError("Ratio calculation requires exactly two assets.")
    ratio = pair_prices.iloc[:, 0] / pair_prices.iloc[:, 1]
    return ratio


def calculate_bollinger_bands(pair_prices, lookback_window, num_std):
    ratios = calculate_ratios(pair_prices)
    means = ratios.rolling(lookback_window).mean()
    stds = ratios.rolling(lookback_window).std()
    upper_band = means + num_std * stds
    lower_band = means - num_std * stds

    return pd.DataFrame(
        {
            "ratio": ratios,
            "mean": means,
            "std": stds,
            "upper_band": upper_band,
            "lower_band": lower_band,
        }
    )


def generate_signals(data):
    holding_state = "None"

    data["long_signal"] = False
    data["short_signal"] = False
    data["exit_signal"] = False

    for i in range(len(data)):
        ratio = data["ratio"].iloc[i]
        upper = data["upper_band"].iloc[i]
        lower = data["lower_band"].iloc[i]
        mean = data["mean"].iloc[i]

        if holding_state == "None":
            if ratio < lower:
                data.loc[data.index[i], "long_signal"] = True
                holding_state = "Long"
            elif ratio > upper:
                data.loc[data.index[i], "short_signal"] = True
                holding_state = "Short"

        elif holding_state == "Long":
            if ratio >= mean:
                data.loc[data.index[i], "exit_signal"] = True
                holding_state = "None"

        elif holding_state == "Short":
            if ratio <= mean:
                data.loc[data.index[i], "exit_signal"] = True
                holding_state = "None"
    return data


bolligner_band_ratios = calculate_bollinger_bands(merged_data, 20, 1)
bollinger_band_with_signals = generate_signals(bolligner_band_ratios)
print(bollinger_band_with_signals)

"""
def backtest_pairs_strategy(pair_prices, signals_data, hedge_weights):
    data = pair_prices.copy().join(
        signals_data[["long_signal", "short_signal", "exit_signal"]], how="inner"
    )
    data = data.join(hedge_weights, rsuffix="_weight")

    cash = 100000
    asset1 = 0
    asset2 = 0
    portfolio_time_series = []
    multiplier = 1000

    for i in range(20, len(data)):
        row = data.iloc[i]
        if row["long_signal"] == True:  # buy gold short silver
            cash -= (row["GLD"] * multiplier) * row["GLD_weight"]
            asset1 += row["GLD_weight"] * multiplier
            cash += (row["SLV"] * multiplier) * row["SLV_weight"]
            asset2 -= row["SLV_weight"] * multiplier
        elif row["short_signal"] == True:  # buy silver short gold
            cash -= (row["SLV"] * multiplier) * row["SLV_weight"]
            asset2 += row["SLV_weight"] * multiplier
            cash += (row["GLD"] * multiplier) * row["GLD_weight"]
            asset1 -= row["GLD_weight"] * multiplier
        elif row["exit_signal"] == True:
            cash += asset1 * row["GLD"] + asset2 * row["SLV"]
            asset1 = 0
            asset2 = 0
        portfolio_time_series.append(
            int(cash + asset1 * row["GLD"] + asset2 * row["SLV"])
        )
    correct_index = data.index[20:]
    portfolio_series = pd.Series(portfolio_time_series, index=correct_index)
    return portfolio_series
"""


def backtest_pairs_strategy(
    pair_prices, signals_data, hedge_weights, initial_cash=100000, trade_size=10000
):
    data = pair_prices.copy().join(
        signals_data[["long_signal", "short_signal", "exit_signal"]], how="inner"
    )
    data = data.join(hedge_weights, rsuffix="_weight")

    cash = initial_cash
    asset1 = 0.0
    asset2 = 0.0
    holding_state = "None"
    portfolio_time_series = []

    for i in range(20, len(data)):
        row = data.iloc[i]

        # Normalize hedge weights to total 1× exposure
        gross = abs(row["GLD_weight"]) + abs(row["SLV_weight"])
        w1 = row["GLD_weight"] / gross
        w2 = row["SLV_weight"] / gross

        if holding_state == "None":
            if row["long_signal"]:
                # Long spread = long GLD, short SLV
                cash -= trade_size * w1
                asset1 += trade_size * w1 / row["GLD"]
                cash += trade_size * w2  # since w2 < 0, this adds cash
                asset2 += trade_size * w2 / row["SLV"]
                holding_state = "Long"

            elif row["short_signal"]:
                # Short spread = short GLD, long SLV
                cash += trade_size * w1
                asset1 += -trade_size * w1 / row["GLD"]
                cash -= trade_size * w2
                asset2 += -trade_size * w2 / row["SLV"]
                holding_state = "Short"

        elif holding_state != "None" and row["exit_signal"]:
            # Close positions
            cash += asset1 * row["GLD"] + asset2 * row["SLV"]
            asset1 = 0.0
            asset2 = 0.0
            holding_state = "None"

        # Track daily portfolio value
        portfolio_value = cash + asset1 * row["GLD"] + asset2 * row["SLV"]
        portfolio_time_series.append(portfolio_value)

    correct_index = data.index[20:]
    return pd.Series(portfolio_time_series, index=correct_index)


final = backtest_pairs_strategy(merged_data, bollinger_band_with_signals, hedge_weights)
print(final)


def plot_cum_return(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Strategy Return")
    plt.title("Pairs Trading Strategy Backtest with Hedge Weights")
    plt.legend()
    plt.show()


plot_cum_return(final)


def plot_bollinger_bands_ratio(data):
    plt.figure(figsize=(12, 6))

    # Plot ratio and Bollinger Bands
    plt.plot(data["ratio"], label="Ratio", color="blue", linewidth=1.2)
    plt.plot(data["mean"], label="Mean", color="black", linewidth=1.1)
    plt.plot(data["upper_band"], "--", label="Upper Band", color="red", alpha=0.7)
    plt.plot(data["lower_band"], "--", label="Lower Band", color="green", alpha=0.7)

    # Long entry points
    plt.scatter(
        data.index[data["long_signal"]],
        data["ratio"][data["long_signal"]],
        color="green",
        label="Long Entry",
        marker="^",
        s=100,
        alpha=0.8,
    )

    # Short entry points
    plt.scatter(
        data.index[data["short_signal"]],
        data["ratio"][data["short_signal"]],
        color="red",
        label="Short Entry",
        marker="v",
        s=100,
        alpha=0.8,
    )

    # Exit points (optional)
    plt.scatter(
        data.index[data["exit_signal"]],
        data["ratio"][data["exit_signal"]],
        color="orange",
        label="Exit",
        marker="x",
        s=80,
        alpha=0.8,
    )

    plt.title("Ratio-Based Bollinger Bands with Entry/Exit Signals")
    plt.xlabel("Date")
    plt.ylabel("Price Ratio")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


plot_bollinger_bands_ratio(bolligner_band_ratios)


def plot_cointegration(asset_pairs, data, coint_vector):
    spread = data.dot(coint_vector)
    spread.name = "Stationary Spread"
    plt.figure(figsize=(12, 6))
    spread.plot(
        title=f"Stationary Cointegrating Spread ({asset_pairs[0]} & {asset_pairs[1]})",
        legend=True,
    )
    mean = spread.mean()
    std = spread.std()
    plt.axhline(mean, color="red", linestyle="--", label="Mean")
    plt.axhline(mean + 2 * std, color="green", linestyle=":", label="+2 Std Dev")
    plt.axhline(mean - 2 * std, color="green", linestyle=":", label="-2 Std Dev")

    plt.legend(loc="upper left")
    plt.ylabel("Spread Value")
    plt.grid(True, alpha=0.5)
    plt.show()


def compute_time_range(end_date, asset_pairs):
    new_start_date = end_date
    dt_end_date = datetime.strptime(end_date, "%Y-%m-%d")
    new_dt_end_date = dt_end_date + relativedelta(years=3)
    new_end_date_str = new_dt_end_date.strftime("%Y-%m-%d")
    time_series = yf.download(
        asset_pairs, start=new_start_date, end=new_end_date_str, auto_adjust=True
    )["Close"].dropna()
    return time_series


def compute_spread(time_series, coint_vector):
    spread = time_series.dot(coint_vector)
    spread_df = pd.DataFrame(spread, columns=["Spread"])
    return spread_df


def plot_ratio(data, asset):
    ratio = data[asset[0]] / data[asset[1]]
    plt.figure(figsize=(12, 6))
    ratio.plot(
        title=f"Ratio between ({asset_pairs[0]} & {asset_pairs[1]})", legend=True
    )
    plt.legend()
    plt.ylabel("Ratio Value")
    plt.grid(True, alpha=0.5)
    plt.show()


def plot_bollinger(data):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=data.index, y=data["Spread"], mode="lines", name="Price")
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Mean"],
            mode="lines",
            name="Middle Bollinger Band",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Upper"],
            mode="lines",
            name="Upper Bollinger Band",
            line=dict(color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Lower"],
            mode="lines",
            name="Lower Bollinger Band",
            line=dict(color="green"),
        )
    )

    # --- Identify touch points ---
    upper_touch = data["Spread"] >= data["Upper"]
    lower_touch = data["Spread"] <= data["Lower"]
    # tolerance = 0.08 * data["Spread"].std()
    # middle_touch = (data["Spread"] - data["Mean"]).abs() <= tolerance

    upper_points = data[upper_touch]
    lower_points = data[lower_touch]
    # middle_points = data[middle_touch]

    is_above = data["Spread"] > data["Mean"]
    was_above = is_above.shift(1)
    middle_crossing = is_above != was_above
    middle_crossing.iloc[0] = False
    middle_points = data[middle_crossing]

    # --- Add markers for touches ---
    fig.add_trace(
        go.Scatter(
            x=middle_points.index,
            y=middle_points["Spread"],
            mode="markers",
            name="Cross Middle Band",
            marker=dict(color="blue", size=10, symbol="star"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=upper_points.index,
            y=upper_points["Spread"],
            mode="markers",
            name="Touch Upper Band",
            marker=dict(color="red", size=10, symbol="star"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lower_points.index,
            y=lower_points["Spread"],
            mode="markers",
            name="Touch Lower Band",
            marker=dict(color="green", size=10, symbol="star"),
        )
    )

    # --- Layout ---
    fig.update_layout(
        title="Pair Trading Spread with Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Spread Value",
        showlegend=True,
        template="plotly_white",
    )

    fig.show()


"""
asset_pairs = ["GLD", "SLV"]
result, merged_data, coint_vector = johansen_test_summary(
    asset_pairs, start_date, end_date
)
print(result)
print(coint_vector)

time_series = compute_time_range(end_date, asset_pairs)
test_spread = compute_spread(time_series, coint_vector)

spread1 = generate_bollinger_signals(test_spread, 2, 10)
print(spread1.head(50))
"""
