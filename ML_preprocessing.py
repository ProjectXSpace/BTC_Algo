import ccxt
import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional, Dict


def fetch_crypto_data(symbol: str, timeframe: str, start_date: str) -> pd.DataFrame:
    """
    Fetches historical cryptocurrency data from Binance using the ccxt library.

    :param symbol: The symbol for the cryptocurrency pair (e.g., 'BTC/USDT').
    :param timeframe: Timeframe for the data (e.g., '1d' for daily).
    :param start_date: Start date for data in 'YYYY-MM-DD' format.
    :return: Pandas DataFrame containing the OHLCV data.
    """
    exchange = ccxt.binance()
    limit = 1000
    since = exchange.parse8601(f"{start_date}T00:00:00Z")

    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if len(ohlcv) == 0:
                break

            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + exchange.parse_timeframe(timeframe) * 60 * 1000
        except ccxt.NetworkError as e:
            print(f"Network error: {e}")
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}")
            break
        except ccxt.BaseError as e:
            print(f"CCXT base error: {e}")
            break

    symbol_modified = symbol.replace("/", ":")
    df = pd.DataFrame(
        all_ohlcv,
        columns=[
            f"{symbol_modified}_timestamp",
            f"{symbol_modified}_open",
            f"{symbol_modified}_high",
            f"{symbol_modified}_low",
            f"{symbol_modified}_close",
            f"{symbol_modified}_volume",
        ],
    )

    df[f"{symbol_modified}_timestamp"] = pd.to_datetime(
        df[f"{symbol_modified}_timestamp"], unit="ms"
    )
    df.set_index(f"{symbol_modified}_timestamp", inplace=True)

    return df


def add_target(df: pd.DataFrame, symbol: str, day_to_forecast: int) -> pd.DataFrame:
    """
    Adds a target column to the DataFrame for forecasting.

    :param df: Pandas DataFrame containing the historical data.
    :param symbol: The symbol for the cryptocurrency pair.
    :param day_to_forecast: Number of days to forecast.
    :return: DataFrame with the target column added.
    """
    symbol = symbol.replace("/", ":")
    days_to_shift = day_to_forecast * 24
    df[f"{symbol}_target"] = (
        df[f"{symbol}_close"].pct_change(periods=days_to_shift).shift(-days_to_shift)
    )
    return df


def get_features_and_target(
    symbol: str, feature_lags: List[int] = [3, 9, 16], day_to_forecast: int = 7
) -> pd.DataFrame:
    """
    Generates features and target variable for the given cryptocurrency symbol.

    :param symbol: The symbol for the cryptocurrency pair.
    :param feature_lags: List of integers representing the lags for feature generation.
    :param day_to_forecast: Number of days to forecast.
    :return: DataFrame with features and target.
    """
    symbol = symbol.replace("/", ":")
    features_df = pd.read_csv(
        f"{symbol}_price_data.csv", parse_dates=True, index_col=f"{symbol}_timestamp"
    )

    required_columns = [
        f"{symbol}_close",
        f"{symbol}_high",
        f"{symbol}_low",
        f"{symbol}_volume",
    ]
    if not all(col in features_df.columns for col in required_columns):
        raise ValueError("Required columns are missing in the DataFrame")

    # Moving Averages
    for ma in [9, 20, 50, 200]:
        features_df[f"{symbol}_sma_{ma}"] = talib.SMA(
            features_df[f"{symbol}_close"], timeperiod=ma
        )

    # RSI
    features_df[f"{symbol}_rsi"] = talib.RSI(
        features_df[f"{symbol}_close"], timeperiod=14
    )

    # Bollinger Bands
    upperband, middleband, lowerband = talib.BBANDS(
        features_df[f"{symbol}_close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    features_df[f"{symbol}_bollinger_up"] = upperband
    features_df[f"{symbol}_bollinger_down"] = lowerband

    # ADX
    features_df[f"{symbol}_adx"] = talib.ADX(
        features_df[f"{symbol}_high"],
        features_df[f"{symbol}_low"],
        features_df[f"{symbol}_close"],
        timeperiod=14,
    )

    # MACD
    macd, macdsignal, macdhist = talib.MACD(
        features_df[f"{symbol}_close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    features_df[f"{symbol}_macd_diff"] = macd - macdsignal

    # OBV
    features_df[f"{symbol}_obv"] = talib.OBV(
        features_df[f"{symbol}_close"], features_df[f"{symbol}_volume"]
    )

    # Ichimoku Cloud
    nine_period_high = features_df[f"{symbol}_high"].rolling(window=9).max()
    nine_period_low = features_df[f"{symbol}_low"].rolling(window=9).min()
    features_df[f"{symbol}_ichimoku_conversion"] = (
        nine_period_high + nine_period_low
    ) / 2

    # Stochastic Oscillator
    stochastic_k, stochastic_d = talib.STOCH(
        features_df[f"{symbol}_high"],
        features_df[f"{symbol}_low"],
        features_df[f"{symbol}_close"],
    )
    features_df[f"{symbol}_stochastic_k"] = stochastic_k
    features_df[f"{symbol}_stochastic_d"] = stochastic_d

    # Aroon Indicator
    aroon_up, aroon_down = talib.AROON(
        features_df[f"{symbol}_high"], features_df[f"{symbol}_low"], timeperiod=14
    )
    features_df[f"{symbol}_aroon_up"] = aroon_up
    features_df[f"{symbol}_aroon_down"] = aroon_down

    # Lagged metrics and deltas calculation
    for lag in feature_lags:
        features_df[f"{symbol}_rsi_lag_{lag}"] = features_df[f"{symbol}_rsi"].shift(lag)
        features_df[f"{symbol}_macd_diff_lag_{lag}"] = features_df[
            f"{symbol}_macd_diff"
        ].shift(lag)
        features_df[f"{symbol}_obv_lag_{lag}"] = features_df[f"{symbol}_obv"].shift(lag)
        features_df[f"{symbol}_ichimoku_conversion_lag_{lag}"] = features_df[
            f"{symbol}_ichimoku_conversion"
        ].shift(lag)
        features_df[f"{symbol}_stochastic_k_lag_{lag}"] = stochastic_k.shift(lag)
        features_df[f"{symbol}_stochastic_d_lag_{lag}"] = stochastic_d.shift(lag)
        features_df[f"{symbol}_aroon_up_lag_{lag}"] = aroon_up.shift(lag)
        features_df[f"{symbol}_aroon_down_lag_{lag}"] = aroon_down.shift(lag)

        features_df[f"{symbol}_rsi_delta_{lag}"] = features_df[f"{symbol}_rsi"].diff(
            lag
        )
        features_df[f"{symbol}_macd_diff_delta_{lag}"] = features_df[
            f"{symbol}_macd_diff"
        ].diff(lag)
        features_df[f"{symbol}_obv_delta_{lag}"] = features_df[f"{symbol}_obv"].diff(
            lag
        )
        features_df[f"{symbol}_ichimoku_conversion_delta_{lag}"] = features_df[
            f"{symbol}_ichimoku_conversion"
        ].diff(lag)
        features_df[f"{symbol}_stochastic_k_delta_{lag}"] = stochastic_k.diff(lag)
        features_df[f"{symbol}_stochastic_d_delta_{lag}"] = stochastic_d.diff(lag)
        features_df[f"{symbol}_aroon_up_delta_{lag}"] = aroon_up.diff(lag)
        features_df[f"{symbol}_aroon_down_delta_{lag}"] = aroon_down.diff(lag)

    # Add target and handle missing values
    features_df = add_target(features_df, symbol, day_to_forecast)
    features_df.drop(
        columns=[
            f"{symbol}_open",
            f"{symbol}_high",
            f"{symbol}_low",
            f"{symbol}_close",
            f"{symbol}_volume",
        ],
        inplace=True,
    )
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.dropna(inplace=True)

    return features_df


def prepare_data_for_ML(
    symbol: str,
    feature_lags: List[int] = [3, 9, 16],
    day_to_forecast: int = 7,
    random_state = 99,
    fetch_data_params: Optional[Dict[str, any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepares data for machine learning by generating features and splitting into train and test sets.

    :param symbol: The symbol for the cryptocurrency pair.
    :param feature_lags: List of integers representing the lags for feature generation.
    :param day_to_forecast: Number of days to forecast.
    :param fetch_data_params: Optional dictionary of parameters to pass to the fetch_crypto_data function.
    :return: Tuple of (X_train, X_test, y_train, y_test).
    """
    symbol_modified = symbol.replace("/", ":")

    # Fetch new data if fetch_data_params is provided
    if fetch_data_params is not None:
        fetch_crypto_data(symbol, **fetch_data_params)

    df = get_features_and_target(symbol_modified, feature_lags, day_to_forecast)
    X = df.drop(columns=f"{symbol_modified}_target")
    y = df[f"{symbol_modified}_target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state
    )

    return (X_train, X_test, y_train, y_test)


# Example usage:
# params = {'timeframe': '1h', 'start_date': '2015-01-01'}
# X_train, X_test, y_train, y_test = prepare_data_for_ML("BTC/USDT", fetch_data_params=params)
