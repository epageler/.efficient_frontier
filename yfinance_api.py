# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:50:42 2024

@author: evan_
"""
from typing import Tuple
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf  # type: ignore
import streamlit as st
# from pprint import pprint


# ---------------------------------------------------------------------------- #
def get_investment_names(tickers: list[str]) -> Tuple[str, pd.DataFrame]:
    """
    Retrieve investment names for list of tickers.

    If invalid ticker is in list, err contains an error message with the invalid ticker and the returned df is empty.

    Args:
        tickers (list[str]): list of tickers

    Returns:
        str:
            If an invalid ticker is included in list, message specify
            invalid ticker. Empty string if no errors.
        pd.DataFrame:
            Column Heading(s): shortName,
            Index: ticker
            df Contents: Short Name for each investment. Empty df if errors.
    """
    err = ""
    investment_names = pd.DataFrame()
    for t in tickers:
        try:
            investment_names.loc[t, "longName"] = yf.Ticker(t).info["longName"]
        except:
            err = f"Invalid Ticker: {t}"
            investment_names = pd.DataFrame()  # return empty df if error
            return err, investment_names  # return immediately if error
    return err, investment_names


def get_max_inception_date(tickers: list[str]) -> datetime:
    """Returns max inception date from a list of investments

    Args:
        tickers (list[str]): list of investment tickers

    Returns:
        datetime: maximum inception date in Epoch UTC
    """
    max_inception_date: datetime = 0
    for t in tickers:
        d = yf.Ticker(t).info["firstTradeDateMilliseconds"]/1000
        if d > max_inception_date:
            max_inception_date = d
    return max_inception_date


def get_names_and_inceptions(tickers: list[str]) -> Tuple[str, pd.DataFrame]:
    """
    Retrieve investment names and inception dates for list of tickers.

    If invalid ticker is in list, err contains an error message with the invalid ticker and the returned df is empty.

    Args:
        tickers (list[str]): list of tickers

    Returns:
        str:
            If an invalid ticker is included in list, message specify
            invalid ticker. Empty string if no errors.
        pd.DataFrame:
            Column Heading(s): Name, Inception
            Index: ticker
            df Contents:
                Long Name for each investment. Inception Date for each investment as datetime object.
                Empty df if errors.
    """
    err = ""
    names_inception = pd.DataFrame()
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            names_inception.loc[t, "Name"] = info["longName"]
            d = info["firstTradeDateMilliseconds"]
            names_inception.loc[t, "Inception"] = d/1000
        except:
            err = f"Invalid Ticker: {t}"
            investment_names = pd.DataFrame()  # return empty df if error
            return err, investment_names  # return immediately if error
    if err == "":
        names_inception["Inception"] = pd.to_datetime(
            names_inception["Inception"], unit="s", utc=True
        )
    return err, names_inception


# ---------------------------------------------------------------------------- #
def get_adj_daily_close(
    tickers: list[str], start: str | datetime, end: str | datetime
) -> pd.DataFrame:
    """
    Retrieve adjusted daily closing prices for a list of tickers over a specified
    date range. Order of columns in returned df is same as order in tickers argument.

    Note: Assumes that the tickers are valid. So, recommendation is call the get_investment_names
    function first. That function includes an error if any entry in tickers is invalid.

    Args:
        tickers (list[str]): List of tickers to be retrieved
        start_date (str): Start date in format YYYY-MM-DD
        end_date (str): End date in format YYYY-MM-DD

    Returns:
        pd.DataFrame:
            Column Heading(s): tickers
            Index: Date
            df Contents: Adjusted daily closing prices
    """
    # Retrieve daily
    adj = yf.download(tickers, start=start, end=end, interval="1d",auto_adjust=False)["Adj Close"][tickers]
    return adj


def get_previous_close(ticker: str) -> float:
    """_summary_

    Args:
        ticker (str): Ticker of investment

    Returns:
        float: previous day closing price
    """
    return yf.Ticker(ticker).info["previousClose"]


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    tickers = ["BIL", "AGG", "TIP", "MUB", "PFF", "IVV", "IWM", "EFA", "EEM", "IYR"]
    # start: str = "2023-05-30"
    # end: str = "2024-05-30"

    # err, names = get_investment_names(tickers)
    # if err != "":
    #     print(err)
    # else:
    #     adj_daily_close = get_adj_daily_close(tickers, start, end)
    #     print(adj_daily_close.head(10))
    # -------------------------------------

    d=get_max_inception_date(tickers)
    print(d)
    t=datetime.fromtimestamp(d)
    print(t.strftime('%m/%d/%Y'))

    # --------------------------------------

    err, df = get_names_and_inceptions(tickers)
    print(f"{err=}")
    print(df.head(20))
    # df["Inception"] = pd.to_datetime(df["Inception"], unit="s", utc=True).dt.strftime(
    #     "%Y-%m-%d"
    # )
    # df["Inception"] = pd.to_datetime(df["Inception"], unit="s", utc=True)
    # print(df.info())
    # print(df)
    # --------------------------------------

    # Get latest yield on 10-year Treasury
    # end_date = datetime.today()
    # start_date = datetime.today()-timedelta(days=7)
    # tnx = yf.download(["^TNX"], start=start_date, end=end_date)
    # print(tnx)
    # rf_rate=tnx.loc[:,"Close"].iloc[-1]
    # print(f"rf_rate: {rf_rate:.2f}%")
    # --------------------------------------

    # Get last price of Ticker
    # last_price: float= yf.Ticker('^TNX').info['previousClose']
    # pprint(f"last_price: {last_price}")
    # --------------------------------------

    # Get yield of 10-year Treasury
    # print(f"10-year Treasury Current Yield: {get_previous_close('^TNX'):2f}%")