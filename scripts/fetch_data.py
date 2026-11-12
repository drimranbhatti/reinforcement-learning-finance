"""Refresh the price snapshots in data/ from Yahoo Finance (adjusted closes)."""
import pathlib
import yfinance as yf

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
TRADE = ["SPY", "QQQ", "TLT", "GLD", "EEM"]


def get(tickers, start):
    px = yf.download(tickers, start=start, end="2026-11-01", auto_adjust=True, progress=False)["Close"]
    return px.dropna(how="all").round(4)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    get(SECTORS + ["SPY"], "1999-01-01").to_csv(DATA / "sector_etfs.csv")
    get(TRADE, "2005-01-01").to_csv(DATA / "etfs_daily.csv")
    print("saved")
