# check.py
# 功能：每月第一個交易日產生操作建議，以下一個月第一個交易日股價驗證正確與否，計算勝率並輸出報表
# 依賴：pip install yfinance pandas numpy matplotlib

import os
import math
import argparse
from functools import lru_cache
from typing import List, Dict, Tuple, Any

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ================== 基本設定 ==================
DEFAULT_TICKERS = [
    # 美股七大龍頭
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # 台灣龍頭
    "2330.TW", "2317.TW", "2454.TW", "2412.TW", "2881.TW","2327.TW", "2308.TW", "3293.TW"
]

CHINESE_NAME_MAP = {
    "AAPL": "蘋果",
    "MSFT": "微軟",
    "GOOGL": "Alphabet(谷歌)",
    "AMZN": "亞馬遜",
    "NVDA": "輝達",
    "META": "Meta(臉書)",
    "TSLA": "特斯拉",
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2454.TW": "聯發科",
    "2412.TW": "中華電",
    "2881.TW": "富邦金",
    "2327.TW": "國巨",
    "2308.TW": "台達電",
    "3293.TW": "鈊象"
}

# 預設邏輯門檻（可用參數覆蓋）
BUY_IF_MONTHLY_CHANGE_LEQ = -0.03   # 本月相對上月跌幅 ≤ -3% → 買進
SELL_IF_MONTHLY_CHANGE_GEQ =  0.07  # 本月相對上月漲幅 ≥ +7% → 賣出
HOLD_BAND_FOR_CORRECTNESS  =  0.05  # 驗證持有正確的容忍度 ±5%

# ============== 新增：估值 + 與 app.py 一致的操作建議邏輯 ==============
def classify_type(info_full: Dict[str, Any]):
    sector = safe_get(info_full, "sector", "")
    industry = safe_get(info_full, "industry", "")
    revenue_growth = safe_get(info_full, "revenueGrowth", None)
    earnings_growth = safe_get(info_full, "earningsGrowth", None)
    profit_margin = safe_get(info_full, "profitMargins", None)
    dividend_yield_raw = safe_get(info_full, "dividendYield", None)
    # Yahoo Finance API 的 dividendYield 返回的是百分比數值，需要除以 100
    dividend_yield = dividend_yield_raw / 100 if dividend_yield_raw is not None else None
    beta = safe_get(info_full, "beta", None)
    it = (industry or '').lower(); sec = (sector or '').lower()
    
    # 判斷是否為成長導向產業
    growth_sectors = ["technology", "consumer cyclical"]
    growth_keywords = ["software", "semiconductor", "auto manufacturers", "electric", "ai", "cloud"]
    is_growth_industry = (sec in growth_sectors) or any(k in it for k in growth_keywords)
    
    if ("real estate" in sec) or any(k in it for k in ["reit","real estate","property"]):
        stock_type = "資產股"
    elif dividend_yield is not None and dividend_yield >= 0.04 and (revenue_growth or 0) < 0.10 and (beta is None or beta < 0.9):
        stock_type = "高股息/定存股"
    elif profit_margin is not None and profit_margin < 0:
        stock_type = "虧損轉機股"
    elif any(k in it for k in ["steel","iron","chem","chemical","marine","shipping","freight","aluminum","metals","mining"]) or sec in ["materials","energy"]:
        stock_type = "景氣循環股"
    # 成長股判斷：1) 成長率達標 OR 2) 屬於成長產業且營收成長>0 且 Beta>1.3（高波動）
    elif (revenue_growth is not None and revenue_growth >= 0.15) or (earnings_growth is not None and earnings_growth >= 0.15) or \
         (is_growth_industry and (revenue_growth or 0) > 0 and beta is not None and beta > 1.3):
        stock_type = "成長股"
    else:
        stock_type = "穩定獲利股"
    return stock_type

def compute_basic_valuations(info_full: Dict[str, Any], price: float) -> Dict[str, Any]:
    trailing_pe = safe_get(info_full, "trailingPE", None)
    forward_eps = safe_get(info_full, "forwardEps", None)
    book_value = safe_get(info_full, "bookValue", None)
    rev_ps = safe_get(info_full, "revenuePerShare", None)
    profit_margin = safe_get(info_full, "profitMargins", None)
    earnings_growth = safe_get(info_full, "earningsGrowth", None)
    revenue_growth = safe_get(info_full, "revenueGrowth", None)
    div_rate = safe_get(info_full, "trailingAnnualDividendRate", None)
    roe = safe_get(info_full, "returnOnEquity", None)

    eps_ttm = None
    if price and trailing_pe and trailing_pe > 0:
        eps_ttm = price / trailing_pe
    elif forward_eps:
        eps_ttm = forward_eps

    def apply_growth(mult):
        g = earnings_growth if earnings_growth is not None else revenue_growth
        if g is None:
            return mult
        g_used = max(min(g, 0.60), -0.40)
        f = 1 + g_used
        return tuple(m * f for m in mult)

    def apply_roe(mult):
        if roe is None: return mult
        if roe > 0.15: return tuple(m*1.2 for m in mult)
        if roe < 0.08: return tuple(m*0.8 for m in mult)
        return mult

    def apply_margin(mult):
        if profit_margin is None: return mult
        if profit_margin > 0.20: return tuple(m*1.3 for m in mult)
        if 0.10 <= profit_margin <= 0.20: return tuple(m*1.1 for m in mult)
        if profit_margin < 0.05: return tuple(m*0.7 for m in mult)
        return mult

    stock_type = classify_type(info_full)
    cheap=fair=expensive=None
    try:
        if stock_type == "資產股":
            mult=(0.6,1.0,1.5); mult=apply_roe(mult)
            if book_value:
                cheap,fair,expensive=(book_value*mult[0],book_value*mult[1],book_value*mult[2])
        elif stock_type=="高股息/定存股":
            if div_rate and div_rate>0:
                cheap,fair,expensive=(div_rate/0.06, div_rate/0.05, div_rate/0.04)
            elif eps_ttm:
                base=(12,18,25); cheap,fair,expensive=(eps_ttm*base[0],eps_ttm*base[1],eps_ttm*base[2])
        elif stock_type=="虧損轉機股":
            mult=(1.0,1.5,2.5); mult=apply_margin(mult)
            if rev_ps:
                cheap,fair,expensive=(rev_ps*mult[0],rev_ps*mult[1],rev_ps*mult[2])
            elif eps_ttm:
                base=(10,15,20); cheap,fair,expensive=(eps_ttm*base[0],eps_ttm*base[1],eps_ttm*base[2])
        elif stock_type=="成長股":
            if eps_ttm:
                base=(18,28,40); base=apply_growth(base)
                cheap,fair,expensive=(eps_ttm*base[0],eps_ttm*base[1],eps_ttm*base[2])
            elif rev_ps:
                mult=(1.5,3.0,5.0); mult=apply_margin(mult)
                cheap,fair,expensive=(rev_ps*mult[0],rev_ps*mult[1],rev_ps*mult[2])
        elif stock_type=="景氣循環股":
            base=(8,15,22)
            if eps_ttm:
                cheap,fair,expensive=(eps_ttm*base[0],eps_ttm*base[1],eps_ttm*base[2])
            elif rev_ps:
                mult=(0.5,1.0,2.0); cheap,fair,expensive=(rev_ps*mult[0],rev_ps*mult[1],rev_ps*mult[2])
        else: # 穩定獲利股
            if eps_ttm:
                base=(12,18,25); cheap,fair,expensive=(eps_ttm*base[0],eps_ttm*base[1],eps_ttm*base[2])
            elif book_value:
                mult=(1.5,2.5,3.5); mult=apply_roe(mult)
                cheap,fair,expensive=(book_value*mult[0],book_value*mult[1],book_value*mult[2])
    except Exception:
        cheap=fair=expensive=None
    return {"stock_type":stock_type,"cheap":cheap,"fair":fair,"expensive":expensive}

def suggest_action(price, cheap, fair, expensive):
    if not price or not cheap or not fair or not expensive:
        return "資料不足"
    try:
        if price <= cheap: return "分批買進"
        elif price <= 0.95 * fair: return "買進"
        elif price <= expensive: return "持有/減碼"
        else: return "賣出/減碼"
    except Exception:
        return "—"

# ================== 常用工具 ==================
def safe_get(d: Dict, key: str, default=None):
    try:
        v = d.get(key, default)
        if v is None:
            return default
        return v
    except Exception:
        return default

def pct_to_str(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return "N/A"

def month_starts(start: str, end: str) -> List[pd.Timestamp]:
    """
    產生 [start, end] 區間內每個月的第一天 Timestamp (month start)。
    start/end 可為 'YYYY-MM' 或 'YYYY-MM-DD'
    """
    s = pd.to_datetime(start).to_period("M").to_timestamp()  # 月初
    e = pd.to_datetime(end).to_period("M").to_timestamp()    # 月初
    return list(pd.date_range(s, e, freq="MS"))

@lru_cache(maxsize=512)
def fetch_monthly_first_close(symbol: str, start: str, end: str) -> pd.Series:
    """
    取每月第一個「交易日」的收盤價（月初價）
    - 自動處理遇假日的情況（往後第一個交易日）
    - 回傳 Series，index 為該月第一天（MS），value 為該月第一個交易日的 Close
    """
    s = pd.to_datetime(start).to_period("M").to_timestamp()
    e = pd.to_datetime(end).to_period("M").to_timestamp()
    # 兩側多抓幾天以涵蓋假期
    hist = yf.Ticker(symbol).history(
        start=s - pd.Timedelta(days=7),
        end=e + pd.offsets.MonthEnd(1) + pd.Timedelta(days=7),
        interval="1d",
        auto_adjust=False
    )
    if hist is None or hist.empty:
        return pd.Series(dtype=float)

    idx = hist.index.tz_localize(None) if getattr(hist.index, "tz", None) else hist.index
    df = hist.copy()
    df.index = idx
    df["Month"] = df.index.to_period("M")
    # 每個月的第一筆（已按時間順序）
    first = df.groupby("Month").head(1)
    srs = first["Close"].copy()
    # index 對齊成每月第一天（MS）
    srs.index = srs.index.to_period("M").to_timestamp()
    # 範圍裁剪
    srs = srs[(srs.index >= s) & (srs.index <= e)]
    srs.name = symbol
    return srs

def compute_recommendation_on_date(cur_price: float, cheap: float, fair: float, expensive: float) -> str:
    """用估值帶生成操作建議（與 app.py 一致）。"""
    return suggest_action(cur_price, cheap, fair, expensive)

def backtest_recommendations(
    tickers: List[str],
    start: str,
    end: str,
    hold_band: float,
    buy_thr: float,  # 兼容舊參數，已不使用
    sell_thr: float  # 兼容舊參數，已不使用
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """使用估值帶操作建議的回測：建議 = 分批買進 / 買進 / 持有/減碼 / 賣出/減碼。
    評分準則：
      - 分批買進 / 買進：下月月初價上漲 → 正確
      - 賣出/減碼：下月月初價下跌 → 正確
      - 持有/減碼：|下月報酬| ≤ hold_band → 正確
    注意：估值帶使用當前快照，存在前視偏差，只用於檢驗規則表現粗略方向。
    """
    months = month_starts(start, end)
    if len(months) < 2:
        raise ValueError("回測區間至少需包含兩個月")

    # 取得估值帶 (一次)
    valuation_map: Dict[str, Dict[str, Any]] = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).get_info()
            if not isinstance(info, dict): info = {}
        except Exception:
            info = {}
        # 近一日現價
        try:
            hcur = yf.Ticker(tk).history(period="1d")
            last_price = float(hcur["Close"].iloc[-1]) if not hcur.empty else None
        except Exception:
            last_price = None
        valuation_map[tk] = compute_basic_valuations(info, last_price)

    all_rows = []
    for ticker in tickers:
        mpx = fetch_monthly_first_close(ticker, start, end)
        if mpx is None or mpx.empty:
            continue
        px = mpx.reindex(months).dropna()
        if len(px) < 2:
            continue
        cheap = valuation_map[ticker]["cheap"]
        fair = valuation_map[ticker]["fair"]
        expensive = valuation_map[ticker]["expensive"]
        for i in range(len(px)-1):
            cur_dt = px.index[i]
            cur_price = float(px.iloc[i])
            nxt_price = float(px.iloc[i+1])
            rec = compute_recommendation_on_date(cur_price, cheap, fair, expensive)
            pct = (nxt_price - cur_price) / cur_price
            if rec in ["分批買進","買進"]:
                is_correct = pct > 0
            elif rec == "賣出/減碼":
                is_correct = pct < 0
            elif rec == "持有/減碼":
                is_correct = abs(pct) <= hold_band
            else:
                is_correct = False
            all_rows.append({
                "date": cur_dt,
                "month": cur_dt.strftime("%Y-%m"),
                "ticker": ticker,
                "chinese_name": CHINESE_NAME_MAP.get(ticker, ticker),
                "current_price": cur_price,
                "next_price": nxt_price,
                "price_change_pct": pct * 100,
                "recommendation": rec,
                "is_correct": is_correct
            })

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df, {}, pd.DataFrame()
    monthly_acc = {m: grp["is_correct"].mean() * 100 for m, grp in df.groupby("month")}
    stock_summary = df.groupby(["ticker", "chinese_name"]).agg(
        Total_Trades=("is_correct", "count"),
        Correct_Trades=("is_correct", "sum"),
        Accuracy_Rate=("is_correct", "mean"),
        Avg_Price_Change=("price_change_pct", "mean"),
        Price_Volatility=("price_change_pct", "std"),
    )
    stock_summary["Accuracy_Rate"] = stock_summary["Accuracy_Rate"] * 100
    stock_summary = stock_summary.round(2)
    return df, monthly_acc, stock_summary

def render_backtest_matrix_html(df: pd.DataFrame, monthly_acc: Dict[str, float], stock_summary: pd.DataFrame, out_html: str):
    """
    產生彩色 HTML 矩陣：index=股票，columns=月份，cell=✓/✗ + 漲跌幅%
    """
    if df.empty:
        with open(out_html, "w", encoding="utf-8") as f:
            f.write("<h3>無回測資料（可能區間過短或資料源暫時不可用）</h3>")
        return

    def cell_text(r):
        arrow = "↑" if r["price_change_pct"] > 0 else ("↓" if r["price_change_pct"] < 0 else "→")
        sign = "+" if r["price_change_pct"] > 0 else ""
        mark = "✓" if r["is_correct"] else "✗"
        return f'{mark} {arrow} {sign}{r["price_change_pct"]:.1f}%'

    df2 = df.copy()
    df2["cell"] = df2.apply(cell_text, axis=1)
    pivot = df2.pivot_table(values="cell", index=["ticker", "chinese_name"], columns="month", aggfunc="first").fillna("—")
    style_map = df.set_index(["ticker", "chinese_name", "month"])["is_correct"].to_dict()
    months_sorted = sorted(df2["month"].unique())

    css = """
    <style>
      table.bt {border-collapse: collapse; width: 100%;}
      table.bt th, table.bt td {border: 1px solid #ddd; padding: 6px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", "Microsoft JhengHei", Arial, sans-serif; font-size: 12px;}
      table.bt th {background: #f6f7f9; position: sticky; top: 0;}
      td.ok {background: #e8f5e9; color: #1b5e20; font-weight: 600;}
      td.fail {background: #ffebee; color: #b71c1c; font-weight: 600;}
      td.neutral {background: #f3f4f6; color: #374151; font-weight: 600;}
      .small {color:#6b7280; font-size:12px}
      .kpi {display:inline-block; margin-right:16px; padding:6px 10px; background:#f3f4f6; border-radius:6px;}
      .wrap {max-width: 1200px; margin: 0 auto;}
    </style>
    """

    overall_acc = df["is_correct"].mean() * 100
    kpi_html = f"""
    <div style="margin:10px 0;">
      <span class="kpi">整體準確率：{overall_acc:.1f}%</span>
      <span class="kpi">樣本數：{len(df):,}</span>
    </div>
    """

    monthly_html = ""
    if monthly_acc:
        months_sorted2 = sorted(monthly_acc.keys())
        monthly_html = "<div class='small'>月度準確率：" + " | ".join([f"{m}: {monthly_acc[m]:.1f}%" for m in months_sorted2]) + "</div>"

    header = "<tr><th>代號</th><th>名稱</th>" + "".join([f"<th>{m}</th>" for m in months_sorted]) + "</tr>"

    rows_html = []
    for (ticker, cname), row in pivot.iterrows():
        tds = [f"<td>{ticker}</td>", f"<td>{cname}</td>"]
        for m in months_sorted:
            val = row.get(m, "—")
            if val == "—":
                cls = "neutral"
            else:
                cls = "ok" if style_map.get((ticker, cname, m), False) else "fail"
            tds.append(f'<td class="{cls}">{val}</td>')
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    summary_html = "<h4>每檔統計摘要</h4>"
    if not stock_summary.empty:
        summary_html += stock_summary.reset_index().to_html(index=False)

    html = f"""
    {css}
    <div class="wrap">
      <h2>股票操作建議月度回測報告</h2>
      {kpi_html}
      {monthly_html}
      <h3>回測矩陣（✓/✗ + 漲跌幅%）</h3>
      <table class="bt">
        {header}
        {''.join(rows_html)}
      </table>
      {summary_html}
    </div>
    """
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

def save_monthly_accuracy_chart(monthly_acc: Dict[str, float], out_path: str):
    if not monthly_acc:
        return
    acc_df = pd.DataFrame(sorted(monthly_acc.items()), columns=["Month", "Accuracy"])
    acc_df["Month"] = pd.to_datetime(acc_df["Month"])
    plt.figure(figsize=(9, 4))
    plt.plot(acc_df["Month"], acc_df["Accuracy"], marker="o", color="#2563eb")
    plt.title("月度投資建議準確率趨勢")
    plt.ylabel("準確率 (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="每月一日建議 → 下月驗證 的回測工具")
    parser.add_argument("--tickers", type=str, default="",
                        help="以逗號分隔之股票清單，如 AAPL,MSFT,2330.TW；未給則用內建 DEFAULT_TICKERS")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="回測起始（YYYY-MM 或 YYYY-MM-DD），預設 2024-01-01")
    parser.add_argument("--end", type=str, default=None,
                        help="回測終止（YYYY-MM 或 YYYY-MM-DD），預設：本月月初")
    parser.add_argument("--hold_band", type=float, default=HOLD_BAND_FOR_CORRECTNESS,
                        help="判定持有正確的容忍區間（默認 0.05 → ±5%）")
    parser.add_argument("--buy_thr", type=float, default=BUY_IF_MONTHLY_CHANGE_LEQ,
                        help="買進門檻（本月相對上月漲跌幅 ≤ buy_thr → 買進），預設 -0.03")
    parser.add_argument("--sell_thr", type=float, default=SELL_IF_MONTHLY_CHANGE_GEQ,
                        help="賣出門檻（本月相對上月漲跌幅 ≥ sell_thr → 賣出），預設 0.07")
    parser.add_argument("--outdir", type=str, default="backtest_output",
                        help="輸出資料夾，預設 backtest_output")
    args = parser.parse_args()

    # 解析輸入的 tickers
    if args.tickers.strip():
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = DEFAULT_TICKERS

    # 預設 end 為本月月初
    if not args.end:
        end = pd.Timestamp.today().to_period("M").to_timestamp().strftime("%Y-%m-%d")
    else:
        end = args.end

    os.makedirs(args.outdir, exist_ok=True)

    print("開始回測...")
    print(f"- 標的：{', '.join(tickers)}")
    print(f"- 區間：{args.start} → {end}")
    print(f"- 規則：跌幅≤{args.buy_thr:.2%} 買進；漲幅≥{args.sell_thr:.2%} 賣出；否則持有")
    print(f"- 持有判定容忍度：±{args.hold_band:.1%}\n")

    df, monthly_acc, stock_summary = backtest_recommendations(
        tickers=tickers,
        start=args.start,
        end=end,
        hold_band=args.hold_band,
        buy_thr=args.buy_thr,
        sell_thr=args.sell_thr
    )

    if df.empty:
        print("無回測資料（可能資料源暫時不可用或區間過短）。")
        return

    # 輸出檔
    detail_csv = os.path.join(args.outdir, "backtest_detail.csv")
    summary_csv = os.path.join(args.outdir, "backtest_summary.csv")
    html_path = os.path.join(args.outdir, "backtest_matrix.html")
    chart_path = os.path.join(args.outdir, "monthly_accuracy.png")

    df.sort_values(["ticker", "date"]).to_csv(detail_csv, index=False, encoding="utf-8-sig")
    stock_summary.reset_index().to_csv(summary_csv, index=False, encoding="utf-8-sig")
    render_backtest_matrix_html(df, monthly_acc, stock_summary, out_html=html_path)
    save_monthly_accuracy_chart(monthly_acc, out_path=chart_path)

    # 主控台輸出摘要
    overall_acc = df["is_correct"].mean() * 100
    print("=== 回測完成 ===")
    print(f"整體準確率：{overall_acc:.1f}%")
    print("每檔勝率：")
    show_df = stock_summary.reset_index()[["ticker", "chinese_name", "Total_Trades", "Correct_Trades", "Accuracy_Rate", "Avg_Price_Change"]]
    show_df = show_df.rename(columns={
        "ticker": "Ticker",
        "chinese_name": "Name",
        "Total_Trades": "樣本數",
        "Correct_Trades": "正確數",
        "Accuracy_Rate": "勝率(%)",
        "Avg_Price_Change": "平均月變動(%)"
    })
    show_df["平均月變動(%)"] = show_df["平均月變動(%)"].round(2)
    print(show_df.to_string(index=False))

    print("\n輸出檔案：")
    print(f"- 明細 CSV：{detail_csv}")
    print(f"- 摘要 CSV：{summary_csv}")
    print(f"- HTML 矩陣：{html_path}")
    print(f"- 月度準確率圖：{chart_path}")

if __name__ == "__main__":
    main()
