import os
import math
import traceback
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify
import requests
import yfinance as yf
try:
	from yfinance import Search as YFSearch
except Exception:
	YFSearch = None

app = Flask(__name__)

# ================== 基本設定 ==================
DEFAULT_TICKERS = [
	# 美股七大龍頭
	"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
	# 台灣龍頭
	"2330.TW", "2317.TW", "2454.TW", "2412.TW", "2881.TW","2327.TW", "2308.TW", "3293.TWO"
]

TYPE_OPTIONS = [
	"成長股",
	"資產股",
	"高股息/定存股",
	"虧損轉機股",
	"景氣循環股",
	"穩定獲利股",
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
    "3293.TWO": "鈊象",
}

# ================== 工具函式 ==================
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

def num_to_str(x, digits=2):
	if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
		return "N/A"
	try:
		return f"{x:.{digits}f}"
	except Exception:
		return "N/A"

def fmt_price(x, ccy):
	if x is None:
		return "N/A"
	try:
		return f"{x:,.2f} {ccy}" if ccy else f"{x:,.2f}"
	except Exception:
		return "N/A"

def _cache_dir():
	base = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(base, "cache")
	os.makedirs(path, exist_ok=True)
	return path

def _cache_path():
	return os.path.join(_cache_dir(), "stock_cache.json")

def _load_cache(today_str: str) -> Dict[str, Any]:
	path = _cache_path()
	try:
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
		if data.get("date") != today_str:
			return {"date": today_str, "items": {}}
		if "items" not in data or not isinstance(data.get("items"), dict):
			data["items"] = {}
		return data
	except Exception:
		return {"date": today_str, "items": {}}

def _to_jsonable(value):
	if isinstance(value, dict):
		return {k: _to_jsonable(v) for k, v in value.items()}
	if isinstance(value, list):
		return [_to_jsonable(v) for v in value]
	try:
		json.dumps(value)
		return value
	except Exception:
		try:
			return float(value)
		except Exception:
			return str(value)

def _save_cache(cache_data: Dict[str, Any]):
	path = _cache_path()
	try:
		with open(path, "w", encoding="utf-8") as f:
			json.dump(_to_jsonable(cache_data), f, ensure_ascii=False)
	except Exception:
		pass

def _tw_cache_path():
	return os.path.join(_cache_dir(), "tw_symbols.json")

def _load_tw_cache(today_str: str) -> Dict[str, Any]:
	path = _tw_cache_path()
	try:
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
		if data.get("date") != today_str:
			return {"date": today_str, "items": []}
		if "items" not in data or not isinstance(data.get("items"), list):
			data["items"] = []
		return data
	except Exception:
		return {"date": today_str, "items": []}

def _save_tw_cache(data: Dict[str, Any]):
	path = _tw_cache_path()
	try:
		with open(path, "w", encoding="utf-8") as f:
			json.dump(_to_jsonable(data), f, ensure_ascii=False)
	except Exception:
		pass

def _fetch_tw_symbols(today_str: str) -> List[Dict[str, str]]:
	cache_data = _load_tw_cache(today_str)
	if cache_data.get("items"):
		return cache_data["items"]
	items: List[Dict[str, str]] = []
	endpoints = [
		("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", ".TW"),
		("https://openapi.twse.com.tw/v1/opendata/t187ap03_O", ".TWO"),
	]
	for url, suffix in endpoints:
		try:
			resp = requests.get(url, timeout=12)
			resp.raise_for_status()
			data = resp.json()
			if isinstance(data, list):
				for row in data:
					code = str(row.get("公司代號") or row.get("公司代碼") or "").strip()
					name = str(row.get("公司簡稱") or row.get("公司名稱") or "").strip()
					if code and name:
						items.append({"symbol": f"{code}{suffix}", "name": name})
		except Exception:
			continue
	cache_data = {"date": today_str, "items": items}
	_save_tw_cache(cache_data)
	return items

def merge_display_name(symbol: str, en_name: str) -> str:
	cn = CHINESE_NAME_MAP.get(symbol)
	if not cn:
		return en_name
	if cn in en_name:
		return en_name
	return f"{cn} ({en_name})" if en_name and en_name != symbol else cn

def current_price(tkr, info_full) -> float:
	price = None
	try:
		fi = tkr.fast_info
		price = safe_get(fi, "lastPrice", None)
	except Exception:
		price = None
	if not price:
		price = safe_get(info_full, "currentPrice", None)
	if not price:
		try:
			h = tkr.history(period="1d")
			if not h.empty:
				price = float(h["Close"].iloc[-1])
		except Exception:
			price = None
	return price

def detect_applicable(sector, industry) -> List[str]:
	applicable = []
	sec = (sector or "").lower()
	ind = (industry or "").lower()
	if ("financial" in sec) or any(k in ind for k in ["bank", "insurance", "capital markets", "financial"]):
		applicable.append("金融/保險股")
	if any(k in ind for k in ["steel", "iron", "chem", "chemical", "marine", "shipping", "freight", "aluminum", "metals", "mining"]):
		applicable.append("鋼鐵/化工/航運")
	return applicable

def classify_type(info_full: Dict[str, Any]):
	sector = safe_get(info_full, "sector", "")
	industry = safe_get(info_full, "industry", "")
	revenue_growth = safe_get(info_full, "revenueGrowth", None)
	earnings_growth = safe_get(info_full, "earningsGrowth", None)
	profit_margin = safe_get(info_full, "profitMargins", None)
	dividend_yield_raw = safe_get(info_full, "dividendYield", None)
	# Yahoo Finance API 的 dividendYield 返回的是百分比數值（如 3.82 = 3.82%），需要除以 100 轉換為小數
	dividend_yield = dividend_yield_raw / 100 if dividend_yield_raw is not None else None
	dividend_yield_avg_raw = safe_get(info_full, "fiveYearAvgDividendYield", None)
	dividend_yield_avg = dividend_yield_avg_raw / 100 if dividend_yield_avg_raw is not None else None
	beta = safe_get(info_full, "beta", None)
	applicable = detect_applicable(sector, industry)
	it = (industry or "").lower(); sec = (sector or "").lower()
	
	# 判斷是否為成長導向產業（科技、電動車、AI等）
	growth_sectors = ["technology", "consumer cyclical"]
	growth_keywords = ["software", "semiconductor", "auto manufacturers", "electric", "ai", "cloud"]
	is_growth_industry = (sec in growth_sectors) or any(k in it for k in growth_keywords)

	defensive_sectors = ["communication services", "utilities"]
	defensive_keywords = ["telecom", "wireless", "broadband", "telecommunication", "utility"]
	is_defensive_income_industry = (sec in defensive_sectors) or any(k in it for k in defensive_keywords)
	yield_for_class = dividend_yield if dividend_yield is not None else dividend_yield_avg
	core_high_yield = (dividend_yield is not None and dividend_yield >= 0.04 and (revenue_growth or 0) < 0.10 and (beta is None or beta < 0.9))
	defensive_income = (
		is_defensive_income_industry
		and yield_for_class is not None and yield_for_class >= 0.03
		and (beta is None or beta <= 0.8)
	)
	
	strong_growth = (
		(revenue_growth is not None and revenue_growth >= 0.15)
		or (earnings_growth is not None and earnings_growth >= 0.15)
		or (is_growth_industry and (revenue_growth or 0) > 0 and beta is not None and beta > 1.3)
	)

	if strong_growth:
		stock_type = "成長股"
	elif ("real estate" in sec) or any(k in it for k in ["reit", "real estate", "property"]):
		stock_type = "資產股"
	elif (core_high_yield or defensive_income) and not strong_growth:
		stock_type = "高股息/定存股"
	elif profit_margin is not None and profit_margin < 0:
		stock_type = "虧損轉機股"
	elif any(k in it for k in ["steel", "iron", "chem", "chemical", "marine", "shipping", "freight", "aluminum", "metals", "mining"]) or sec in ["materials", "energy"]:
		stock_type = "景氣循環股"
	else:
		stock_type = "穩定獲利股"
	cond = {
		"sector": sector,
		"industry": industry,
		"revenueGrowth": revenue_growth,
		"earningsGrowth": earnings_growth,
		"profitMargins": profit_margin,
		"dividendYield": dividend_yield,  # 已修正為真實比例
		"dividendYield5Y": dividend_yield_avg,
		"beta": beta,
		"payoutRatio": safe_get(info_full, "payoutRatio", None),
	}
	return stock_type, applicable, cond

def compute_cycle_eps(ticker_obj, info_full: Dict[str, Any]):
	"""計算 3~5 年平均 EPS (週期平滑)。透過 ticker.earnings DataFrame (Earnings/share count)。"""
	try:
		shares = safe_get(info_full, "sharesOutstanding", None)
		if not shares or shares <= 0:
			return None
		df = getattr(ticker_obj, "earnings", None)
		if df is None or df.empty:
			return None
		df_tail = df.tail(5)
		if df_tail.empty:
			return None
		net_series = df_tail.get("Earnings")
		if net_series is None:
			return None
		vals = [float(v) for v in net_series.tolist() if isinstance(v,(int,float))]
		if len(vals) < 2:
			return None
		avg_net = sum(vals)/len(vals)
		cycle_eps = avg_net / shares
		return cycle_eps if cycle_eps and cycle_eps>0 else None
	except Exception:
		return None

def compute_valuations(stock_type: str, applicable: List[str], info_full: Dict[str, Any], price: float, *, ticker_obj=None):
	currency = safe_get(info_full, "currency", "") or safe_get(info_full, "financialCurrency", "") or ""
	trailing_pe = safe_get(info_full, "trailingPE", None)
	forward_eps = safe_get(info_full, "forwardEps", None)
	book_value = safe_get(info_full, "bookValue", None)
	roe = safe_get(info_full, "returnOnEquity", None)
	rev_ps = safe_get(info_full, "revenuePerShare", None)
	profit_margin = safe_get(info_full, "profitMargins", None)
	earnings_growth = safe_get(info_full, "earningsGrowth", None)
	revenue_growth = safe_get(info_full, "revenueGrowth", None)
	dividend_yield_raw = safe_get(info_full, "dividendYield", None)
	dividend_yield = dividend_yield_raw / 100 if dividend_yield_raw is not None else None
	dividend_yield_avg_raw = safe_get(info_full, "fiveYearAvgDividendYield", None)
	dividend_yield_avg = dividend_yield_avg_raw / 100 if dividend_yield_avg_raw is not None else None
	payout_ratio = safe_get(info_full, "payoutRatio", None)
	div_rate_raw = safe_get(info_full, "trailingAnnualDividendRate", None)
	# trailingAnnualDividendRate 通常是絕對金額（如 1.02 美元），不需要轉換
	div_rate = div_rate_raw
	eps_ttm = None
	if price and trailing_pe and trailing_pe > 0:
		eps_ttm = price / trailing_pe
	elif forward_eps:
		eps_ttm = forward_eps
	def apply_growth(mult):
		# 新邏輯：倍數 * (1 + 成長率)，成長率 g 取 earningsGrowth 優先，否則 revenueGrowth
		g = earnings_growth if earnings_growth is not None else revenue_growth
		if g is None:
			return mult
		low, mid, high = mult
		try:
			# 限制影響幅度，避免極端值
			if g > 0.60:
				g_used = 0.60
			elif g < -0.40:
				g_used = -0.40
			else:
				g_used = g
			factor = 1 + g_used  # 例如 25% 成長 → 1.25
			return (low * factor, mid * factor, high * factor)
		except Exception:
			return mult
	def apply_roe(mult):
		low, mid, high = mult; r = roe
		if r is None: return mult
		try:
			if r > 0.15: return (low*1.2, mid*1.2, high*1.2)
			elif r < 0.08: return (low*0.8, mid*0.8, high*0.8)
			return mult
		except Exception: return mult
	def apply_margin(mult):
		low, mid, high = mult; m = profit_margin
		if m is None: return mult
		try:
			if m > 0.20: return (low*1.3, mid*1.3, high*1.3)
			elif 0.10 <= m <= 0.20: return (low*1.1, mid*1.1, high*1.1)
			elif m < 0.05: return (low*0.7, mid*0.7, high*0.7)
			return mult
		except Exception: return mult
	cheap=fair=expensive=None; method=""; method_detail=""
	try:
		if "金融/保險股" in applicable or stock_type=="資產股":
			method="P/B"; mult=(0.8,1.2,1.8) if "金融/保險股" in applicable else (0.6,1.0,1.5); mult=apply_roe(mult)
			if book_value:
				cheap,fair,expensive=(book_value*mult[0], book_value*mult[1], book_value*mult[2])
				method_detail=f"每股淨值×倍數；ROE調整後倍數={tuple(round(m,2) for m in mult)}"
			elif eps_ttm:
				method="P/E(後備)"; base=(12,18,25); base=apply_growth(base)
				cheap,fair,expensive=(eps_ttm*base[0], eps_ttm*base[1], eps_ttm*base[2])
				method_detail=f"EPS×倍數；成長調整後倍數={tuple(round(b,2) for b in base)}"
		elif stock_type=="高股息/定存股":
			method="DDM/殖利率帶"
			base_yield=None; base_source=""
			if dividend_yield_avg:
				base_yield=dividend_yield_avg; base_source="5Y平均殖利率"
			elif dividend_yield:
				base_yield=dividend_yield; base_source="近12月殖利率"
			elif price and div_rate and price>0:
				base_yield=max(div_rate/price, 0.0001); base_source="即時殖利率"
			if div_rate and div_rate>0 and base_yield:
				base_yield=max(base_yield, 0.005)
				premium=max(base_yield*0.3, 0.01)
				discount=max(base_yield*0.2, 0.005)
				cheap_yield=min(base_yield + premium, base_yield + 0.04)
				fair_yield=base_yield
				expensive_yield=max(base_yield - discount, 0.01)
				cheap=fair=expensive=None
				try:
					cheap = div_rate / cheap_yield
					fair = div_rate / fair_yield
					expensive = div_rate / expensive_yield
				except Exception:
					cheap=fair=expensive=None
				method_detail=f"動態殖利率帶（基準 {base_source} {base_yield*100:.2f}% → 高/中/低殖利率={cheap_yield*100:.2f}%/{fair_yield*100:.2f}%/{expensive_yield*100:.2f}%）"
				if payout_ratio is not None and payout_ratio>1.1 and all(v is not None for v in [cheap,fair,expensive]):
					adj = 0.85 if payout_ratio>=1.5 else 0.93
					cheap*=adj; fair*=adj; expensive*=adj
					method_detail += f"；Payout {payout_ratio:.2f} → 價格×{adj:.2f}"
			elif eps_ttm:
				method="P/E(後備)"; base=(12,18,25)
				cheap,fair,expensive=(eps_ttm*base[0], eps_ttm*base[1], eps_ttm*base[2])
				method_detail="無股利資料，改用EPS×P/E"
		elif stock_type=="虧損轉機股":
			method="P/S"; mult=(1.0,1.5,2.5); mult=apply_margin(mult)
			if rev_ps:
				cheap,fair,expensive=(rev_ps*mult[0], rev_ps*mult[1], rev_ps*mult[2])
				method_detail=f"每股營收×倍數；利潤率調整後倍數={tuple(round(m,2) for m in mult)}"
			elif eps_ttm:
				method="P/E(後備)"; base=(10,15,20)
				cheap,fair,expensive=(eps_ttm*base[0], eps_ttm*base[1], eps_ttm*base[2])
				method_detail="無營收資料，改用EPS×P/E"
		elif stock_type=="成長股":
			if eps_ttm:
				method="P/E"; base=(18,28,40)  # 調整後基準
				growth_rate = earnings_growth if earnings_growth is not None else revenue_growth
				base_adj = apply_growth(base)
				peg_note = ""
				# PEG 檢核： forward P/E / (growth% * 100)，滿足 <=1.2 則額外 +10%
				try:
					if forward_eps and growth_rate and growth_rate>0:
						pe_f = price / forward_eps if forward_eps else None
						if pe_f and pe_f>0:
							peg = pe_f / (growth_rate*100)
							if peg <= 1.2:
								base_adj = tuple(b*1.10 for b in base_adj)
								peg_note = f"；PEG={peg:.2f}<=1.2，上調10%"
				except Exception:
					pass
				cheap,fair,expensive=(eps_ttm*base_adj[0], eps_ttm*base_adj[1], eps_ttm*base_adj[2])
				if growth_rate is not None:
					g_used = max(min(growth_rate,0.60), -0.40)
					factor = 1 + g_used
					method_detail=f"EPS×倍數；成長率={growth_rate:.2%}→因子{factor:.2f}→調整後倍數={tuple(round(b,2) for b in base_adj)}{peg_note}"
				else:
					method_detail=f"EPS×倍數（無成長率調整）倍數={tuple(round(b,2) for b in base_adj)}{peg_note}"
			elif rev_ps:
				method="P/S"; mult=(1.5,3.0,5.0); mult=apply_margin(mult)
				cheap,fair,expensive=(rev_ps*mult[0], rev_ps*mult[1], rev_ps*mult[2])
				method_detail=f"每股營收×倍數；利潤率調整後倍數={tuple(round(m,2) for m in mult)}"
		elif stock_type=="景氣循環股":
			cycle = compute_cycle_eps(ticker_obj, info_full) if ticker_obj else None
			base_eps = cycle if cycle else eps_ttm
			if base_eps:
				method = "P/E(週期平滑)" if cycle else "P/E"
				base=(8,15,22)
				cheap,fair,expensive=(base_eps*base[0], base_eps*base[1], base_eps*base[2])
				method_detail=("平均(3~5年)EPS×倍數" if cycle else "EPS×倍數") + "（循環股基準 8/15/22）"
			elif rev_ps:
				method="P/S(後備)"; mult=(0.5,1.0,2.0)
				cheap,fair,expensive=(rev_ps*mult[0], rev_ps*mult[1], rev_ps*mult[2])
				method_detail="每股營收×倍數（循環股基準 0.5/1/2）"
		else:
			method="P/E"; base=(12,18,25)
			if eps_ttm:
				cheap,fair,expensive=(eps_ttm*base[0], eps_ttm*base[1], eps_ttm*base[2])
				method_detail="EPS×倍數（穩定獲利基準 12/18/25）"
			elif book_value:
				method="P/B(後備)"; mult=(1.5,2.5,3.5); mult=apply_roe(mult)
				cheap,fair,expensive=(book_value*mult[0], book_value*mult[1], book_value*mult[2])
				method_detail=f"每股淨值×倍數；ROE調整後倍數={tuple(round(m,2) for m in mult)}"
	except Exception:
		cheap=fair=expensive=None; method="N/A"; method_detail="估值計算發生例外"
	return {"currency":currency or "", "cheap":cheap, "fair":fair, "expensive":expensive, "method":method, "method_detail":method_detail}

def analyst_range(info_full: Dict[str, Any]):
	return {
		"low": safe_get(info_full, "targetLowPrice", None),
		"mean": safe_get(info_full, "targetMeanPrice", None),
		"high": safe_get(info_full, "targetHighPrice", None),
		"n": safe_get(info_full, "numberOfAnalystOpinions", None),
		"rec": safe_get(info_full, "recommendationKey", None),
	}

def analyst_gap_ratio(fair, analyst_mean) -> float:
	if fair is None or analyst_mean is None:
		return None
	try:
		base = max(abs(float(fair)), 1e-6)
		return abs(float(analyst_mean) - float(fair)) / base
	except Exception:
		return None

def confidence_score(price, cheap, fair, expensive, analyst_mean, analyst_n, cond: Dict[str, Any]) -> str:
	"""以估值一致性、分析師一致性、波動與獲利品質計算信心度。"""
	score = 50.0
	if all(v is not None for v in [price, cheap, fair, expensive]):
		score += 10
	gap = analyst_gap_ratio(fair, analyst_mean)
	if gap is not None:
		if gap <= 0.15:
			score += 15
		else:
			score -= 20
	beta = cond.get("beta")
	if beta is not None:
		if beta < 1.0:
			score += 10
		elif beta > 1.5:
			score -= 10
	profit_margin = cond.get("profitMargins")
	if profit_margin is not None and profit_margin < 0:
		score -= 15
	payout_ratio = cond.get("payoutRatio")
	if payout_ratio is not None and payout_ratio > 1.0:
		score -= 10
	if analyst_n is not None:
		if analyst_n >= 6:
			score += 5
		elif analyst_n < 3:
			score -= 5
	score = max(0, min(100, score))
	if score >= 80:
		level = "高"
	elif score >= 60:
		level = "中"
	else:
		level = "低"
	return f"{level} ({score:.0f})"

def suggest_action(price, cheap, fair, expensive):
	if not price or not cheap or not fair or not expensive:
		return "資料不足"
	try:
		if price <= cheap: return "分批買進"
		elif price <= 0.95 * fair: return "買進"
		elif price <= expensive: return "持有/減碼"
		else: return "賣出/減碼"
	except Exception: return "—"

def suggest_action_with_analyst(price, cheap, fair, expensive, analyst_low, analyst_mean, analyst_high):
	base = suggest_action(price, cheap, fair, expensive)
	if base in ["資料不足", "—"]:
		return base
	gap = analyst_gap_ratio(fair, analyst_mean)
	valuation_bias = "flat"
	if price <= 0.95 * fair:
		valuation_bias = "up"
	elif price >= expensive:
		valuation_bias = "down"
	analyst_bias = "flat"
	if analyst_mean is not None:
		if analyst_mean >= price * 1.05:
			analyst_bias = "up"
		elif analyst_mean <= price * 0.95:
			analyst_bias = "down"
	elif analyst_low is not None and analyst_high is not None:
		if price < analyst_low:
			analyst_bias = "up"
		elif price > analyst_high:
			analyst_bias = "down"
	if analyst_low is not None and price <= cheap and price <= analyst_low:
		return "強烈買進"
	if analyst_high is not None and price >= expensive and price >= analyst_high:
		return "強烈賣出"
	if gap is not None and gap > 0.15 and analyst_bias in ["up", "down"] and valuation_bias in ["up", "down"] and analyst_bias != valuation_bias:
		return "觀望/分歧"
	return base

def evaluate_action_success(action: str, pct_change: float):
	"""評估歷史操作是否命中方向。"""
	if pct_change is None:
		return None
	if action in ["分批買進", "買進"]:
		return pct_change >= 0.01
	if action == "持有/減碼":
		return abs(pct_change) <= 0.03
	if action == "賣出/減碼":
		return pct_change <= -0.01
	return None

def compute_operation_win_rate(ticker_obj, cheap, fair, expensive):
	if not ticker_obj or not all(v is not None for v in [cheap, fair, expensive]):
		return None
	try:
		required_checks = 12
		hist = ticker_obj.history(period="13mo", interval="1mo")
		if hist is None or hist.empty:
			return None
		hist = hist.dropna(subset=["Close"])
		if len(hist) < required_checks + 1:
			return None
		closes = hist["Close"].tolist()[-(required_checks + 1):]
		wins = 0
		total = 0
		for idx in range(len(closes) - 1):
			price = closes[idx]; nxt = closes[idx + 1]
			if price is None or nxt is None or price <= 0:
				continue
			action = suggest_action(price, cheap, fair, expensive)
			pct_change = (nxt - price) / price
			result = evaluate_action_success(action, pct_change)
			if result is None:
				continue
			total += 1
			if result:
				wins += 1
			if total == required_checks:
				break
		if total == 0:
			return None
		return {"ratio": wins / total, "wins": wins, "total": total}
	except Exception:
		return None

def derive_movement_estimate(price, cheap, fair, expensive, analyst_low, analyst_mean, analyst_high, stock_type: str) -> str:
	"""依估值區間/分析師/股性推估方向與時間。"""
	try:
		if not price or not fair or not cheap or not expensive:
			return "—"
		if price <= cheap: zone="低估區"
		elif price < 0.95*fair: zone="偏低區"
		elif price <= 1.05*fair: zone="合理區"
		elif price < expensive: zone="偏高區"
		else: zone="高估區"
		direction="⇆"; target=None; target_label=None
		if zone in ["低估區","偏低區"]:
			cands=[]
			if fair and fair>price: cands.append((fair,"合理價"))
			if analyst_mean and analyst_mean>price: cands.append((analyst_mean,"分析師均值"))
			if cands:
				cands.sort(key=lambda x:x[0])
				target,target_label=cands[0]; direction="↑"
		elif zone=="合理區":
			if analyst_mean and abs(analyst_mean-price)/price>0.05:
				direction = "↑" if analyst_mean>price else "↓"
				target=analyst_mean; target_label="分析師均值"
			else:
				return "⇆ 合理區（±5%），短期 1-3個月震盪"
		else: # 偏高/高估
			cands=[]
			if fair and fair<price: cands.append((fair,"合理價"))
			if cheap and cheap<price: cands.append((cheap,"便宜價"))
			if cands:
				cands.sort(key=lambda x:x[0], reverse=True)
				target,target_label=cands[0]; direction="↓"
		if not target:
			return zone
		pct_move=(target-price)/price; pct_abs=abs(pct_move)
		fast=["成長股","虧損轉機股"]; mid=["景氣循環股","資產股"]; slow=["穩定獲利股","高股息/定存股"]
		if stock_type in fast: ladders=[(0.05,"0-1個月"),(0.10,"1-2個月"),(0.20,"2-4個月"),(0.35,"3-6個月"),(1.00,"6-9個月")]
		elif stock_type in mid: ladders=[(0.05,"1-2個月"),(0.10,"2-3個月"),(0.20,"3-6個月"),(0.35,"6-9個月"),(1.00,"9-12個月")]
		else: ladders=[(0.05,"1-3個月"),(0.10,"3-6個月"),(0.20,"6-9個月"),(0.35,"9-12個月"),(1.00,"12個月以上")]
		timeframe=ladders[-1][1]
		for th, tf in ladders:
			if pct_abs <= th:
				timeframe=tf; break
		sign_pct=f"{pct_abs*100:.1f}%"; return f"{direction} {sign_pct} → {timeframe}（目標：{target_label} {target:,.2f}）"
	except Exception:
		return "—"

def compute_risk_alert(price, cheap, fair, expensive, analyst_low, analyst_mean, analyst_high, stock_type: str, cond: Dict[str, Any]) -> Tuple[str, str]:
	"""根據估值乖離、分析師區間、成長/獲利/波動指標計算風險提醒。
	回傳 (文字, 等級class) 等級class 之一: risk-high | risk-medium | risk-low
	"""
	try:
		if not price or not fair or not cheap or not expensive:
			return ("資料不足", "risk-low")
		reasons = []
		score = 0
		gap = analyst_gap_ratio(fair, analyst_mean)
		# 估值過熱 / 過度低估
		if price and expensive and (price - expensive) / expensive > 0.10:
			score += 2; reasons.append("高於昂貴價10%+")
		elif price and fair and (price - fair) / fair > 0.10:
			score += 1; reasons.append("高於合理價10%+")
		if price and cheap and (cheap - price) / cheap > 0.15:
			score += 1; reasons.append("遠低於便宜價")
		# 分析師區間乖離
		if analyst_high and price > analyst_high * 1.05:
			score += 2; reasons.append("高於分析師高標5%+")
		if analyst_low and price < analyst_low * 0.90:
			score += 1; reasons.append("跌破分析師低標10%")
		if gap is not None and gap > 0.15:
			score += 2; reasons.append("估值分歧(模型 vs 分析師)")
		if analyst_low and analyst_high and analyst_low > 0:
			try:
				if analyst_high / analyst_low > 1.6:
					score += 1; reasons.append("分析師分歧(區間過寬)")
			except Exception:
				pass
		# 成長/獲利風險
		earnings_growth = cond.get("earningsGrowth")
		revenue_growth = cond.get("revenueGrowth")
		profit_margin = cond.get("profitMargins")
		dividend_yield = cond.get("dividendYield")
		beta = cond.get("beta")
		payout_ratio = cond.get("payoutRatio")
		if profit_margin is not None and profit_margin < 0:
			score += 1; reasons.append("虧損")
		if (earnings_growth is not None and earnings_growth < 0) and stock_type in ["成長股", "虧損轉機股"]:
			score += 1; reasons.append("成長放緩/負成長")
		if revenue_growth is not None and revenue_growth < 0:
			score += 1; reasons.append("營收衰退")
		# 波動風險
		if beta is not None:
			if beta > 2:
				score += 2; reasons.append(f"Beta高({beta:.1f})")
			elif beta > 1.6:
				score += 1; reasons.append(f"Beta偏高({beta:.1f})")
		# 配息風險
		if payout_ratio is not None and payout_ratio > 1.0:
			score += 1; reasons.append("配息率偏高")
		# 股息可持續性 (簡易): 高殖利率但獲利為負
		if dividend_yield and dividend_yield > 0.06 and profit_margin is not None and profit_margin < 0:
			score += 1; reasons.append("高殖利率+虧損風險")
		# 分類 + 高位
		if stock_type in ["成長股", "虧損轉機股"] and price and expensive and price > 0.95 * expensive:
			score += 1; reasons.append("高波動股接近昂貴價")
		# 決定等級
		if score >= 5:
			level = "risk-high"
		elif score >= 3:
			level = "risk-medium"
		else:
			level = "risk-low"
		if not reasons:
			return ("風險低", "risk-low")
		text = "、".join(reasons[:4])  # 最多列 4 個理由
		return (text, level)
	except Exception:
		return ("—", "risk-low")

# ================== 主頁 ==================
@app.route("/api/search", methods=["GET"])
def api_search():
	q = request.args.get("q", "").strip()
	limit = int(request.args.get("limit", "12") or 12)
	if not q:
		return jsonify({"ok": True, "data": []})
	results = []
	q_lower = q.lower()
	# 台股中文/代號動態查詢（上市/上櫃）
	today_str = datetime.now().strftime("%Y-%m-%d")
	try:
		tw_symbols = _fetch_tw_symbols(today_str)
		for item in tw_symbols:
			sym = item.get("symbol", "")
			name = item.get("name", "")
			if q_lower in sym.lower() or q_lower in name.lower():
				results.append({"symbol": sym, "name": name})
				if len(results) >= limit:
					break
	except Exception:
		pass
	# 使用 Yahoo Finance 搜尋（全市場）
	if YFSearch is not None:
		try:
			search = YFSearch(q)
			quotes = getattr(search, "quotes", []) or []
			for item in quotes:
				symbol = item.get("symbol") or ""
				name = item.get("shortname") or item.get("longname") or item.get("name") or symbol
				if symbol:
					results.append({"symbol": symbol, "name": name})
		except Exception:
			pass
	# 若仍無結果，退回內建清單模糊比對
	if not results:
		for sym, name in CHINESE_NAME_MAP.items():
			if q_lower in sym.lower() or q_lower in name.lower():
				results.append({"symbol": sym, "name": name})
	# 去重並限制筆數
	seen = set()
	unique = []
	for r in results:
		key = r.get("symbol", "")
		if key and key not in seen:
			seen.add(key)
			unique.append(r)
		if len(unique) >= limit:
			break
	return jsonify({"ok": True, "data": unique})

@app.route("/", methods=["GET"])
def home():
	tickers_param = request.args.get("tickers", "")
	types_param = request.args.get("types", "")
	symbols = [t.strip() for t in tickers_param.split(",") if t.strip()] if tickers_param else DEFAULT_TICKERS[:]
	current_symbols = symbols[:]
	type_overrides: Dict[str, str] = {}
	if types_param:
		parts = [p.strip() for p in types_param.split(",") if p.strip()]
		for part in parts:
			if ":" not in part:
				continue
			sym, typ = part.split(":", 1)
			sym = sym.strip()
			typ = typ.strip()
			if sym and typ in TYPE_OPTIONS:
				type_overrides[sym] = typ
	today_str = datetime.now().strftime("%Y-%m-%d")
	cache_data = _load_cache(today_str)
	known_list = [
		{
			"symbol": s,
			"name": merge_display_name(s, CHINESE_NAME_MAP.get(s, s))
		}
		for s in DEFAULT_TICKERS
	]
	rows=[]; errors=[]
	for sym in symbols:
		try:
			override_type = type_overrides.get(sym)
			if override_type:
				cached_row = None
			else:
				cached_row = cache_data.get("items", {}).get(sym)
			if cached_row:
				rows.append(cached_row)
				continue
			t = yf.Ticker(sym)
			try:
				info_full = t.get_info();
				if not isinstance(info_full, dict): info_full={}
			except Exception:
				try:
					info_full = t.info; 
					if not isinstance(info_full, dict): info_full={}
				except Exception:
					info_full={}
			name_en = safe_get(info_full, "shortName", None) or safe_get(info_full, "longName", None) or sym
			name = merge_display_name(sym, name_en)
			sector = safe_get(info_full, "sector", ""); industry = safe_get(info_full, "industry", "")
			currency = safe_get(info_full, "currency", "") or safe_get(info_full, "financialCurrency", "") or ""
			price = current_price(t, info_full)
			auto_type, applicable, cond = classify_type(info_full)
			stock_type = override_type if override_type in TYPE_OPTIONS else auto_type
			applicable_str = f"（適用：{'、'.join(applicable)}）" if applicable else ""
			val = compute_valuations(stock_type, applicable, info_full, price, ticker_obj=t)
			ar = analyst_range(info_full)
			analysts_str="N/A"
			if ar["low"] and ar["high"]:
				mean_part = f"，均值 {num_to_str(ar['mean'])}" if ar['mean'] else ""
				n_part = f"，{ar['n']}位" if ar['n'] else ""
				analysts_str = f"{num_to_str(ar['low'])} ~ {num_to_str(ar['high'])}{mean_part}{n_part}"
			validity = "估值：1季內；分析師：3–6個月（>12個月權重下調）"
			movement_est = derive_movement_estimate(price, val["cheap"], val["fair"], val["expensive"], ar["low"], ar["mean"], ar["high"], stock_type)
			risk_text, risk_level = compute_risk_alert(price, val["cheap"], val["fair"], val["expensive"], ar["low"], ar["mean"], ar["high"], stock_type, cond)
			conf = confidence_score(price, val["cheap"], val["fair"], val["expensive"], ar["mean"], ar["n"], cond)
			win_rate_stats = compute_operation_win_rate(t, val["cheap"], val["fair"], val["expensive"])
			if win_rate_stats:
				win_rate_disp = f"{win_rate_stats['ratio']*100:.0f}% ({win_rate_stats['wins']}/{win_rate_stats['total']})"
				if win_rate_stats["total"] < 12:
					win_rate_disp += "，樣本偏少"
			else:
				win_rate_disp = "資料不足"
			action = suggest_action_with_analyst(price, val["cheap"], val["fair"], val["expensive"], ar["low"], ar["mean"], ar["high"])
			cond_parts=[]
			if sector: cond_parts.append(f"Sector: {sector}")
			if industry: cond_parts.append(f"Industry: {industry}")
			if cond.get("revenueGrowth") is not None: cond_parts.append(f"營收成長: {pct_to_str(cond['revenueGrowth'])}")
			if cond.get("earningsGrowth") is not None: cond_parts.append(f"盈餘成長: {pct_to_str(cond['earningsGrowth'])}")
			if cond.get("profitMargins") is not None: cond_parts.append(f"淨利率: {pct_to_str(cond['profitMargins'])}")
			if cond.get("dividendYield") is not None: cond_parts.append(f"殖利率: {pct_to_str(cond['dividendYield'])}")
			if cond.get("beta") is not None: cond_parts.append(f"Beta: {num_to_str(cond['beta'],2)}")
			if val["method"]: cond_parts.append(f"估值法: {val['method']}")
			if val["method_detail"]: cond_parts.append(val["method_detail"])
			fair_raw = val.get("fair")
			gap = analyst_gap_ratio(fair_raw, ar.get("mean"))
			divergence = gap is not None and gap > 0.15
			if divergence:
				cond_parts.append(f"模型 vs 分析師均值差距 {gap*100:.0f}%")
			cond_str = "；".join(cond_parts) if cond_parts else "—"
			risk_tags = []
			if risk_text not in ["—", "資料不足", "風險低"]:
				risk_tags = [p.strip() for p in risk_text.split("、") if p.strip()]
			rows.append({
				"symbol": sym,
				"name": name,
				"type": stock_type,
				"type_auto": auto_type,
				"type_suffix": applicable_str,
				"condition": cond_str,
				"divergence": divergence,
				"cheap_raw": val["cheap"],
				"fair_raw": val["fair"],
				"expensive_raw": val["expensive"],
				"price_raw": price,
				"cheap": fmt_price(val["cheap"], currency),
				"fair": fmt_price(val["fair"], currency),
				"expensive": fmt_price(val["expensive"], currency),
				"analyst": (analysts_str + (f" {currency}" if currency and analysts_str != 'N/A' else "")),
				"validity": validity,
				"duration": movement_est,
				"risk": risk_text,
				"risk_tags": risk_tags,
				"risk_level": risk_level,
				"confidence": conf,
				"advice": action,
				"win_rate": win_rate_disp,
				"price": fmt_price(price, currency),
			})
			if not override_type:
				cache_data.setdefault("items", {})[sym] = rows[-1]
		except Exception as e:
			errors.append(f"{sym}: {str(e)}"); traceback.print_exc()
	_save_cache(cache_data)

	generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	table_rows_html=""
	for r in rows:
		highlight_key=None
		try:
			price_raw=r.get("price_raw"); candidates=[]
			for k in ["cheap_raw","fair_raw","expensive_raw"]:
				v=r.get(k)
				if (price_raw is not None) and (v is not None): candidates.append((k, abs(price_raw - v)))
			if candidates:
				candidates.sort(key=lambda x:x[1]); highlight_key=candidates[0][0]
		except Exception: highlight_key=None
		cheap_disp=r["cheap"]; fair_disp=r["fair"]; expensive_disp=r["expensive"]
		if highlight_key=="cheap_raw": cheap_disp=f'<span class="near-price">{cheap_disp}</span>'
		elif highlight_key=="fair_raw": fair_disp=f'<span class="near-price">{fair_disp}</span>'
		elif highlight_key=="expensive_raw": expensive_disp=f'<span class="near-price">{expensive_disp}</span>'
		cond_extra = ""
		if r.get("divergence"):
			cond_extra = " <span class=\"flag-diverge\" title=\"估值分歧：模型合理價與分析師均值差距>15%\">DIV</span>"
		risk_cell = r["risk"]
		risk_tags = r.get("risk_tags") or []
		if risk_tags:
			tags_html = "".join([f'<span class="risk-tag">{t}</span>' for t in risk_tags])
			risk_cell = f'<span class="risk-tags" title="{r["risk"]}">{tags_html}</span>'
		type_selected = "auto" if r.get("type") == r.get("type_auto") else r.get("type")
		auto_label = f"自動判斷（{r.get('type_auto')}）"
		options = [f'<option value="auto"{(" selected" if type_selected=="auto" else "")}>{auto_label}</option>']
		for opt in TYPE_OPTIONS:
			selected = " selected" if type_selected == opt else ""
			options.append(f'<option value="{opt}"{selected}>{opt}</option>')
		type_select_html = f'<select class="type-select" data-symbol="{r["symbol"]}">{"".join(options)}</select>'
		table_rows_html += f"""
		<tr>
		  <td>{r['symbol']}</td>
		  <td>{r['name']}</td>
		  <td>{type_select_html}<span class="type-suffix">{r.get('type_suffix','')}</span></td>
		  <td>{r['price']}</td>
		  <td style=\"max-width:480px;white-space:normal;\">{r['condition']}{cond_extra}</td>
		  <td>{cheap_disp}</td>
		  <td>{fair_disp}</td>
		  <td>{expensive_disp}</td>
		  <td>{r['analyst']}</td>
		  <td>{r['validity']}</td>
		  <td>{r['duration']}</td>
		  <td class=\"{r['risk_level']}\">{risk_cell}</td>
		  <td>{r['confidence']}</td>
		  <td>{r['win_rate']}</td>
		  <td>{r['advice']}</td>
		</tr>"""

	current_symbols_json = json.dumps(current_symbols, ensure_ascii=False)
	known_list_json = json.dumps(known_list, ensure_ascii=False)
	type_overrides_json = json.dumps(type_overrides, ensure_ascii=False)
	tips_payload = {
		"risk": {
			"title": "風險提醒（多標籤）",
			"bullets": [
				"以 score 計分，並把觸發條件轉成標籤（最多顯示 4 個）。",
				"估值過熱：高於昂貴價 10%+（+2）；或高於合理價 10%+（+1）。",
				"估值偏低：遠低於便宜價 15%+（+1）。",
				"分析師偏離：高於分析師高標 5%+（+2）；跌破分析師低標 10%（+1）。",
				"估值分歧：|分析師均值-模型合理價| / |模型合理價| > 15%（+2）。",
				"分析師分歧：分析師區間 high/low > 1.6（+1）。",
				"基本面/成長：虧損（+1）；營收衰退（+1）；成長股/轉機股且盈餘負成長（+1）。",
				"波動：Beta > 2（+2）；Beta > 1.6（+1）。",
				"配息：payoutRatio > 1（+1）；高殖利率+虧損（+1）。",
				"等級：score>=5 高風險；score>=3 中風險；否則低風險。",
			],
		},
		"confidence": {
			"title": "信心度（0–100）",
			"bullets": [
				"以 50 分為基準，依資料完整度、一致性、波動、獲利品質加減分；最後限制在 0~100。",
				"估值資料齊全（現價+便宜/合理/昂貴）→ +10。",
				"分析師一致性：gap=|均值-合理價|/|合理價|；gap<=15% → +15，否則 -20。",
				"波動：beta<1 → +10；beta>1.5 → -10。",
				"獲利：profitMargins<0 → -15。",
				"配息：payoutRatio>1 → -10。",
				"分析師人數：>=6 → +5；<3 → -5。",
				"等級：>=80 高；>=60 中；其餘低。",
			],
		},
		"win_rate": {
			"title": "操作勝率（近 12 期月度命中率）",
			"bullets": [
				"取近 13 個月月 K，最多檢查 12 次（需要至少 13 根月收盤）。",
				"每一期用當期價格與『固定估值帶』產生建議，再用下一期報酬驗證是否命中。",
				"買進/分批買進：下期報酬 >= +1% 視為命中。",
				"持有/減碼：下期報酬介於 -3%~+3% 視為命中。",
				"賣出/減碼：下期報酬 <= -1% 視為命中。",
				"輸出：命中率 %（命中數/樣本數）；樣本 < 12 會標註『樣本偏少』。",
				"注意：估值帶用當前快照，屬規則一致性參考，非嚴格無前視回測。",
			],
		},
		"advice": {
			"title": "操作建議（估值帶 + 分析師修正）",
			"bullets": [
				"先用估值帶判斷：P<=便宜價→分批買進；P<=0.95*合理價→買進；P<=昂貴價→持有/減碼；否則賣出/減碼。",
				"若同時滿足 P<=便宜價 且 P<=分析師低標 → 升級『強烈買進』。",
				"若同時滿足 P>=昂貴價 且 P>=分析師高標 → 升級『強烈賣出』。",
				"若模型合理價與分析師均值差距>15%，且兩者方向相反（都明確看多/看空）→ 顯示『觀望/分歧』。",
			],
		},
	}
	tips_json = json.dumps(tips_payload, ensure_ascii=False)
	html = f"""
	<html><head><meta charset='utf-8'/><title>股票估值儀表板</title>
	<style>
	  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC","Microsoft JhengHei",Arial,sans-serif; }}
	  .container {{ max-width:98%; margin:20px auto; }}
	  .toolbar {{ display:flex; flex-direction:column; gap:8px; margin:12px 0 16px; }}
	  .search-row {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
	  .search-row input {{ padding:8px 10px; min-width:260px; border:1px solid #ccc; border-radius:6px; }}
	  .search-row button {{ padding:8px 12px; border:0; background:#1976d2; color:#fff; border-radius:6px; cursor:pointer; }}
	  .search-row button.secondary {{ background:#546e7a; }}
	  .search-results {{ display:flex; flex-wrap:wrap; gap:6px; }}
	  .search-item {{ background:#f1f5f9; padding:6px 8px; border-radius:6px; cursor:pointer; font-size:12px; }}
	  .watchlist {{ display:flex; flex-wrap:wrap; gap:6px; }}
	  .watch-chip {{ background:#e8f0fe; padding:6px 8px; border-radius:999px; font-size:12px; display:flex; align-items:center; gap:6px; }}
	  .watch-chip button {{ border:0; background:#ef4444; color:#fff; border-radius:999px; padding:0 6px; cursor:pointer; }}
	  table {{ width:100%; border-collapse:collapse; }}
	  th,td {{ border:1px solid #ddd; padding:8px; font-size:14px; vertical-align:top; }}
	  th {{ background:#f6f8fa; position:sticky; top:0; }}
	  tr:nth-child(even) {{ background:#fbfbfb; }}
	  .meta {{ color:#555; font-size:12px; margin:8px 0 16px; }}
	  .near-price {{ color:#d00; font-weight:700; }}
	  .risk-tags {{ display:flex; flex-wrap:wrap; gap:4px; }}
	  .risk-tag {{ display:inline-block; padding:2px 6px; border-radius:999px; font-size:12px; background:#fff; border:1px solid #e5e7eb; color:#374151; }}
	  .flag-diverge {{ display:inline-block; margin-left:6px; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:700; color:#b91c1c; border:1px solid #ef4444; background:#fff1f2; }}
	  .type-select {{ margin-right:6px; padding:4px 6px; border:1px solid #d1d5db; border-radius:6px; background:#fff; font-size:12px; }}
	  .type-suffix {{ color:#6b7280; font-size:12px; margin-left:2px; }}
	  .tip-icon {{ display:inline-block; margin-left:4px; color:#6b7280; font-size:12px; cursor:pointer; user-select:none; }}
	  dialog.tip-dialog {{ border:1px solid #e5e7eb; border-radius:10px; padding:0; width:min(720px, 92vw); }}
	  dialog.tip-dialog::backdrop {{ background: rgba(0,0,0,0.35); }}
	  .tip-head {{ display:flex; align-items:center; justify-content:space-between; padding:12px 14px; border-bottom:1px solid #e5e7eb; background:#f8fafc; }}
	  .tip-title {{ font-weight:700; }}
	  .tip-close {{ border:1px solid #d1d5db; background:#fff; border-radius:8px; padding:6px 10px; cursor:pointer; }}
	  .tip-body {{ padding:12px 14px; }}
	  .tip-body ul {{ margin:0; padding-left:18px; }}
	  .tip-body li {{ margin:6px 0; font-size:14px; color:#111827; }}
	  .risk-high {{ background:#ffe5e5; color:#c40000; font-weight:600; }}
	  .risk-medium {{ background:#fff6e0; color:#b26b00; }}
	  .risk-low {{ background:#e9f9ed; color:#0f6b2f; }}
	</style></head><body>
	  <div class='container'>
		<h1>股票估值儀表板</h1>
		<div class='meta'>生成時間：{generated_at} | 資料：Yahoo Finance（yfinance）</div>
			<div class='meta'>使用：?tickers=AAPL,MSFT,2330.TW 自訂</div>
		<div class='toolbar'>
		  <div class='search-row'>
			<input id='stockSearch' type='text' placeholder='輸入股票代號或關鍵字，例如：AAPL / 台積電' />
			<button id='addBtn'>加入列表</button>
			<button id='clearBtn' class='secondary'>清空列表</button>
		  </div>
		  <div class='search-results' id='searchResults'></div>
		  <div class='watchlist' id='watchList'></div>
		</div>
		<table><thead><tr>
		  <th>股票代號</th>
		  <th>名稱</th>
		  <th>類型</th>
		  <th>目前價位</th>
		  <th>條件</th>
		  <th>便宜價</th>
		  <th>合理價</th>
		  <th>昂貴價</th>
		  <th>分析師價位區間</th>
		  <th>有效期限</th>
		  <th>預估上漲或下跌多久</th>
		  <th>風險提醒 <span class="tip-icon" data-tip="risk" title="點擊查看完整算法">ⓘ</span></th>
		  <th>信心度 <span class="tip-icon" data-tip="confidence" title="點擊查看完整算法">ⓘ</span></th>
		  <th>操作勝率 <span class="tip-icon" data-tip="win_rate" title="點擊查看完整算法">ⓘ</span></th>
		  <th>操作建議 <span class="tip-icon" data-tip="advice" title="點擊查看完整算法">ⓘ</span></th>
		</tr></thead><tbody>
		{table_rows_html}
		</tbody></table>
		<script id='current-tickers' type='application/json'>{current_symbols_json}</script>
		<script id='known-tickers' type='application/json'>{known_list_json}</script>
		<script id='type-overrides' type='application/json'>{type_overrides_json}</script>
		<script id='tips-json' type='application/json'>{tips_json}</script>
		<dialog id="tipDialog" class="tip-dialog">
		  <div class="tip-head">
			<div class="tip-title" id="tipTitle">說明</div>
			<button class="tip-close" id="tipClose" type="button">關閉</button>
		  </div>
		  <div class="tip-body" id="tipBody"></div>
		</dialog>
		<script>
		(() => {{
		  const currentTickers = JSON.parse(document.getElementById('current-tickers').textContent || '[]');
		  const knownTickers = JSON.parse(document.getElementById('known-tickers').textContent || '[]');
		  const typeOverrides = JSON.parse(document.getElementById('type-overrides').textContent || '{{}}');
		  const tips = JSON.parse(document.getElementById('tips-json').textContent || '{{}}');
		  const tipDialog = document.getElementById('tipDialog');
		  const tipTitle = document.getElementById('tipTitle');
		  const tipBody = document.getElementById('tipBody');
		  const tipClose = document.getElementById('tipClose');
		  const searchInput = document.getElementById('stockSearch');
		  const resultsEl = document.getElementById('searchResults');
		  const watchEl = document.getElementById('watchList');
		  const addBtn = document.getElementById('addBtn');
		  const clearBtn = document.getElementById('clearBtn');
		  let lastController = null;
		
		  const normalize = (v) => (v || '').toString().trim();
		  const uniqueList = (arr) => Array.from(new Set(arr.filter(Boolean)));
		  const debounce = (fn, wait = 250) => {{
			let t = null;
			return (...args) => {{
			  if (t) clearTimeout(t);
			  t = setTimeout(() => fn(...args), wait);
			}};
		  }};
		
		  const buildTypesParam = (overrides, list) => {{
			const parts = [];
			(list || []).forEach((sym) => {{
			  const t = overrides[sym];
			  if (t) parts.push(`${{sym}}:${{t}}`);
			}});
			return parts.join(',');
		  }};
		  const updateUrl = (list, overrides = typeOverrides) => {{
			const url = new URL(window.location.href);
			if (list.length) url.searchParams.set('tickers', list.join(','));
			else url.searchParams.delete('tickers');
			const typesParam = buildTypesParam(overrides, list);
			if (typesParam) url.searchParams.set('types', typesParam);
			else url.searchParams.delete('types');
			window.location.href = url.toString();
		  }};
		
		  const renderWatchlist = () => {{
			watchEl.innerHTML = '';
			currentTickers.forEach((sym) => {{
			  const chip = document.createElement('span');
			  chip.className = 'watch-chip';
			  chip.textContent = sym;
			  const btn = document.createElement('button');
			  btn.textContent = '×';
			  btn.title = '刪除';
			  btn.addEventListener('click', () => {{
				const next = currentTickers.filter((s) => s !== sym);
				updateUrl(next);
			  }});
			  chip.appendChild(btn);
			  watchEl.appendChild(chip);
			}});
		  }};
		
		  const renderResults = (items = []) => {{
			resultsEl.innerHTML = '';
			const seen = new Set();
			const hits = (items || []).filter((k) => {{
			  if (!k.symbol || seen.has(k.symbol)) return false;
			  seen.add(k.symbol);
			  return true;
			}}).slice(0, 12);
			hits.forEach((k) => {{
			  const el = document.createElement('span');
			  el.className = 'search-item';
			  el.textContent = `${{k.symbol}} · ${{k.name}}`;
			  el.addEventListener('click', () => addTicker(k.symbol));
			  resultsEl.appendChild(el);
			}});
		  }};
		
		  const fetchYahoo = async (keyword) => {{
			const kw = normalize(keyword);
			if (!kw) return [];
			if (lastController) lastController.abort();
			lastController = new AbortController();
			try {{
			  const url = new URL('/api/search', window.location.origin);
			  url.searchParams.set('q', kw);
			  const res = await fetch(url.toString(), {{ signal: lastController.signal }});
			  const data = await res.json();
			  return (data && data.data) ? data.data : [];
			}} catch (e) {{
			  return [];
			}}
		  }};
		
		  const addTicker = (value) => {{
			const sym = normalize(value);
			if (!sym) return;
			const next = uniqueList([...currentTickers, sym]);
			updateUrl(next);
		  }};
		
		  addBtn.addEventListener('click', () => addTicker(searchInput.value));
		  clearBtn.addEventListener('click', () => updateUrl([]));
		  tipClose.addEventListener('click', () => tipDialog.close());
		  tipDialog.addEventListener('click', (e) => {{
			// 點擊遮罩區域關閉
			const rect = tipDialog.getBoundingClientRect();
			const inDialog = (e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom);
			if (!inDialog) tipDialog.close();
		  }});
		  document.querySelectorAll('.tip-icon[data-tip]').forEach((el) => {{
			el.addEventListener('click', (e) => {{
			  e.preventDefault();
			  const key = el.dataset.tip;
			  const tip = tips[key];
			  if (!tip) return;
			  tipTitle.textContent = tip.title || '說明';
			  const ul = document.createElement('ul');
			  (tip.bullets || []).forEach((b) => {{
				const li = document.createElement('li');
				li.textContent = b;
				ul.appendChild(li);
			  }});
			  tipBody.innerHTML = '';
			  tipBody.appendChild(ul);
			  tipDialog.showModal();
			}});
		  }});
		  document.querySelectorAll('.type-select').forEach((sel) => {{
			sel.addEventListener('change', (e) => {{
			  const sym = e.target.dataset.symbol;
			  const val = e.target.value;
			  if (val === 'auto') delete typeOverrides[sym];
			  else typeOverrides[sym] = val;
			  updateUrl(currentTickers, typeOverrides);
			}});
		  }});
		  const onInput = debounce(async (e) => {{
			const keyword = e.target.value;
			resultsEl.innerHTML = '';
			const external = await fetchYahoo(keyword);
			renderResults(external);
		  }}, 300);
		  searchInput.addEventListener('input', onInput);
		  searchInput.addEventListener('keydown', (e) => {{
			if (e.key === 'Enter') {{
			  e.preventDefault();
			  addTicker(searchInput.value);
			}}
		  }});
		
		  renderWatchlist();
		}})();
		</script>
	  </div>
	</body></html>
	"""

	return html

if __name__ == "__main__":
	port=int(os.environ.get("PORT","10000"))
	app.run(host="0.0.0.0", port=port, threaded=True)

