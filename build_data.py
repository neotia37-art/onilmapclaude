# -*- coding: utf-8 -*-
"""
O'NEIL MOBILE — 데이터 빌더 v2
GitHub Actions가 실행해 data/*.json 을 만든다. 모바일 앱은 이 JSON만 읽는다.

출력
  data/summary.json        시장(지수·FTD·Pulse·공포탐욕) · 환율 · 종목요약 · 알림
  data/stock/<CODE>.json   종목 상세(베이스·재무·밸류·매도신호·신뢰구간·ETF·뉴스)
"""
from __future__ import annotations
import json, math, os, re, sys, traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

KST = timezone(timedelta(hours=9))
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
os.makedirs(os.path.join(DATA, "stock"), exist_ok=True)

try:
    import requests
except Exception:
    requests = None
try:
    import FinanceDataReader as fdr
    HAS_FDR = True
except Exception:
    HAS_FDR = False
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False
try:
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        from pykrx import stock as krx
    HAS_KRX = True
except Exception:
    krx = None
    HAS_KRX = False

LOG = []


def log(sec, st, msg):
    LOG.append({"section": sec, "status": st, "msg": str(msg)[:180]})
    print(f"[{st}] {sec}: {msg}", flush=True)


def R(v, d=2):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, d)
    except Exception:
        return None


US_INDICES = [("^DJI", "다우존스"), ("^GSPC", "S&P 500"), ("^IXIC", "나스닥")]
KR_INDICES = [("KS11", "코스피"), ("KQ11", "코스닥")]
Z80 = 1.2816
RET_WIN = [("1개월", 21), ("3개월", 63), ("6개월", 126),
           ("12개월", 252), ("24개월", 504), ("36개월", 756)]

_ALIAS = {"open": "Open", "adjopen": "Open", "시가": "Open",
          "high": "High", "adjhigh": "High", "고가": "High",
          "low": "Low", "adjlow": "Low", "저가": "Low",
          "close": "Close", "종가": "Close",
          "adjclose": "AdjClose", "수정종가": "AdjClose",
          "volume": "Volume", "vol": "Volume", "거래량": "Volume"}

ETF_BRAND = ("KODEX", "TIGER", "KBSTAR", "ARIRANG", "HANARO", "KOSEF", "ACE",
             "SOL ", "RISE", "PLUS ", "TIMEFOLIO", "히어로즈", "마이다스", "WOORI")
TRUSTED_US = ["reuters", "bloomberg", "wall street journal", "wsj", "barron", "cnbc",
              "associated press", "ap news", "investor's business daily", "marketwatch",
              "financial times", "yahoo finance", "forbes", "axios", "fortune", "nikkei"]
TRUSTED_KR = ["연합뉴스", "한국경제", "매일경제", "서울경제", "이데일리", "머니투데이",
              "파이낸셜뉴스", "조선비즈", "아시아경제", "헤럴드경제", "뉴시스", "전자신문"]
NEWS_TOPICS = [
    ("실적", ["실적", "영업이익", "매출", "어닝", "가이던스", "컨센서스",
             "earnings", "revenue", "guidance", "profit", "beats", "misses", "eps"]),
    ("신제품", ["신제품", "출시", "개발", "양산", "특허", "차세대",
              "launch", "unveil", "chip", "next-gen"]),
    ("수주·계약", ["수주", "계약", "공급", "파트너", "contract", "deal", "order"]),
    ("애널리스트", ["목표주가", "투자의견", "상향", "하향", "price target", "upgrade",
                 "downgrade", "analyst", "rating"]),
    ("기관·수급", ["지분", "자사주", "배당", "stake", "buyback", "dividend"]),
    ("M&A", ["인수", "합병", "매각", "acquire", "merger", "acquisition"]),
    ("규제·소송", ["규제", "소송", "제재", "조사", "과징금", "리콜", "관세",
                "lawsuit", "regulator", "probe", "fine", "recall", "tariff", "ban"]),
    ("공급망", ["공급망", "원자재", "감산", "재고", "supply chain", "shortage"]),
]
POS_W = ["상향", "호조", "최대", "신기록", "돌파", "수주", "흑자", "급증", "확대",
         "beat", "record", "surge", "upgrade", "raise", "strong", "win", "jump"]
NEG_W = ["하향", "부진", "감소", "적자", "소송", "리콜", "제재", "축소", "지연", "우려",
         "miss", "cut", "downgrade", "fall", "plunge", "delay", "probe", "weak"]


# ════════════════════════════════════════════════════════════════════
# 시세
# ════════════════════════════════════════════════════════════════════
def as_series(x):
    return x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x


def clean(df):
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        a = [str(c[0]) for c in d.columns]
        b = [str(c[-1]) for c in d.columns]
        d.columns = a if len(set(a)) >= len(set(b)) else b
    std = {}
    for c in list(d.columns):
        s = str(c).strip()
        k = _ALIAS.get(s.lower().replace(" ", "").replace("_", "")) or _ALIAS.get(s)
        if k and k not in std:
            std[k] = c
    cc = std.get("Close") or std.get("AdjClose")
    if cc is None:
        return None
    out = pd.DataFrame(index=d.index)
    out["Close"] = pd.to_numeric(as_series(d[cc]), errors="coerce")
    for k in ("Open", "High", "Low", "Volume"):
        if k in std:
            out[k] = pd.to_numeric(as_series(d[std[k]]), errors="coerce")
    try:
        out.index = pd.to_datetime(out.index)
    except Exception:
        return None
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[out["Close"].notna() & np.isfinite(out["Close"]) & (out["Close"] > 0)]
    if len(out) == 0:
        return None
    for c in ("Open", "High", "Low"):
        out[c] = out[c].fillna(out["Close"]) if c in out.columns else out["Close"]
    out["Volume"] = out["Volume"].fillna(0) if "Volume" in out.columns else 0.0
    return out[["Open", "High", "Low", "Close", "Volume"]]


def is_kr(code):
    s = str(code).upper()
    return bool(re.fullmatch(r"[0-9A-Z]{6}", s)) and bool(re.search(r"\d", s))


def fetch(code, market=None):
    code = str(code).strip().upper()
    kr = market == "KR" or (market is None and is_kr(code))
    tries = []
    if kr:
        if HAS_FDR:
            tries.append(("fdr", lambda: fdr.DataReader(code)))
        if HAS_KRX:
            tries.append(("pykrx", lambda: krx.get_market_ohlcv_by_date(
                "20050101", datetime.now(KST).strftime("%Y%m%d"), code)))
        if HAS_YF:
            for sfx in (".KS", ".KQ"):
                tries.append((f"yf{sfx}", lambda s=sfx: yf.Ticker(code + s).history(
                    period="max", auto_adjust=True)))
    else:
        if HAS_YF:
            tries.append(("yf", lambda: yf.Ticker(code).history(period="max", auto_adjust=True)))
            tries.append(("yf-dl", lambda: yf.download(code, period="max", progress=False,
                                                       auto_adjust=True, threads=False)))
        if HAS_FDR:
            tries.append(("fdr", lambda: fdr.DataReader(code)))
    for nm, fn in tries:
        try:
            d = clean(fn())
            if d is not None and len(d) >= 60:
                return d, nm
        except Exception as e:
            log(f"시세 {code}", "재시도", f"{nm}: {type(e).__name__}")
    return None, None


def fetch_index(code, market):
    tries = []
    if market == "KR" and HAS_FDR:
        tries.append(("fdr", lambda: fdr.DataReader(code, "2018-01-01")))
    if HAS_YF:
        yq = {"KS11": "^KS11", "KQ11": "^KQ11"}.get(code, code)
        tries.append(("yf", lambda: yf.Ticker(yq).history(period="6y", auto_adjust=True)))
    if HAS_FDR:
        tries.append(("fdr2", lambda: fdr.DataReader(code, "2018-01-01")))
    for nm, fn in tries:
        try:
            d = clean(fn())
            if d is not None and len(d) > 200:
                return d, nm
        except Exception as e:
            log(f"지수 {code}", "재시도", f"{nm}: {type(e).__name__}")
    return None, None


# ════════════════════════════════════════════════════════════════════
# 시장
# ════════════════════════════════════════════════════════════════════
def distribution_days(df, lookback=25):
    d = df.tail(lookback + 1).copy()
    if float(d["Volume"].fillna(0).sum()) == 0:
        return []
    d["chg"] = d["Close"].pct_change() * 100
    last = float(d["Close"].iloc[-1])
    out = []
    for i in range(1, len(d)):
        if d["chg"].iloc[i] <= -0.2 and d["Volume"].iloc[i] > d["Volume"].iloc[i - 1]:
            if last < float(d["Close"].iloc[i]) * 1.05:
                out.append({"date": d.index[i].strftime("%Y-%m-%d"), "chg": R(d["chg"].iloc[i])})
    return out


def detect_ftd(df, min_gain=1.2, corr_pct=4.0, lookback=520):
    d = df.tail(lookback).copy()
    d["chg"] = d["Close"].pct_change() * 100
    novol = float(d["Volume"].fillna(0).sum()) == 0
    dd = (d["Close"] / d["Close"].cummax() - 1) * 100
    below = (dd <= -corr_pct).values
    if not below.any():
        return {"state": "no_correction", "max_dd": R(dd.min(), 1), "cur_dd": R(dd.iloc[-1], 1)}
    s = int(np.where(below)[0][-1])
    while s > 0 and dd.values[s - 1] < -0.5:
        s -= 1
    lp = s + int(np.argmin(d["Close"].values[s:]))
    lc = float(d["Close"].iloc[lp])
    ftd, i = None, lp + 1
    while i < len(d):
        if d["Close"].iloc[i] < lc:
            lp, lc = i, float(d["Close"].iloc[i])
            i += 1
            continue
        n = i - lp + 1
        volok = True if novol else bool(d["Volume"].iloc[i] > d["Volume"].iloc[i - 1])
        if n >= 4 and d["chg"].iloc[i] >= min_gain and volok:
            ftd = {"date": d.index[i].strftime("%Y-%m-%d"), "gain": R(d["chg"].iloc[i]),
                   "day": int(n), "low_date": d.index[lp].strftime("%Y-%m-%d"),
                   "low_close": R(lc),
                   "vol_x": (None if novol else
                             R(d["Volume"].iloc[i] / max(1.0, d["Volume"].iloc[i - 1])))}
            break
        i += 1
    if ftd is None:
        return {"state": "rally_attempt", "low_date": d.index[lp].strftime("%Y-%m-%d"),
                "low_close": R(lc), "rally_day": int(len(d) - lp),
                "max_dd": R(dd.min(), 1), "cur_dd": R(dd.iloc[-1], 1)}
    post = d.loc[pd.Timestamp(ftd["date"]):]
    ftd.update({"state": "failed" if bool((post["Close"] < ftd["low_close"]).any()) else "confirmed",
                "since": int((d.index[-1] - pd.Timestamp(ftd["date"])).days),
                "ret_since": R(float(d["Close"].iloc[-1] /
                                     d.loc[pd.Timestamp(ftd["date"]), "Close"] - 1) * 100, 1),
                "max_dd": R(dd.min(), 1), "cur_dd": R(dd.iloc[-1], 1)})
    return ftd


def index_state(name, code, df):
    c = df["Close"]
    ma50 = float(c.rolling(50).mean().iloc[-1]) if len(df) > 50 else None
    ma200 = float(c.rolling(200).mean().iloc[-1]) if len(df) > 200 else None
    px = float(c.iloc[-1])
    dds = distribution_days(df)
    tail = df.tail(260)
    return {"name": name, "code": code, "close": R(px),
            "chg": R(c.iloc[-1] / c.iloc[-2] * 100 - 100),
            "date": df.index[-1].strftime("%Y-%m-%d"), "ma50": R(ma50), "ma200": R(ma200),
            "above50": bool(ma50 and px > ma50), "above200": bool(ma200 and px > ma200),
            "ftd": detect_ftd(df), "dd": dds, "dd_count": len(dds),
            "from_high": R(px / float(c.tail(252).max()) * 100 - 100, 1),
            "series": [[d.strftime("%Y-%m-%d"), R(v)] for d, v in zip(tail.index, tail["Close"])]}


def market_pulse(states):
    n = max(1, len(states))
    conf = [s for s in states if s["ftd"].get("state") == "confirmed"]
    noc = [s for s in states if s["ftd"].get("state") == "no_correction"]
    rally = [s for s in states if s["ftd"].get("state") == "rally_attempt"]
    dd_avg = float(np.mean([s["dd_count"] for s in states])) if states else 0.0
    ab200 = sum(1 for s in states if s["above200"])
    ab50 = sum(1 for s in states if s["above50"])
    uptrend = (len(conf) + len(noc)) >= math.ceil(n / 2) and ab200 >= math.ceil(n / 2)
    if not uptrend and rally:
        state, label = "rally_attempt", "랠리 시도 중"
    elif not uptrend:
        state, label = "correction", "조정 국면"
    elif dd_avg >= 4:
        state, label = "under_pressure", "압박받는 상승세"
    else:
        state, label = "confirmed_uptrend", "확정된 상승세"

    if state in ("correction", "rally_attempt"):
        lo, hi = 0, 20
    elif state == "under_pressure":
        lo, hi = (20, 40) if dd_avg >= 5 else (40, 60)
    else:
        if ab50 == n and dd_avg <= 1:
            lo, hi = 80, 100
        elif ab50 >= math.ceil(n / 2) and dd_avg <= 3:
            lo, hi = 60, 80
        else:
            lo, hi = 40, 60
        if min([s["ftd"].get("since", 999) for s in conf], default=999) <= 2:
            lo, hi = min(lo, 40), min(hi, 60)

    reasons = []
    for s in states:
        f, st = s["ftd"], s["ftd"].get("state")
        if st == "confirmed":
            reasons.append(f'{s["name"]} · {f["date"]} FTD 확인 (저점 {f["day"]}일차 '
                           f'{f["gain"]:+.2f}%) · 이후 {f.get("ret_since", 0):+.1f}%')
        elif st == "rally_attempt":
            reasons.append(f'{s["name"]} · {f["low_date"]} 저점 후 랠리 '
                           f'{f.get("rally_day", "?")}일차 — FTD 아직 없음')
        elif st == "failed":
            reasons.append(f'{s["name"]} · {f["date"]} FTD가 저점 이탈로 무효화')
        else:
            reasons.append(f'{s["name"]} · 조정 없이 상승 지속 (최대 {f.get("max_dd", 0):.1f}%)')
        reasons.append(f'{s["name"]} · 분산일 {s["dd_count"]}개 · '
                       f'200일선 {"위" if s["above200"] else "아래"} · '
                       f'50일선 {"위" if s["above50"] else "아래"}')

    A = {
        "confirmed_uptrend": ("FTD가 확인되어 신규 매수를 시작할 수 있는 국면입니다.", [
            f"한 번에 다 넣지 말고 권장 노출도 {lo}~{hi}% 안에서 나눠 들어가세요. 첫 종목이 "
            "3~5% 오르면 다음을 추가하고, 손절이 연달아 나오면 판단이 틀린 것이니 줄입니다.",
            "돌파 종목은 거래량이 50일 평균의 1.4배 이상인지 반드시 확인하세요. "
            "거래량 없는 돌파는 가짜일 확률이 높습니다.",
            "매일 분산일 개수를 세세요. 25일 안에 6개가 쌓이면 상승세 종료 신호입니다.",
            "FTD 이후 3~4주 안에 나오는 첫 돌파들이 가장 성공률이 높습니다."]),
        "under_pressure": (f"분산일이 평균 {dd_avg:.0f}개까지 쌓여 상승세가 압박받고 있습니다.", [
            f"신규 매수를 줄이고 노출도를 {lo}~{hi}%로 낮추세요.",
            "보유 종목 중 손실 중인 것부터 정리합니다. 이익 종목은 50일선 지지를 확인하세요.",
            "새 돌파가 실패하기 시작하면(돌파 직후 피봇 아래로) 가장 빠른 경고입니다.",
            "분산일 6개 도달 시 조정으로 간주하고 신규 매수를 멈춥니다."]),
        "rally_attempt": ("저점은 나왔고 반등을 시도 중입니다. 아직 매수 시점이 아닙니다.", [
            "저점에서 4일차 이후, 지수가 크게 오르며 거래량까지 늘어난 날(FTD)이 나와야 "
            "시작 신호입니다. 그 전에 사는 것이 가장 흔한 실패입니다.",
            "지금 할 일은 매수가 아니라 관심종목 정리입니다. 조정 중에도 저점이 얕고 "
            "RS가 높은 종목을 골라 두세요.",
            "FTD가 나오면 그렇게 골라둔 종목들이 가장 먼저 베이스를 돌파합니다."]),
        "correction": ("조정 국면입니다. 신규 매수를 멈추고 현금을 지킬 때입니다.", [
            "오닐 기준 이 구간에서 산 종목은 4개 중 3개가 실패합니다.",
            "보유 종목은 손절선을 기계적으로 지킵니다. -7~8%면 이유를 찾지 말고 매도합니다.",
            "다음 상승은 반드시 '저점 → 랠리 시도 → FTD' 순서로 옵니다.",
            "조정기에 할 일은 매매가 아니라 다음 사이클 주도주 후보를 만드는 것입니다."]),
    }
    head, acts = A[state]
    return {"state": state, "label": label, "exposure": [lo, hi], "dd_avg": R(dd_avg, 1),
            "headline": head, "reasons": reasons, "actions": acts,
            "above200": f"{ab200}/{n}", "above50": f"{ab50}/{n}"}


def scale(v, lo, hi):
    if v is None:
        return None
    return float(max(0, min(100, (v - lo) / (hi - lo) * 100)))


def fear_greed(market, states, uni_r):
    comps, lead = [], None
    for s in states:
        if (market == "US" and s["name"] == "S&P 500") or (market == "KR" and s["name"] == "코스피"):
            lead = s
    lead = lead or (states[0] if states else None)
    if lead and len(lead["series"]) > 130:
        ser = [p[1] for p in lead["series"] if p[1] is not None]
        if len(ser) > 130:
            mom = (ser[-1] / float(np.mean(ser[-125:])) - 1) * 100
            comps.append(["주가 모멘텀", R(scale(mom, -6, 6), 0),
                          f'지수가 125일 평균 대비 {mom:+.1f}%', "평균 위 = 탐욕"])
            rv = pd.Series(ser).pct_change().rolling(20).std() * math.sqrt(252) * 100
            cur, hist = float(rv.iloc[-1]), rv.dropna()
            vp = float((hist < cur).mean() * 100)
            comps.append(["변동성", R(100 - vp, 0),
                          f'20일 실현변동성 {cur:.1f}% · 백분위 {vp:.0f}%', "낮을수록 탐욕"])
    if uni_r:
        br = float(np.mean([1 if x > 0 else 0 for x in uni_r]) * 100)
        comps.append(["시장 폭", R(scale(br, 25, 75), 0),
                      f'상승 종목 비율 {br:.0f}%', "많을수록 탐욕"])
    if states:
        ab = sum(1 for s in states if s["above50"])
        comps.append(["추세 강도", R(ab / len(states) * 100, 0),
                      f'50일선 상회 {ab}/{len(states)}', "모두 위면 탐욕"])
        ddm = float(np.mean([s["dd_count"] for s in states]))
        comps.append(["기관 매도(분산일)", R(scale(6 - ddm, 0, 6), 0),
                      f'분산일 평균 {ddm:.1f}개', "적을수록 탐욕"])
    vals = [c[1] for c in comps if c[1] is not None]
    if not vals:
        return None
    sc = int(round(float(np.mean(vals))))
    lab = ("극단적 공포" if sc <= 24 else "공포" if sc <= 44 else "중립" if sc <= 55
           else "탐욕" if sc <= 75 else "극단적 탐욕")
    kind = "fail" if (sc <= 24 or sc >= 76) else "warn" if (sc <= 44 or sc >= 56) else "idle"
    note = ("극단 구간입니다. 공포일 때는 FTD를 기다렸다 사고, 탐욕일 때는 신규 매수를 줄이고 "
            "보유 종목의 고점 신호를 챙기세요." if kind == "fail" else
            "극단 구간은 아닙니다. 시장 국면(FTD·분산일) 판단을 우선하세요.")
    return {"score": sc, "label": lab, "kind": kind, "comps": comps, "note": note}


# ════════════════════════════════════════════════════════════════════
# 환율
# ════════════════════════════════════════════════════════════════════
def rsi_of(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def build_fx():
    df = None
    if HAS_FDR:
        try:
            df = clean(fdr.DataReader("USD/KRW", "2015-01-01"))
        except Exception as e:
            log("환율", "재시도", f"fdr: {type(e).__name__}")
    if df is None and HAS_YF:
        try:
            df = clean(yf.Ticker("KRW=X").history(period="10y", auto_adjust=True))
        except Exception as e:
            log("환율", "실패", type(e).__name__)
    if df is None or len(df) < 300:
        log("환율", "실패", "데이터 없음")
        return None
    c = df["Close"].dropna()
    cur = float(c.iloc[-1])
    avgs, dev = {}, {}
    for lab, n in [("1개월", 21), ("6개월", 126), ("1년", 252), ("2년", 504), ("3년", 756)]:
        if len(c) > n:
            a = float(c.tail(n).mean())
            avgs[lab] = R(a, 1)
            dev[lab] = R(cur / a * 100 - 100, 1)
    w3 = c.tail(756)
    pctile = float((w3 < cur).mean() * 100) if len(w3) > 100 else None
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma60 = float(c.rolling(60).mean().iloc[-1])
    ma120 = float(c.rolling(120).mean().iloc[-1])
    trend = ("상승(원화 약세)" if ma20 > ma60 > ma120 else
             "하락(원화 강세)" if ma20 < ma60 < ma120 else "혼조")
    r = rsi_of(c)
    plans = []
    if cur > ma20 and r > 65:
        v = ("환전 보류", "fail", f"20일선 위 + RSI {r:.0f} 과열 — 단기 고점 위험")
    elif cur < ma20 and r < 40:
        v = ("소액 분할 환전", "pass", f"20일선 아래 + RSI {r:.0f} — 단기 저가권")
    else:
        v = ("중립 · 소액 분할", "warn", f"20일선 대비 {cur/ma20*100-100:+.1f}% · RSI {r:.0f}")
    plans.append(["1개월", "단기 모멘텀", v[0], v[1], v[2]])
    for lab, key, th in [("3개월", "6개월", 2), ("6개월", "1년", 3)]:
        d = dev.get(key)
        if d is None:
            continue
        vv = (("적극 환전", "pass") if d <= -th else ("분할 환전", "warn") if d <= th
              else ("환전 축소", "fail"))
        plans.append([lab, "평균 회귀", vv[0], vv[1], f"{key} 평균 대비 {d:+.1f}%"])
    if pctile is not None:
        vv = (("적극 환전", "pass") if pctile <= 30 else ("분할 환전", "warn") if pctile <= 70
              else ("최소 환전", "fail"))
        plans.append(["1년", "장기 위치", vv[0], vv[1], f"3년 백분위 {pctile:.0f}%"])
    if pctile is None:
        ratio, guide = 30, "데이터 부족 — 균등 분할"
    elif pctile <= 20:
        ratio, guide = 70, "3년래 최저권 — 필요분 대부분 지금 환전"
    elif pctile <= 40:
        ratio, guide = 50, "평균 이하 — 절반 환전 후 추가 하락 시 보충"
    elif pctile <= 60:
        ratio, guide = 30, "평균권 — 3분의 1만 환전하고 분할 대기"
    elif pctile <= 80:
        ratio, guide = 15, "평균 이상 — 최소한만 환전"
    else:
        ratio, guide = 5, "3년래 최고권 — 급하지 않으면 대기"
    trig = [[f"{lab} 평균 도달", avgs[lab], R(avgs[lab] / cur * 100 - 100, 1)]
            for lab in ("1년", "2년", "3년") if lab in avgs and avgs[lab] < cur]
    tail = c.tail(500)
    log("환율", "성공", f"{len(c)}일 · 현재 {cur:,.1f}")
    return {"cur": R(cur, 1), "avgs": avgs, "dev": dev, "pctile": R(pctile, 0),
            "ma20": R(ma20, 1), "ma60": R(ma60, 1), "ma120": R(ma120, 1), "trend": trend,
            "rsi": R(r, 0), "plans": plans, "now_ratio": ratio, "guide": guide,
            "triggers": trig, "hi52": R(float(c.tail(252).max()), 1),
            "lo52": R(float(c.tail(252).min()), 1),
            "series": [[d.strftime("%Y-%m-%d"), R(v, 1)] for d, v in zip(tail.index, tail)]}


# ════════════════════════════════════════════════════════════════════
# 베이스
# ════════════════════════════════════════════════════════════════════
def weekly(df):
    return df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                     "Close": "last", "Volume": "sum"}).dropna()


def zigzag(w, pct=8.0):
    hi_a, lo_a = w["High"].values, w["Low"].values
    piv, d = [], 0
    hi, hp, lo, lp = hi_a[0], 0, lo_a[0], 0
    for i in range(1, len(w)):
        if d >= 0 and hi_a[i] > hi:
            hi, hp = hi_a[i], i
        if d <= 0 and lo_a[i] < lo:
            lo, lp = lo_a[i], i
        if d >= 0 and lo_a[i] <= hi * (1 - pct / 100):
            piv.append((hp, "H", hi)); d = -1; lo, lp = lo_a[i], i
        elif d <= 0 and hi_a[i] >= lo * (1 + pct / 100):
            piv.append((lp, "L", lo)); d = 1; hi, hp = hi_a[i], i
    return piv


def build_bases(dfd, zp=8.0):
    w = weekly(dfd)
    if len(w) < 20:
        return []
    out, used = [], -1
    for hp, typ, hv in zigzag(w, zp):
        if typ != "H" or hp <= used:
            continue
        after = w.iloc[hp + 1:]
        if len(after) < 3:
            break
        bo = np.where(after["Close"].values > hv)[0]
        done = len(bo) > 0
        endpos = hp + 1 + int(bo[0]) if done else len(w) - 1
        seg = w.iloc[hp:endpos + 1]
        lowv = float(seg["Low"].min())
        lowpos = hp + int(np.argmin(seg["Low"].values))
        depth = (hv - lowv) / hv * 100
        weeks = endpos - hp
        if depth < 7 or depth > 72 or weeks < 3:
            continue
        sd, ld = w.index[hp], w.index[lowpos]
        ed = dfd.index[-1] if not done else w.index[endpos]
        pre = dfd.loc[:sd].tail(160)
        prior = (hv / float(pre["Low"].min()) - 1) * 100 if len(pre) > 30 else None
        rng = hv - lowv
        u = int((seg["Low"] <= lowv + rng * 0.33).sum()) / max(1, len(seg)) * 100
        li = max(1, lowpos - hp)
        lv = float(seg["Volume"].iloc[:li].mean())
        vb = (float(seg["Volume"].iloc[li:].mean()) / lv) if lv > 0 else None
        out.append({"start": sd, "low_date": ld, "end": ed, "left_high": float(hv),
                    "low": lowv, "depth": depth, "weeks": float(weeks), "completed": done,
                    "prior": prior, "u_ratio": u, "vol_bal": vb})
        used = endpos
    cnt, prev = 0, None
    for b in out:
        cnt = 1 if (prev is not None and b["low"] < prev) else cnt + 1
        b["count"] = cnt
        prev = b["low"]
    return out


def detect_handle(dfd, b):
    seg = dfd.loc[b["low_date"]:]
    if len(seg) < 8:
        return None
    lh, low = b["left_high"], b["low"]
    if float(seg["High"].max()) < low + (lh - low) * 0.60:
        return None
    p = int(np.argmax(seg["High"].values))
    h = seg.iloc[p:]
    if len(h) < 4:
        return None
    hh, hl = float(h["High"].iloc[0]), float(h["Low"].min())
    depth = (hh - hl) / hh * 100
    vma = dfd["Volume"].rolling(50).mean()
    bv = float(vma.loc[h.index[0]]) if not np.isnan(vma.loc[h.index[0]]) else 0.0
    dry = float(h["Volume"].mean()) / bv if bv > 0 else None
    half = len(h) // 2
    wedge = (float(h["Low"].iloc[half:].mean()) > float(h["Low"].iloc[:half].mean())) and depth < 4
    return {"start": h.index[0], "high": hh, "low": hl, "depth": depth, "days": len(h),
            "dry": dry, "ok_depth": 3 <= depth <= 20, "ok_pos": hl >= (lh + low) / 2,
            "wedge": wedge}


def classify(dfd, b, h):
    depth, weeks = b["depth"], b["weeks"]
    pre = dfd.loc[:b["start"]].tail(60)
    run = (b["left_high"] / float(pre["Low"].min()) - 1) * 100 if len(pre) > 15 else 0
    if run >= 60 and depth <= 25 and 2.5 <= weeks <= 8:
        return "하이 타이트 플래그"
    if depth <= 15 and weeks >= 4:
        return "플랫 베이스"
    if weeks >= 6 and h is not None:
        return "컵 위드 핸들"
    if weeks >= 6:
        return "컵 (핸들 미형성)"
    if weeks >= 3:
        return "짧은 조정"
    return "베이스 미형성"


def flaws_of(b, h):
    f = []
    if b["depth"] > 35: f.append(f'깊이 과다 {b["depth"]:.0f}%')
    if b["weeks"] < 5: f.append(f'기간 부족 {b["weeks"]:.0f}주')
    if b["u_ratio"] < 15 and b["weeks"] >= 6: f.append("V자형 · 매물 소화 부족")
    if b["vol_bal"] is not None and b["vol_bal"] < 0.8: f.append("우측 거래량 부족")
    if b["prior"] is not None and b["prior"] < 30: f.append(f'선행 상승 부족 {b["prior"]:.0f}%')
    if b["count"] >= 3: f.append(f'후기 베이스 {b["count"]}차')
    if h:
        if h["depth"] > 20: f.append(f'핸들 과대 {h["depth"]:.0f}%')
        if not h["ok_pos"]: f.append("핸들이 베이스 하단")
        if h["wedge"]: f.append("쐐기형 핸들")
        if h["dry"] is not None and h["dry"] >= 1.0: f.append("핸들 거래량 안 마름")
    return f


def anatomy(dfd, b, h):
    vma = float(dfd["Volume"].tail(250).mean()) or 1.0
    segs = []

    def seg(lbl, s, e, note):
        sl = dfd.loc[s:e]
        if len(sl) < 2:
            return
        segs.append({"label": lbl, "start": s.strftime("%Y-%m-%d"), "end": e.strftime("%Y-%m-%d"),
                     "days": len(sl), "chg": R(sl["Close"].iloc[-1] / sl["Close"].iloc[0] * 100 - 100, 1),
                     "vol": R(float(sl["Volume"].mean()) / vma), "note": note})
    seg("① 좌측 하락", b["start"], b["low_date"],
        "실망 매물이 나오는 구간. 거래량이 늘며 떨어지는 것이 정상")
    seg("② 우측 회복", b["low_date"], (h["start"] if h else b["end"]),
        "누군가 물량을 받아내는 구간. 좌측보다 거래량이 많아야 매집 근거")
    if h:
        seg("③ 핸들", h["start"], dfd.index[-1], "마지막 흔들기. 거래량이 말라야 정상")
    return segs


def kr_tick(p):
    for lim, t in [(2000, 1), (5000, 5), (20000, 10), (50000, 50),
                   (200000, 100), (500000, 500)]:
        if p < lim:
            return t
    return 1000


# ════════════════════════════════════════════════════════════════════
# 매도 신호 · 리스크
# ════════════════════════════════════════════════════════════════════
def topping(df, ma):
    out = []
    c, v = df["Close"], df["Volume"]
    px = float(c.iloc[-1])
    vma = float(v.rolling(50).mean().iloc[-1]) if float(v.sum()) > 0 else 0
    r15 = float(c.iloc[-1] / c.iloc[-16] - 1) * 100 if len(c) > 16 else None
    out.append(["클라이맥스 급등", bool(r15 is not None and r15 >= 25),
                (f'3주 {r15:+.1f}%' if r15 is not None else "—"), "25% 이상",
                "2~3주 만에 25~50% 급등은 상승 마지막 국면일 확률이 높습니다"])
    if ma.get(200):
        ext = px / ma[200] * 100 - 100
        out.append(["200일선 이격", bool(ext >= 70), f'{ext:+.1f}%', "70% 이상",
                    "200일선에서 70% 이상 벌어지면 되돌림 위험이 커집니다"])
    if len(df) > 60:
        seg = df.tail(60)
        dg = seg["Close"].pct_change() * 100
        big, rec = float(dg.max()), float(dg.iloc[-1])
        out.append(["최대 상승일 출현", bool(rec >= big * 0.999 and rec > 0),
                    f'당일 {rec:+.1f}% / 60일 최대 {big:+.1f}%', "돌파 후 최대 상승일",
                    "상승 후반의 최대 상승일은 흔히 천장 신호입니다"])
        vd = seg["Volume"].idxmax()
        vt = ((df.index[-1] - vd).days <= 5 and
              float(seg.loc[vd, "Close"]) < float(seg.loc[vd, "Open"]))
        out.append(["최대 거래량 음봉", bool(vt),
                    f'{vd:%Y-%m-%d} · {float(seg.loc[vd,"Volume"])/max(vma,1):.1f}배',
                    "최근 5일 내", "최대 거래량에 종가가 밀리면 기관이 넘기는 중입니다"])
    if len(df) > 5:
        gap = float(df["Open"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
        hi52 = float(df["High"].tail(252).max())
        out.append(["소진 갭", bool(gap >= 3 and px >= hi52 * 0.95), f'{gap:+.1f}%',
                    "고점권 +3% 갭", "긴 상승 뒤 위로 뜬 갭은 마지막 매수세일 수 있습니다"])
    w = weekly(df).tail(3)
    if len(w) >= 2:
        rng = all((w["High"] - w["Low"]) / w["Low"] * 100 > 8)
        flat = abs(float(w["Close"].iloc[-1] / w["Close"].iloc[0] - 1)) * 100 < 3
        out.append(["레일로드 트랙", bool(rng and flat), "2주 큰 변동·제자리",
                    "변동 8%+ & 순변화 3% 미만", "크게 오르내렸는데 제자리면 매수·매도가 팽팽합니다"])
    return out


def bottoming(df, base, ma):
    out = []
    c, v = df["Close"], df["Volume"]
    px = float(c.iloc[-1])
    vma = float(v.rolling(50).mean().iloc[-1]) if float(v.sum()) > 0 else 0
    if base:
        low = base["low"]
        seg = df.loc[base["low_date"]:]
        under = float(seg["Low"].min()) <= low * 1.001
        out.append(["언더컷 & 랠리", bool(under and px > low * 1.05),
                    f'저점 대비 {px/low*100-100:+.1f}%', "저점 이탈 후 5% 반등",
                    "저점을 살짝 깨고 올라오면 약한 손이 털린 좋은 신호입니다"])
    w = weekly(df).tail(3)
    if len(w) >= 3:
        cl = w["Close"].values
        mx = max(abs(cl[i] / cl[i - 1] - 1) * 100 for i in range(1, len(cl)))
        out.append(["3주 타이트", bool(mx <= 1.5), f'주간 변동 {mx:.1f}%', "1.5% 이내",
                    "주가가 좁게 붙으면 매도 물량이 말랐다는 뜻입니다"])
    if vma > 0 and len(df) > 60:
        r10 = df.tail(10)
        dry = float(r10["Volume"].mean()) / vma
        out.append(["거래량 건조", bool(dry < 0.75), f'{dry:.2f}배', "0.75배 미만",
                    "팔 사람이 없어 거래가 마르는 것은 바닥권의 전형입니다"])
        big = r10[(r10["Close"] > r10["Open"]) & (r10["Volume"] > vma * 1.5)]
        out.append(["대량 매수 출현", bool(len(big) > 0), f'최근 10일 {len(big)}일',
                    "거래량 1.5배 상승일", "거래가 마른 뒤 대량 상승일은 기관 진입 신호입니다"])
    if ma.get(50):
        out.append(["50일선 지지", bool(ma[50] * 0.98 <= px <= ma[50] * 1.05),
                    f'{px/ma[50]*100:.1f}%', "-2%~+5%",
                    "상승 종목이 50일선에서 받쳐지면 추가 매수 자리입니다"])
    return out


def sell_pressure(df, ma, tops, base):
    score, why = 0, []
    px = float(df["Close"].iloc[-1])
    vsum = float(df["Volume"].sum())
    vma = float(df["Volume"].rolling(50).mean().iloc[-1]) if vsum > 0 else 0
    n = sum(1 for t in tops if t[1])
    score += min(48, n * 16)
    if n:
        why.append(f"고점 신호 {n}건 발생")
    if ma.get(50) and px < ma[50]:
        hv = df.tail(10)
        heavy = hv[(hv["Close"] < hv["Close"].shift(1)) & (hv["Volume"] > vma * 1.3)]
        if len(heavy):
            score += 25; why.append("대량 거래를 동반한 50일선 이탈")
        else:
            score += 12; why.append("50일선 아래")
    if ma.get(200) and px < ma[200]:
        score += 20; why.append("200일선 이탈 — 장기 추세 훼손")
    d = df.tail(51)
    up = int(((d["Close"] > d["Close"].shift(1)) & (d["Volume"] > vma)).sum()) if vma else 0
    dn = int(((d["Close"] < d["Close"].shift(1)) & (d["Volume"] > vma)).sum()) if vma else 0
    if dn and up / max(1, dn) < 0.8:
        score += 12; why.append(f"분산 우위 (매집 {up} vs 분산 {dn})")
    if base and px < base["low"]:
        score += 15; why.append("베이스 저점 이탈")
    score = int(min(100, score))
    act = ("전량 매도 검토" if score >= 65 else "절반 익절 / 비중 축소" if score >= 45
           else "보유 · 경계" if score >= 25 else "보유 유지")
    kind = "fail" if score >= 65 else "warn" if score >= 25 else "pass"
    return {"score": score, "action": act, "kind": kind, "why": why}


def risk_profile(df, idx_series=None):
    lr = np.log(df["Close"]).diff().dropna()
    if len(lr) < 120:
        return None
    win = lr.tail(252)
    sd, mu = float(win.std()), float(win.mean())
    ci = []
    for lab, n in [("1일", 1), ("1주", 5), ("1개월", 21), ("3개월", 63),
                   ("6개월", 126), ("1년", 252)]:
        s, m = sd * math.sqrt(n), mu * n
        h = (df["Close"].pct_change(n).dropna() * 100)
        ok = len(h) > max(60, n * 2)
        ci.append({"기간": lab, "sigma": R(s * 100, 1),
                   "lo": R((math.exp(m - Z80 * s) - 1) * 100, 1),
                   "hi": R((math.exp(m + Z80 * s) - 1) * 100, 1),
                   "elo": R(h.quantile(0.10), 1) if ok else None,
                   "ehi": R(h.quantile(0.90), 1) if ok else None})
    s_ = df["Close"]
    fmin = s_[::-1].rolling(21, min_periods=2).min()[::-1]
    fmax = s_[::-1].rolling(61, min_periods=2).max()[::-1]
    h, l, cp = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1] / float(df["Close"].iloc[-1]) * 100)
    beta = None
    if idx_series is not None:
        try:
            j = pd.DataFrame({"a": df["Close"]}).join(pd.DataFrame({"b": idx_series}),
                                                      how="inner").dropna().tail(252)
            r1, r2 = j["a"].pct_change().dropna(), j["b"].pct_change().dropna()
            if len(r1) > 100 and r2.var() > 0:
                beta = float(np.cov(r1, r2)[0][1] / r2.var())
        except Exception:
            pass
    cc = df["Close"].tail(504)
    mdd2 = float(((cc / cc.cummax()) - 1).min() * 100)
    tv = float((df["Close"] * df["Volume"]).tail(20).mean())
    ann = sd * math.sqrt(252) * 100
    sd60 = float(lr.tail(60).std() * math.sqrt(252) * 100)
    return {"sd": R(sd * 100), "ann": R(ann, 1), "ann60": R(sd60, 1),
            "regime": ("확대" if sd60 > ann * 1.15 else "축소" if sd60 < ann * 0.85 else "안정"),
            "skew": R(float(win.skew())), "kurt": R(float(win.kurtosis()), 1), "ci": ci,
            "hit_stop": R(float(((fmin / s_ - 1) <= -0.08).mean() * 100), 0),
            "hit_tgt": R(float(((fmax / s_ - 1) >= 0.20).mean() * 100), 0),
            "atr": R(atr), "beta": R(beta), "mdd2y": R(mdd2, 1), "turnover": R(tv, 0)}


# ════════════════════════════════════════════════════════════════════
# 재무
# ════════════════════════════════════════════════════════════════════
def stmt_tab(stmt):
    mp = {"Total Revenue": "매출액", "Operating Income": "영업이익",
          "Net Income": "순이익", "Diluted EPS": "EPS"}
    rows = {}
    for eng, kor in mp.items():
        hit = [r for r in stmt.index if str(r) == eng or eng in str(r)]
        if hit:
            s = stmt.loc[hit[0]]
            rows[kor] = {(c.strftime("%Y-%m") if hasattr(c, "strftime") else str(c)):
                         (None if pd.isna(v) else float(v)) for c, v in s.items()}
    if not rows:
        return None
    t = pd.DataFrame(rows).T
    return t[sorted(t.columns)]


def growth(cur, prev):
    if cur is None or prev is None or prev == 0:
        return None
    if prev < 0:
        return 999.0 if cur > 0 else None
    return (cur / prev - 1) * 100


def fin_rows(tab, lag, last=3):
    if tab is None or tab.empty:
        return None, ""
    cols = list(tab.columns)
    use, kind = (lag, "전년 동기 대비") if len(cols) >= lag + 1 else (1, "직전 대비")
    out = []
    for i in range(max(0, len(cols) - last), len(cols)):
        rec = {"기간": cols[i]}
        for m in ("매출액", "영업이익", "순이익", "EPS"):
            if m not in tab.index:
                continue
            cur = tab.loc[m, cols[i]]
            cur = None if pd.isna(cur) else float(cur)
            rec[m] = R(cur, 4 if m == "EPS" else 0)
            prev = tab.loc[m, cols[i - use]] if i - use >= 0 else None
            prev = None if (prev is None or pd.isna(prev)) else float(prev)
            rec[m + "증감"] = R(growth(cur, prev), 1)
        out.append(rec)
    return out, kind


def _blank_fin():
    F = {k: None for k in ("per", "fper", "pbr", "psr", "peg", "roe", "opm", "npm",
                           "debt", "mktcap", "shares", "eps_ttm", "inst", "sector")}
    F.update({"q": None, "y": None, "q_kind": "", "y_kind": "", "q_eps": None,
              "q_eps_prev": None, "y_eps": None, "y_eps_prev": None, "q_sales": None,
              "src": []})
    return F


def _fill_from_yf(F, tk, price, tag):
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    if info and len(info) > 5:
        for k, key, mul in [("per", "trailingPE", 1), ("fper", "forwardPE", 1),
                            ("pbr", "priceToBook", 1),
                            ("psr", "priceToSalesTrailing12Months", 1),
                            ("peg", "trailingPegRatio", 1), ("roe", "returnOnEquity", 100),
                            ("opm", "operatingMargins", 100), ("npm", "profitMargins", 100),
                            ("debt", "debtToEquity", 1), ("mktcap", "marketCap", 1),
                            ("shares", "sharesOutstanding", 1),
                            ("inst", "heldPercentInstitutions", 100),
                            ("eps_ttm", "trailingEps", 1)]:
            if F.get(k) is None and info.get(key) is not None:
                F[k] = R(float(info[key]) * mul, 1 if mul == 100 else 2)
        F["sector"] = F["sector"] or info.get("sector")
        F["src"].append(tag)
    try:
        q = tk.quarterly_income_stmt
        if q is not None and not q.empty:
            q = q.iloc[:, ::-1]
            if F["q"] is None:
                F["q"], F["q_kind"] = fin_rows(stmt_tab(q), 4)
            er = [r for r in q.index if "Diluted EPS" in str(r)]
            if er:
                s = q.loc[er[0]].astype(float).dropna()
                if len(s) >= 5 and F["q_eps"] is None:
                    F["q_eps"], F["q_eps_prev"] = R(s.iloc[-1], 4), R(s.iloc[-5], 4)
                if F["per"] is None and price and len(s) >= 4 and float(s.iloc[-4:].sum()) > 0:
                    F["per"] = R(price / float(s.iloc[-4:].sum()))
                    F["src"].append("PER 직접 산출")
            rv = [r for r in q.index if str(r) == "Total Revenue"]
            if rv:
                s = q.loc[rv[0]].astype(float).dropna()
                if len(s) >= 5 and s.iloc[-5] > 0 and F["q_sales"] is None:
                    F["q_sales"] = R(s.iloc[-1] / s.iloc[-5] * 100 - 100, 1)
        a = tk.income_stmt
        if a is not None and not a.empty:
            a = a.iloc[:, ::-1]
            if F["y"] is None:
                F["y"], F["y_kind"] = fin_rows(stmt_tab(a), 1)
            er = [r for r in a.index if "Diluted EPS" in str(r)]
            if er:
                s = a.loc[er[0]].astype(float).dropna()
                if len(s) >= 2 and F["y_eps"] is None:
                    F["y_eps"], F["y_eps_prev"] = R(s.iloc[-1], 4), R(s.iloc[-2], 4)
        qb = tk.quarterly_balance_sheet
        if qb is not None and not qb.empty and (F["roe"] is None or F["pbr"] is None):
            qb = qb.iloc[:, ::-1]
            eq = [r for r in qb.index if "Stockholders Equity" in str(r)]
            sh = [r for r in qb.index if "Ordinary Shares Number" in str(r)]
            ni = [r for r in q.index if str(r) == "Net Income"] if q is not None else []
            if eq:
                e0 = float(qb.loc[eq[0]].dropna().iloc[-1])
                if e0 > 0 and ni and F["roe"] is None:
                    s = q.loc[ni[0]].astype(float).dropna()
                    if len(s) >= 4:
                        F["roe"] = R(float(s.iloc[-4:].sum()) / e0 * 100, 1)
                        F["src"].append("ROE 직접 산출")
                if e0 > 0 and sh and price:
                    n = float(qb.loc[sh[0]].dropna().iloc[-1])
                    if n > 0:
                        if F["pbr"] is None:
                            F["pbr"] = R(price / (e0 / n))
                        if F["mktcap"] is None:
                            F["mktcap"] = R(price * n, 0)
    except Exception:
        pass
    return F


def us_fund(code, price):
    F = _blank_fin()
    if not HAS_YF:
        return F
    return _fill_from_yf(F, yf.Ticker(code), price, "yfinance")


def kr_fund(code, price, seg=None):
    F = _blank_fin()
    if HAS_KRX:
        try:
            end = datetime.now(KST).strftime("%Y%m%d")
            st0 = (datetime.now(KST) - timedelta(days=30)).strftime("%Y%m%d")
            f = krx.get_market_fundamental(st0, end, code)
            if f is not None and not f.empty:
                F["per"], F["pbr"] = R(f["PER"].iloc[-1]), R(f["PBR"].iloc[-1])
                F["eps_ttm"] = R(f["EPS"].iloc[-1], 0)
                bps = float(f["BPS"].iloc[-1])
                if bps > 0:
                    F["roe"] = R(float(f["EPS"].iloc[-1]) / bps * 100, 1)
                F["src"].append("pykrx")
            cap = krx.get_market_cap(st0, end, code)
            if cap is not None and not cap.empty:
                F["mktcap"] = R(cap["시가총액"].iloc[-1], 0)
                F["shares"] = R(cap["상장주식수"].iloc[-1], 0)
        except Exception as e:
            log(f"재무 {code}", "재시도", f"pykrx: {type(e).__name__}")
    if HAS_YF:
        for sfx in ([".KS", ".KQ"] if seg != "KOSDAQ" else [".KQ", ".KS"]):
            try:
                tk = yf.Ticker(code + sfx)
                info = tk.info or {}
                if not info or len(info) < 5:
                    continue
                _fill_from_yf(F, tk, price, f"yfinance {code}{sfx}")
                break
            except Exception:
                continue
    return F


# ════════════════════════════════════════════════════════════════════
# ETF
# ════════════════════════════════════════════════════════════════════
def is_etf_kr(code, name):
    if HAS_KRX:
        try:
            if code in set(krx.get_etf_ticker_list(datetime.now(KST).strftime("%Y%m%d"))):
                return True
        except Exception:
            pass
    up = str(name or "").upper()
    return any(b in up for b in ETF_BRAND) or up.endswith("ETF") or "ETN" in up


def us_is_etf(code):
    if not HAS_YF:
        return False
    try:
        i = yf.Ticker(code).info or {}
        return (i.get("quoteType") or "").upper() in ("ETF", "MUTUALFUND")
    except Exception:
        return False


def etf_pack(code, market, name, df):
    P = {"expense": None, "aum": None, "deviation": None, "track_err": None, "yield": None,
         "category": None, "holdings": [], "conc": None, "rets": [], "src": []}
    c = df["Close"]
    for lab, n in RET_WIN:
        r = R(float(c.iloc[-1] / c.iloc[-n - 1] - 1) * 100, 1) if len(c) > n else None
        cagr = None
        if r is not None and n >= 252:
            cagr = R(((1 + r / 100) ** (1 / (n / 252)) - 1) * 100, 1)
        P["rets"].append({"기간": lab, "r": r, "cagr": cagr})
    if market == "KR" and HAS_KRX:
        d0 = datetime.now(KST)
        for back in range(12):
            ds = (d0 - timedelta(days=back)).strftime("%Y%m%d")
            try:
                pdf = krx.get_etf_portfolio_deposit_file(code, ds)
                if pdf is None or pdf.empty:
                    continue
                p = pdf.reset_index()
                cc = p.columns[0]
                wc = next((x for x in p.columns if "비중" in str(x)), None)
                vc = next((x for x in p.columns if "금액" in str(x)), None)
                if wc is not None:
                    w = pd.to_numeric(p[wc], errors="coerce")
                elif vc is not None:
                    w = pd.to_numeric(p[vc], errors="coerce")
                    w = w / w.sum() * 100
                else:
                    continue
                t = pd.DataFrame({"code": p[cc].astype(str).str.zfill(6), "w": w}).dropna()
                t = t[t["w"] > 0].sort_values("w", ascending=False)
                hold = []
                for r_ in t.head(20).itertuples():
                    nm2 = r_.code
                    try:
                        nm2 = krx.get_market_ticker_name(r_.code) or r_.code
                    except Exception:
                        pass
                    hold.append({"code": r_.code, "name": nm2, "w": R(r_.w)})
                P["holdings"] = hold
                P["src"].append(f"pykrx PDF {ds}")
                break
            except Exception:
                continue
        for fn, key, col in ((getattr(krx, "get_etf_price_deviation", None), "deviation", "괴리율"),
                             (getattr(krx, "get_etf_tracking_error", None), "track_err", "추적")):
            if fn is None:
                continue
            try:
                r_ = fn((d0 - timedelta(days=20)).strftime("%Y%m%d"),
                        d0.strftime("%Y%m%d"), code)
                if r_ is not None and not r_.empty:
                    cn = next((x for x in r_.columns if col in str(x)), None)
                    if cn:
                        P[key] = R(r_[cn].iloc[-1])
            except Exception:
                pass
        try:
            cap = krx.get_market_cap((d0 - timedelta(days=20)).strftime("%Y%m%d"),
                                     d0.strftime("%Y%m%d"), code)
            if cap is not None and not cap.empty:
                P["aum"] = R(cap["시가총액"].iloc[-1], 0)
        except Exception:
            pass
        if requests:
            try:
                html = requests.get(f"https://finance.naver.com/item/coinfo.naver?code={code}",
                                    timeout=8, headers={"User-Agent": "Mozilla/5.0"}).text
                m = re.search(r"총\s*보수[^0-9]{0,20}([0-9]+\.?[0-9]*)\s*%", html)
                if m:
                    P["expense"] = float(m.group(1))
            except Exception:
                pass
    elif HAS_YF:
        try:
            info = yf.Ticker(code).info or {}
            P["aum"] = R(info.get("totalAssets"), 0)
            P["category"] = info.get("category")
            P["yield"] = R((info.get("yield") or 0) * 100, 2) or None
            e = info.get("netExpenseRatio")
            P["expense"] = R(e if (e and e > 1) else (info.get("annualReportExpenseRatio") or 0) * 100)
        except Exception:
            pass
        try:
            h = yf.Ticker(code).funds_data.top_holdings
            if h is not None and not h.empty:
                t = h.reset_index()
                sym = t.columns[0]
                nc = next((x for x in t.columns if "Name" in str(x)), sym)
                wc = next((x for x in t.columns if "Percent" in str(x) or "Holding" in str(x)), None)
                if wc is not None:
                    w = pd.to_numeric(t[wc], errors="coerce")
                    w = w * 100 if float(w.max()) <= 1.5 else w
                    P["holdings"] = [{"code": str(t[sym].iloc[i]), "name": str(t[nc].iloc[i]),
                                      "w": R(w.iloc[i])}
                                     for i in range(len(t)) if w.iloc[i] == w.iloc[i]]
        except Exception:
            pass
    if P["holdings"]:
        ws = np.array([h["w"] for h in P["holdings"] if h["w"] is not None], float)
        if ws.sum() > 0:
            p = ws / ws.sum()
            eff = float(1 / (p ** 2).sum())
            P["conc"] = {"n": len(ws), "top1": R(ws[0], 1), "top5": R(ws[:5].sum(), 1),
                         "top10": R(ws[:10].sum(), 1), "hhi": R((p ** 2).sum() * 10000, 0),
                         "eff": R(eff, 1),
                         "level": ("매우 집중" if eff < 8 else "집중" if eff < 15
                                   else "적정 분산" if eff < 40 else "고도 분산")}
    items = [
        ["총보수", (f'{P["expense"]:.2f}%' if P["expense"] is not None else "미수집"),
         "0.5% 이하", bool(P["expense"] is not None and P["expense"] <= 0.5),
         "매년 자산에서 차감. 같은 지수면 싼 쪽이 정답"],
        ["순자산", (f'{P["aum"]/1e8:,.0f}억' if P["aum"] and market == "KR"
                 else f'${P["aum"]/1e9:,.2f}B' if P["aum"] else "미수집"),
         "1,000억 이상", bool(P["aum"] and (P["aum"] >= 1e11 if market == "KR" else P["aum"] >= 5e8)),
         "작으면 유동성·상장폐지 위험"],
        ["괴리율", (f'{P["deviation"]:+.2f}%' if P["deviation"] is not None else "미수집"),
         "±0.5% 이내", bool(P["deviation"] is not None and abs(P["deviation"]) <= 0.5),
         "시장가와 순자산가치의 차이"],
        ["추적오차", (f'{P["track_err"]:.2f}%' if P["track_err"] is not None else "미수집"),
         "1.0% 이하", bool(P["track_err"] is not None and P["track_err"] <= 1.0),
         "기초지수 추종 정확도"],
        ["분산도", (f'실효 {P["conc"]["eff"]}종목' if P["conc"] else "미수집"),
         "8종목 이상", bool(P["conc"] and P["conc"]["eff"] >= 8),
         "실효 종목수가 적으면 개별 종목처럼 움직임"],
    ]
    got = [i for i in items if i[1] != "미수집"]
    sc = int(round(sum(1 for i in got if i[3]) / max(1, len(got)) * 100))
    P["verdict"] = {"items": items, "score": sc,
                    "grade": "양호" if sc >= 75 else "보통" if sc >= 50 else "주의",
                    "kind": "pass" if sc >= 75 else "warn" if sc >= 50 else "fail"}
    return P


# ════════════════════════════════════════════════════════════════════
# 뉴스
# ════════════════════════════════════════════════════════════════════
def build_news(code, market, limit=12):
    items = []
    if HAS_YF:
        for c in ([code] if market != "KR" else [code + ".KS", code + ".KQ"]):
            try:
                raw = yf.Ticker(c).news or []
            except Exception:
                raw = []
            for n in raw[:25]:
                ct = n.get("content") if isinstance(n.get("content"), dict) else n
                title = ct.get("title") or n.get("title")
                prov = ct.get("provider") or {}
                pub = ((prov.get("displayName") if isinstance(prov, dict) else None)
                       or n.get("publisher") or "")
                url = n.get("link")
                if not url:
                    cu = ct.get("canonicalUrl") or ct.get("clickThroughUrl") or {}
                    url = cu.get("url") if isinstance(cu, dict) else None
                ts = n.get("providerPublishTime") or ct.get("pubDate")
                when = None
                try:
                    when = (datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                            if isinstance(ts, (int, float))
                            else pd.to_datetime(ts).strftime("%Y-%m-%d"))
                except Exception:
                    pass
                if title and pub:
                    items.append({"title": str(title), "pub": str(pub), "url": url, "when": when})
            if items:
                break
    trusted = TRUSTED_KR if market == "KR" else TRUSTED_US
    keep, seen = [], set()
    for it in items:
        if not any(t in it["pub"].lower() for t in trusted):
            continue
        k = it["title"][:40]
        if k in seen:
            continue
        seen.add(k)
        low = it["title"].lower()
        it["topics"] = [nm for nm, kws in NEWS_TOPICS
                        if any(k2.lower() in low for k2 in kws)] or ["일반"]
        pos = sum(1 for w in POS_W if w.lower() in low)
        neg = sum(1 for w in NEG_W if w.lower() in low)
        it["tone"] = "긍정" if pos > neg else "부정" if neg > pos else "중립"
        keep.append(it)
    keep.sort(key=lambda x: x["when"] or "", reverse=True)
    return keep[:limit]


# ════════════════════════════════════════════════════════════════════
# 종합
# ════════════════════════════════════════════════════════════════════
def analyze(code, name, df, market, uni_r, idx_series, pulse_ok, seg=None):
    px = float(df["Close"].iloc[-1])
    c = df["Close"]
    A = {"code": code, "name": name, "market": market, "price": R(px),
         "chg": R(c.iloc[-1] / c.iloc[-2] * 100 - 100),
         "date": df.index[-1].strftime("%Y-%m-%d"),
         "first": df.index[0].strftime("%Y-%m-%d"), "bars": len(df)}
    ma = {n: (float(c.rolling(n).mean().iloc[-1]) if len(df) >= n else None)
          for n in (50, 150, 200)}
    A["ma"] = {str(k): R(v) for k, v in ma.items()}
    A["ma_ok"] = bool(ma[200] and px > ma[50] > ma[150] > ma[200])
    A["from_high"] = R(px / float(df["High"].tail(252).max()) * 100 - 100, 1)
    A["rets"] = {k: (R(c.iloc[-1] / c.iloc[-n - 1] * 100 - 100, 1) if len(c) > n else None)
                 for k, n in [("r1", 21), ("r3", 63), ("r6", 126), ("r12", 252)]}
    vsum = float(df["Volume"].sum())
    vma = float(df["Volume"].rolling(50).mean().iloc[-1]) if vsum > 0 else 0
    A["vol_x"] = R(float(df["Volume"].iloc[-1]) / vma) if vma > 0 else None
    d50 = df.tail(51)
    up = int(((d50["Close"] > d50["Close"].shift(1)) & (d50["Volume"] > vma)).sum()) if vma else 0
    dn = int(((d50["Close"] < d50["Close"].shift(1)) & (d50["Volume"] > vma)).sum()) if vma else 0
    ratio = up / max(1, dn)
    A["acc"] = {"up": up, "dn": dn,
                "grade": "A" if ratio >= 2 else "B" if ratio >= 1.3 else "C" if ratio >= 0.8 else "D"}

    bases = build_bases(df)
    base_obj = None
    if bases:
        b = bases[-1]
        h = detect_handle(df, b) if not b["completed"] else None
        if h and h["ok_depth"] and h["ok_pos"] and not h["wedge"]:
            pv_raw, pv_src = h["high"], "핸들 고점"
        else:
            pv_raw, pv_src = b["left_high"], "좌측 고점"
        pivot = (float(math.ceil(pv_raw * 1.001 / kr_tick(pv_raw)) * kr_tick(pv_raw))
                 if market == "KR" else round(pv_raw + 0.10, 2))
        gap = px / pivot * 100 - 100
        since = (df.index[-1] - b["end"]).days
        if b["completed"] and since > 5:
            stage, kind = "직전 베이스 돌파 완료", "fail"
        elif gap < -10:
            stage, kind = "베이스 형성 중", "idle"
        elif gap < -1:
            stage, kind = "피봇 접근 · 돌파 대기", "warn"
        elif gap <= 5:
            stage, kind = "매수 가능 구간", "pass"
        else:
            stage, kind = "연장 · 추격 금지", "fail"
        base_obj = b
        A["base"] = {"type": classify(df, b, h), "start": b["start"].strftime("%Y-%m-%d"),
                     "low_date": b["low_date"].strftime("%Y-%m-%d"), "depth": R(b["depth"], 1),
                     "weeks": int(round(b["weeks"])), "count": b["count"],
                     "left_high": R(b["left_high"]), "low": R(b["low"]),
                     "flaws": flaws_of(b, h), "pivot_src": pv_src, "u_ratio": R(b["u_ratio"], 0),
                     "vol_bal": R(b["vol_bal"]),
                     "handle": (None if not h else
                                {"depth": R(h["depth"], 1), "days": h["days"], "low": R(h["low"]),
                                 "dry": R(h["dry"]), "start": h["start"].strftime("%Y-%m-%d")}),
                     "anatomy": anatomy(df, b, h)}
        A["history"] = [{"count": x["count"], "start": x["start"].strftime("%Y-%m-%d"),
                         "end": x["end"].strftime("%Y-%m-%d"), "depth": R(x["depth"], 1),
                         "weeks": int(round(x["weeks"])), "done": x["completed"]}
                        for x in bases[-8:]]
        stop_ref = (h["low"] if h else b["low"]) * 0.99
        A.update({"pivot": R(pivot), "gap": R(gap, 1), "stage": stage, "kind": kind,
                  "buy_hi": R(pivot * 1.05), "stop": R(max(pivot * 0.92, stop_ref)),
                  "t1": R(pivot * 1.20), "t2": R(pivot * 1.25)})
    else:
        A.update({"base": None, "history": [], "stage": "베이스 미형성", "kind": "idle",
                  "pivot": None, "gap": None})

    if uni_r and all(A["rets"].get(k) is not None for k in ("r3", "r6", "r12")):
        sc = 0.4 * A["rets"]["r3"] + 0.3 * A["rets"]["r6"] + 0.3 * A["rets"]["r12"]
        A["rs"] = min(99, int(round(float((np.array(uni_r) < sc).mean() * 98))) + 1)
    else:
        A["rs"] = None

    tops = topping(df, ma)
    A["top_sig"] = tops
    A["bot_sig"] = bottoming(df, base_obj, ma)
    A["sell"] = sell_pressure(df, ma, tops, base_obj)
    A["risk"] = risk_profile(df, idx_series)

    A["is_etf"] = bool(is_etf_kr(code, name) if market == "KR" else us_is_etf(code))
    if A["is_etf"]:
        A["etf"], A["fin"] = etf_pack(code, market, name, df), None
    else:
        A["etf"] = None
        A["fin"] = kr_fund(code, px, seg) if market == "KR" else us_fund(code, px)

    A["news"] = build_news(code, market)
    F = A.get("fin") or {}
    q_g = growth(F.get("q_eps"), F.get("q_eps_prev"))
    y_g = growth(F.get("y_eps"), F.get("y_eps_prev"))
    A["growth"] = {"q": R(q_g, 1), "y": R(y_g, 1), "sales": F.get("q_sales")}

    s = 0
    if pulse_ok: s += 15
    if A["rs"] and A["rs"] >= 80: s += 15
    if A["from_high"] is not None and A["from_high"] >= -15 and A["ma_ok"]: s += 10
    if A["acc"]["grade"] in ("A", "B"): s += 10
    if A["base"] and len(A["base"]["flaws"]) <= 1 and A["base"]["count"] <= 2 \
            and A["base"]["weeks"] >= 5: s += 15
    if A["kind"] == "pass": s += 10
    if q_g is not None and q_g >= 25: s += 15
    if y_g is not None and y_g >= 25: s += 10
    if A["is_etf"]:
        s = min(100, int(s * 1.35))          # ETF는 C·A 항목이 없으므로 보정
    A["score"] = s
    A["grade"] = "A" if s >= 80 else "B" if s >= 65 else "C" if s >= 50 else "D"

    A["checklist"] = [
        ["시장이 확정 상승세인가", bool(pulse_ok), "시장"],
        ["분기 EPS +25% 이상", bool(q_g is not None and q_g >= 25), "재무"],
        ["연간 EPS +25% 이상", bool(y_g is not None and y_g >= 25), "재무"],
        ["52주 고점 -15% 이내 + 정배열",
         bool(A["from_high"] is not None and A["from_high"] >= -15 and A["ma_ok"]), "지표"],
        ["거래량 매집 우위", A["acc"]["grade"] in ("A", "B"), "지표"],
        ["RS 80 이상", bool(A["rs"] and A["rs"] >= 80), "지표"],
        ["결함 없는 1~2차 베이스",
         bool(A["base"] and len(A["base"]["flaws"]) <= 1 and A["base"]["count"] <= 2), "베이스"],
        ["매수 구간(피봇~+5%)", A["kind"] == "pass", "베이스"],
        ["매도 압력 45 미만", A["sell"]["score"] < 45, "매도신호"],
        ["유동성 충분", bool(A["risk"] and A["risk"]["turnover"] and
                        A["risk"]["turnover"] > (1e9 if market == "KR" else 2e7)), "리스크"],
    ]
    tail = df.tail(300)
    A["series"] = [[d.strftime("%Y-%m-%d"), R(v), int(vv or 0)]
                   for d, v, vv in zip(tail.index, tail["Close"], tail["Volume"].fillna(0))]
    return A


# ════════════════════════════════════════════════════════════════════
def load_watchlist():
    p = os.path.join(DATA, "watchlist.json")
    if os.path.exists(p):
        try:
            j = json.load(open(p, encoding="utf-8"))
            items = j.get("items", []) if isinstance(j, dict) else j
            out = []
            for x in items:
                if isinstance(x, str):
                    out.append({"code": x})
                elif isinstance(x, dict) and x.get("code"):
                    out.append(x)
            if out:
                return out
        except Exception as e:
            log("watchlist", "실패", e)
    return [{"code": "NVDA"}, {"code": "ANET"}, {"code": "005930"}, {"code": "000660"}]


def universe_returns(market):
    try:
        if market == "KR" and HAS_KRX:
            end = datetime.now(KST).strftime("%Y%m%d")
            r = {}
            for k, days in [("r3", 92), ("r6", 183), ("r12", 365)]:
                ch = krx.get_market_price_change(
                    (datetime.now(KST) - timedelta(days=days)).strftime("%Y%m%d"), end,
                    market="ALL")
                r[k] = ch["등락률"]
            d = pd.DataFrame(r).dropna()
            return (0.4 * d["r3"] + 0.3 * d["r6"] + 0.3 * d["r12"]).tolist()
        if market == "US" and HAS_YF:
            syms = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO LLY JPM V UNH XOM MA JNJ PG COST "
                    "HD ABBV WMT NFLX CRM BAC KO PEP AMD ADBE TMO LIN MRK CVX ACN MCD CSCO ABT "
                    "ORCL DHR WFC TXN INTU IBM QCOM NOW GE CAT AMGN PFE UNP ISRG SPGI RTX BKNG "
                    "HON UBER PGR LOW BLK SYK AMAT ELV TJX VRTX MDT LMT ADI PLD REGN SCHW MU C "
                    "BSX CB ETN KLAC PANW SNPS CDNS MRVL CRWD FTNT ANET DELL ON MCHP NXPI TER "
                    "FSLR GEV VRT COIN SHOP ABNB DASH SNOW DDOG NET ZS TTD PLTR RBLX TEAM MDB "
                    "CEG VST NRG PWR NEE DUK SO AEP SLB HAL OXY COP EOG PSX MPC VLO DE BA GD").split()
            px = yf.download(syms, period="400d", progress=False, auto_adjust=True,
                             threads=False)["Close"].dropna(how="all", axis=1)
            r3 = (px.iloc[-1] / px.iloc[-64] - 1) * 100
            r6 = (px.iloc[-1] / px.iloc[-127] - 1) * 100
            r12 = (px.iloc[-1] / px.iloc[-253] - 1) * 100
            return (0.4 * r3 + 0.3 * r6 + 0.3 * r12).dropna().tolist()
    except Exception as e:
        log(f"RS 유니버스 {market}", "실패", e)
    return None


def name_of(code, market):
    if market == "KR":
        if HAS_KRX:
            for fn in (getattr(krx, "get_market_ticker_name", None),
                       getattr(krx, "get_etf_ticker_name", None)):
                if fn is None:
                    continue
                try:
                    n = fn(code)
                    if n and str(n) != code:
                        return str(n)
                except Exception:
                    pass
    elif HAS_YF:
        try:
            i = yf.Ticker(code).info or {}
            return i.get("shortName") or i.get("longName") or code
        except Exception:
            pass
    return code


def kr_seg(code):
    if HAS_KRX:
        try:
            d = datetime.now(KST).strftime("%Y%m%d")
            for mk in ("KOSPI", "KOSDAQ"):
                if code in set(krx.get_market_ticker_list(d, market=mk)):
                    return mk
        except Exception:
            pass
    return "KOSPI"


def main():
    now = datetime.now(KST)
    market_out, pulses, idx_series, uni = {}, {}, {}, {}
    for mk, lst in (("US", US_INDICES), ("KR", KR_INDICES)):
        states = []
        for code, nm in lst:
            df, src = fetch_index(code, mk)
            if df is None:
                log(f"지수 {nm}", "실패", "모든 소스 실패")
                continue
            states.append(index_state(nm, code, df))
            if mk not in idx_series:
                idx_series[mk] = df["Close"]
            log(f"지수 {nm}", "성공", f'{src} · {len(df)}일 · {float(df["Close"].iloc[-1]):,.2f}')
        if states:
            uni[mk] = universe_returns(mk)
            pulses[mk] = market_pulse(states)
            market_out[mk] = {"indices": states, "pulse": pulses[mk],
                              "fng": fear_greed(mk, states, uni.get(mk))}
        else:
            log(f"{mk} 시장", "실패", "지수를 하나도 못 불러왔습니다")

    fx = build_fx()
    wl = load_watchlist()
    items, details = [], {}
    for w in wl:
        code = str(w.get("code", "")).strip().upper()
        if not code:
            continue
        mk = w.get("market") or ("KR" if is_kr(code) else "US")
        df, src = fetch(code, mk)
        if df is None:
            log(f"종목 {code}", "실패", "시세 없음")
            items.append({"code": code, "market": mk, "name": w.get("name") or code,
                          "error": "시세를 가져오지 못했습니다"})
            continue
        if mk not in uni:
            uni[mk] = universe_returns(mk)
        nm = w.get("name") or name_of(code, mk)
        seg = kr_seg(code) if mk == "KR" else None
        try:
            a = analyze(code, nm, df, mk, uni.get(mk), idx_series.get(mk),
                        pulses.get(mk, {}).get("state") == "confirmed_uptrend", seg)
        except Exception as e:
            log(f"종목 {code}", "실패", f"분석 오류 {type(e).__name__}: {e}")
            items.append({"code": code, "market": mk, "name": nm,
                          "error": f"분석 오류 ({type(e).__name__})"})
            continue
        a["src"], a["seg"] = src, seg
        details[code] = a
        keep = ("code", "name", "market", "price", "chg", "date", "pivot", "gap", "stage",
                "kind", "rs", "score", "grade", "from_high", "buy_hi", "stop", "t1", "t2",
                "is_etf")
        row = {k: a.get(k) for k in keep}
        row["base_type"] = (a.get("base") or {}).get("type")
        row["base_count"] = (a.get("base") or {}).get("count")
        row["flaws"] = len((a.get("base") or {}).get("flaws", []))
        row["sell"] = a["sell"]["score"]
        row["sell_act"] = a["sell"]["action"]
        row["q_growth"] = a["growth"]["q"]
        items.append(row)
        log(f"종목 {code}", "성공", f'{src} · {a["price"]} · {a["stage"]} · {a["score"]}점')

    for code, a in details.items():
        with open(os.path.join(DATA, "stock", f"{code}.json"), "w", encoding="utf-8") as f:
            json.dump(a, f, ensure_ascii=False, separators=(",", ":"), default=str)

    alerts = []
    for it in items:
        if it.get("error"):
            continue
        nm = it.get("name") or it["code"]
        if it.get("kind") == "pass":
            alerts.append({"code": it["code"], "name": nm, "level": "buy",
                           "title": f"{nm} 매수 가능 구간",
                           "body": f'피봇 {it["pivot"]} 대비 {it["gap"]:+.1f}% · '
                                   f'스코어 {it["score"]}점 · RS {it.get("rs") or "—"}'})
        elif it.get("kind") == "warn" and it.get("gap") is not None and it["gap"] >= -3:
            alerts.append({"code": it["code"], "name": nm, "level": "near",
                           "title": f"{nm} 피봇 임박",
                           "body": f'피봇 {it["pivot"]}까지 {it["gap"]:+.1f}% — 돌파 시 거래량 1.4배 확인'})
        if (it.get("sell") or 0) >= 65:
            alerts.append({"code": it["code"], "name": nm, "level": "sell",
                           "title": f"{nm} 매도 압력 높음",
                           "body": f'매도 압력 {it["sell"]}점 · {it.get("sell_act")}'})

    out = {"asof": now.isoformat(timespec="seconds"),
           "asof_kr": now.strftime("%Y-%m-%d %H:%M"),
           "markets": market_out, "fx": fx, "items": items, "alerts": alerts, "log": LOG}
    with open(os.path.join(DATA, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"), default=str)
    print(f"\n완료 · 종목 {len(items)} · 알림 {len(alerts)} · {now:%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
