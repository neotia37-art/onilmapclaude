# -*- coding: utf-8 -*-
"""OpenDart 기반 한국 주식 재무 (CANSLIM C·A). build_data.krfund 가 사용."""
from __future__ import annotations
import json, math, os, re
from datetime import datetime, timedelta, timezone

try:
    import requests
except Exception:
    requests = None

KST = timezone(timedelta(hours=9))
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

DART_KEY = (
    os.environ.get("DART_API_KEY")
    or os.environ.get("OPENDART_API_KEY")
    or ""
).strip()
_CORP_MAP = None


def _R(v, d=2):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, d)
    except Exception:
        return None


def _log(sec, st, msg, LOG=None):
    print(f"[{st}] {sec}: {msg}", flush=True)
    if LOG is not None:
        LOG.append({"section": sec, "status": st, "msg": str(msg)[:180]})


def _parse_amt(s):
    if s is None or s == "" or s == "-":
        return None
    try:
        return float(str(s).replace(",", ""))
    except Exception:
        return None


def load_corp_map(LOG=None):
    global _CORP_MAP
    if _CORP_MAP is not None:
        return _CORP_MAP
    _CORP_MAP = {}
    os.makedirs(DATA, exist_ok=True)
    cache = os.path.join(DATA, "corp_codes.json")
    if os.path.isfile(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                _CORP_MAP = {str(k).zfill(6): str(v) for k, v in json.load(f).items()}
            _log("OpenDart", "성공", f"corp_codes 캐시 {len(_CORP_MAP)}건", LOG)
        except Exception as e:
            _log("OpenDart", "재시도", f"캐시 로드: {e}", LOG)
    if not DART_KEY or not requests:
        return _CORP_MAP
    if len(_CORP_MAP) < 100:
        try:
            import zipfile, io, xml.etree.ElementTree as ET
            r = requests.get(
                "https://opendart.fss.or.kr/api/corpCode.xml",
                params={"crtfc_key": DART_KEY}, timeout=180)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            name = z.namelist()[0]
            root = ET.fromstring(z.read(name))
            m = {}
            for el in root.findall("list"):
                sc = (el.findtext("stock_code") or "").strip()
                cc = (el.findtext("corp_code") or "").strip()
                if sc and cc:
                    m[sc.zfill(6)] = cc
            if m:
                _CORP_MAP = m
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump(m, f, ensure_ascii=False)
                _log("OpenDart", "성공", f"corp_codes 다운로드 {len(m)}건", LOG)
        except Exception as e:
            _log("OpenDart", "실패", f"corp_codes: {type(e).__name__}: {e}", LOG)
    return _CORP_MAP


def dart_corp_code(stock_code, LOG=None):
    m = load_corp_map(LOG)
    return m.get(str(stock_code).zfill(6))


def dart_fetch_accounts(corp_code, year, reprt_code, LOG=None):
    if not DART_KEY or not requests:
        return None
    try:
        r = requests.get(
            "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json",
            params={"crtfc_key": DART_KEY, "corp_code": corp_code,
                    "bsns_year": str(year), "reprt_code": reprt_code},
            timeout=25)
        j = r.json()
        if j.get("status") != "000":
            return None
        rows = j.get("list") or []
        for prefer in ("CFS", "OFS"):
            out = {}
            for x in rows:
                if x.get("fs_div") != prefer or x.get("sj_div") != "IS":
                    continue
                nm = x.get("account_nm") or ""
                th = _parse_amt(x.get("thstrm_amount"))
                fr = _parse_amt(x.get("frmtrm_amount"))
                if "매출액" in nm and "매출원가" not in nm and "순매출" not in nm:
                    out.setdefault("sales", th)
                    out.setdefault("sales_prev", fr)
                elif nm == "영업이익" or (nm.startswith("영업이익") and "비용" not in nm):
                    out.setdefault("op", th)
                    out.setdefault("op_prev", fr)
                elif "당기순이익" in nm and "비지배" not in nm:
                    out.setdefault("ni", th)
                    out.setdefault("ni_prev", fr)
            if out:
                out["fs"] = prefer
                out["year"] = year
                out["reprt"] = reprt_code
                return out
    except Exception as e:
        _log(f"OpenDart {corp_code}", "재시도", f"{year}/{reprt_code}: {type(e).__name__}", LOG)
    return None


def dart_latest_fin(stock_code, LOG=None):
    cc = dart_corp_code(stock_code, LOG)
    if not cc:
        return None
    now = datetime.now(KST)
    years = [now.year, now.year - 1, now.year - 2]
    q_order = [("11014", "3Q"), ("11012", "H1"), ("11013", "1Q")]
    y_order = [("11011", "FY")]
    q = y = None
    for yr in years:
        if q is None:
            for rc, lab in q_order:
                d = dart_fetch_accounts(cc, yr, rc, LOG)
                if d and d.get("ni") is not None:
                    d["label"] = f"{yr} {lab}"
                    q = d
                    break
        if y is None:
            for rc, lab in y_order:
                d = dart_fetch_accounts(cc, yr, rc, LOG)
                if d and d.get("ni") is not None:
                    d["label"] = f"{yr} {lab}"
                    y = d
                    break
        if q and y:
            break
    return {"corp_code": cc, "q": q, "y": y}


def enrich_kr_fund(F, code, price, LOG=None):
    """기존 F dict에 OpenDart 분기/연간 YoY를 채운다."""
    if not DART_KEY:
        return F
    try:
        d = dart_latest_fin(code, LOG)
        if not d:
            return F
        F["src"].append("OpenDart")
        q, y = d.get("q"), d.get("y")

        def g(a, b):
            if a is None or b is None or b == 0:
                return None
            if b < 0:
                return 999.0 if a > 0 else None
            return _R((a / b - 1) * 100, 1)

        if q:
            F["q_eps"] = _R(q.get("ni"), 0)
            F["q_eps_prev"] = _R(q.get("ni_prev"), 0)
            if q.get("sales") and q.get("sales_prev") and q["sales_prev"]:
                F["q_sales"] = _R(q["sales"] / q["sales_prev"] * 100 - 100, 1)
            F["q"] = [{
                "기간": q.get("label", "분기"),
                "매출액": _R(q.get("sales"), 0),
                "매출액증감": g(q.get("sales"), q.get("sales_prev")),
                "영업이익": _R(q.get("op"), 0),
                "영업이익증감": g(q.get("op"), q.get("op_prev")),
                "순이익": _R(q.get("ni"), 0),
                "순이익증감": g(q.get("ni"), q.get("ni_prev")),
                "EPS": None,
                "EPS증감": g(q.get("ni"), q.get("ni_prev")),
            }]
            F["q_kind"] = q.get("label", "OpenDart 분기")
            if q.get("op") is not None and q.get("sales"):
                F["opm"] = _R(q["op"] / q["sales"] * 100, 1)
            if q.get("ni") is not None and q.get("sales"):
                F["npm"] = _R(q["ni"] / q["sales"] * 100, 1)
        if y:
            F["y_eps"] = _R(y.get("ni"), 0)
            F["y_eps_prev"] = _R(y.get("ni_prev"), 0)
            F["y"] = [{
                "기간": y.get("label", "연간"),
                "매출액": _R(y.get("sales"), 0),
                "매출액증감": g(y.get("sales"), y.get("sales_prev")),
                "영업이익": _R(y.get("op"), 0),
                "영업이익증감": g(y.get("op"), y.get("op_prev")),
                "순이익": _R(y.get("ni"), 0),
                "순이익증감": g(y.get("ni"), y.get("ni_prev")),
                "EPS": None,
                "EPS증감": g(y.get("ni"), y.get("ni_prev")),
            }]
            F["y_kind"] = y.get("label", "OpenDart 연간")
    except Exception as e:
        _log(f"재무 {code}", "재시도", f"OpenDart: {type(e).__name__}: {e}", LOG)
    return F
