#!/usr/bin/env python3
"""
Polymarket 5m Signals — fixed for official priceToBeat
- Open: prefer official Gamma priceToBeat (Chainlink TWAP)
- Live: RTDS Chainlink + TWAP topics
- Fallback: Binance
- Multi-judge signal (lead + short mom + med mom)
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List
from collections import deque

import httpx
import websockets
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT").split(",")]
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.78"))
WINDOW_SECONDS = 300
CONFIRM_AT = 210
PRICE_HISTORY_LEN = 60
MIN_LEAD_BPS = 10.0
MIN_MOM_SHORT = 4.0
MIN_MOM_MED = 2.5

CHAINLINK_SYM = {
    "BTCUSDT": "btc/usd",
    "ETHUSDT": "eth/usd",
    "SOLUSDT": "sol/usd",
    "XRPUSDT": "xrp/usd",
}
SYM_FROM_CL = {v: k for k, v in CHAINLINK_SYM.items()}
ASSET_SLUG = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
    "SOLUSDT": "sol",
    "XRPUSDT": "xrp",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class WindowState:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.price_to_beat: Optional[float] = None
        self.window_start: Optional[datetime] = None
        self.window_unix: Optional[int] = None
        self.prices: deque = deque(maxlen=PRICE_HISTORY_LEN)
        self.last_price: Optional[float] = None
        self.signal_sent: bool = False
        self.last_signal: Optional[dict] = None
        self.last_update: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.source: str = ""
        self.official_ptb: Optional[float] = None

    def reset(self, price: float, start: datetime, window_unix: int, source: str = ""):
        self.price_to_beat = price
        self.window_start = start
        self.window_unix = window_unix
        self.prices.clear()
        self.prices.append(price)
        self.last_price = price
        self.signal_sent = False
        self.last_signal = None
        self.last_update = datetime.now(timezone.utc)
        self.last_error = None
        self.source = source
        self.official_ptb = None


state: Dict[str, WindowState] = {s: WindowState(s) for s in SYMBOLS}
signal_history: List[dict] = []
global_error: Optional[str] = None
live_chainlink: Dict[str, float] = {}   # instant
live_twap: Dict[str, float] = {}        # TWAP (preferred for lead)


def momentum_bps(prices: deque, lookback: int = 12) -> float:
    if len(prices) < lookback + 1:
        return 0.0
    return (prices[-1] - prices[-lookback]) / prices[-lookback] * 10000


def generate_signal(price_to_beat: float, current_price: float, seconds_passed: int, prices: deque) -> dict:
    if seconds_passed < CONFIRM_AT:
        return {"side": None, "confidence": 0.0, "reason": "too_early", "score": 0}

    lead_bps = ((current_price - price_to_beat) / price_to_beat) * 10000
    time_left = WINDOW_SECONDS - seconds_passed
    abs_lead = abs(lead_bps)

    if abs_lead < MIN_LEAD_BPS:
        return {"side": None, "confidence": 0.0, "reason": "too_small_lead", "score": 0, "lead_bps": round(lead_bps, 2)}

    mom_short = momentum_bps(prices, lookback=12)
    mom_med = momentum_bps(prices, lookback=25)
    side = "UP" if lead_bps > 0 else "DOWN"

    votes = 0
    if abs_lead >= MIN_LEAD_BPS:
        votes += 1
    if (side == "UP" and mom_short >= MIN_MOM_SHORT) or (side == "DOWN" and mom_short <= -MIN_MOM_SHORT):
        votes += 1
    if (side == "UP" and mom_med >= MIN_MOM_MED) or (side == "DOWN" and mom_med <= -MIN_MOM_MED):
        votes += 1

    if votes < 2:
        return {
            "side": None, "confidence": 0.0, "reason": f"jury_{votes}/3", "score": votes,
            "lead_bps": round(lead_bps, 2), "mom_short": round(mom_short, 2), "mom_med": round(mom_med, 2),
        }

    base = 0.72 + (votes - 2) * 0.06
    lead_boost = min(abs_lead * 0.011, 0.14)
    time_boost = 0.08 if time_left <= 45 else (0.04 if time_left <= 70 else 0.0)
    confidence = min(base + lead_boost + time_boost, 0.96)

    return {
        "side": side, "confidence": round(confidence, 3), "lead_bps": round(lead_bps, 2),
        "time_left": time_left, "score": votes, "mom_short": round(mom_short, 2),
        "mom_med": round(mom_med, 2), "reason": f"jury_{votes}/3",
    }


def is_strong_signal(signal: dict) -> bool:
    return (
        signal.get("side") is not None
        and signal.get("confidence", 0) >= MIN_CONFIDENCE
        and signal.get("score", 0) >= 2
    )


def get_window_start_unix() -> int:
    now = int(datetime.now(timezone.utc).timestamp())
    return now - (now % WINDOW_SECONDS)


def get_current_window_start() -> datetime:
    return datetime.fromtimestamp(get_window_start_unix(), tz=timezone.utc)


async def fetch_binance(client: httpx.AsyncClient, symbol: str) -> Optional[float]:
    for url in [
        f"https://data-api.binance.vision/api/v3/ticker/price?symbol={symbol}",
        f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
    ]:
        try:
            r = await client.get(url, timeout=4.0)
            if r.status_code == 200:
                return float(r.json()["price"])
        except Exception:
            continue
    return None


async def fetch_gamma_ptb(client: httpx.AsyncClient, asset: str, window_unix: int) -> Optional[float]:
    """Official priceToBeat from Gamma (Chainlink TWAP used by Polymarket)."""
    slug = f"{asset}-updown-5m-{window_unix}"
    try:
        r = await client.get(f"https://gamma-api.polymarket.com/events?slug={slug}", timeout=5.0)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        ev = data[0] if isinstance(data, list) else data
        meta = ev.get("eventMetadata") or {}
        ptb = meta.get("priceToBeat")
        if ptb is not None:
            return float(ptb)
    except Exception as e:
        logger.debug(f"Gamma PTB {slug}: {e}")
    return None


async def fetch_gamma_final(client: httpx.AsyncClient, asset: str, window_unix: int) -> Optional[dict]:
    """Official finalPrice + priceToBeat for a finished window."""
    slug = f"{asset}-updown-5m-{window_unix}"
    try:
        r = await client.get(f"https://gamma-api.polymarket.com/events?slug={slug}", timeout=5.0)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        ev = data[0] if isinstance(data, list) else data
        meta = ev.get("eventMetadata") or {}
        ptb = meta.get("priceToBeat")
        final = meta.get("finalPrice")
        if ptb is None:
            return None
        out = {"priceToBeat": float(ptb)}
        if final is not None:
            out["finalPrice"] = float(final)
            out["actual"] = "UP" if float(final) >= float(ptb) else "DOWN"
        return out
    except Exception:
        return None


# ====================== RTDS (instant + TWAP) ======================
async def rtds_loop():
    """Subscribe to both instant Chainlink and TWAP topics."""
    url = "wss://ws-live-data.polymarket.com"
    while True:
        try:
            async with websockets.connect(url, ping_interval=None, close_timeout=5) as ws:
                subs = []
                for cl_sym in CHAINLINK_SYM.values():
                    # instant Chainlink (fallback)
                    subs.append({
                        "topic": "crypto_prices_chainlink",
                        "type": "*",
                        "filters": json.dumps({"symbol": cl_sym}),
                    })
                    # 30s TWAP — primary for 5m markets (docs require type=update)
                    subs.append({
                        "topic": "crypto_prices_twap_thirty",
                        "type": "update",
                        "filters": json.dumps({"symbol": cl_sym}),
                    })
                    # 60s TWAP (also used on some 5m after rule changes)
                    subs.append({
                        "topic": "crypto_prices_twap_sixty",
                        "type": "update",
                        "filters": json.dumps({"symbol": cl_sym}),
                    })
                await ws.send(json.dumps({"action": "subscribe", "subscriptions": subs}))
                logger.info("RTDS subscribed to Chainlink + TWAP 30s/60s")

                async def ping():
                    while True:
                        await asyncio.sleep(5)
                        try:
                            await ws.send("PING")
                        except Exception:
                            break

                ping_task = asyncio.create_task(ping())
                try:
                    async for raw in ws:
                        if raw == "PONG":
                            continue
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        topic = (msg.get("topic") or "").lower()
                        payload = msg.get("payload") or {}
                        sym = (payload.get("symbol") or "").lower()
                        val = payload.get("value")
                        if val is None:
                            continue
                        usdt = SYM_FROM_CL.get(sym)
                        if not usdt:
                            continue
                        price = float(val)
                        if "twap" in topic:
                            live_twap[usdt] = price
                        else:
                            live_chainlink[usdt] = price
                finally:
                    ping_task.cancel()
        except Exception as e:
            logger.warning(f"RTDS reconnect in 3s: {e}")
            await asyncio.sleep(3)


async def price_loop():
    global global_error
    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
        while True:
            try:
                start_unix = get_window_start_unix()
                window_start = get_current_window_start()
                seconds_passed = int((datetime.now(timezone.utc) - window_start).total_seconds())
                any_ok = False

                for symbol in SYMBOLS:
                    ws = state[symbol]
                    asset = ASSET_SLUG[symbol]

                    # Prefer TWAP for live price (matches Polymarket resolution better)
                    price = live_twap.get(symbol) or live_chainlink.get(symbol)
                    source_now = "twap" if symbol in live_twap else ("chainlink" if symbol in live_chainlink else None)
                    if price is None:
                        price = await fetch_binance(client, symbol)
                        source_now = "binance"

                    if price is None:
                        ws.last_error = "no price"
                        continue

                    any_ok = True
                    ws.last_error = None

                    # --- new window? (use unix key — bulletproof) ---
                    if ws.window_unix is None or ws.window_unix != start_unix:
                        # close previous: mark outcomes
                        if ws.window_unix is not None and ws.last_signal:
                            prev_unix = ws.window_unix
                            official = await fetch_gamma_final(client, asset, prev_unix)
                            if official and "actual" in official:
                                actual = official["actual"]
                                final = official.get("finalPrice")
                                predicted = ws.last_signal.get("side")
                                for h in signal_history:
                                    if (h.get("window_unix") == prev_unix
                                            and h.get("symbol") == asset.upper()):
                                        h["result"] = "win" if predicted == actual else "loss"
                                        h["final"] = final
                                        h["actual"] = actual
                                        h["official"] = True
                                        break
                            elif ws.price_to_beat and ws.last_price:
                                actual = "UP" if ws.last_price >= ws.price_to_beat else "DOWN"
                                predicted = ws.last_signal.get("side")
                                for h in signal_history:
                                    if (h.get("window_unix") == prev_unix
                                            and h.get("symbol") == asset.upper()):
                                        h["result"] = "win" if predicted == actual else "loss"
                                        h["final"] = ws.last_price
                                        h["actual"] = actual
                                        h["official"] = False
                                        break

                        # open new window
                        official_ptb = await fetch_gamma_ptb(client, asset, start_unix)
                        if official_ptb is not None:
                            open_px = official_ptb
                            open_src = "gamma-official"
                        else:
                            open_px = live_twap.get(symbol) or live_chainlink.get(symbol) or price
                            open_src = "twap" if symbol in live_twap else ("chainlink" if symbol in live_chainlink else "binance-fallback")

                        ws.reset(open_px, window_start, start_unix, source=open_src)
                        if official_ptb is not None:
                            ws.official_ptb = official_ptb
                        logger.info(f"[{symbol}] NEW window {start_unix} open={open_px:.4f} ({open_src})")
                        continue

                    # still early in window → keep trying to upgrade to official Gamma PTB
                    if seconds_passed < 40 and ws.official_ptb is None:
                        official_ptb = await fetch_gamma_ptb(client, asset, start_unix)
                        if official_ptb is not None:
                            ws.official_ptb = official_ptb
                            ws.price_to_beat = official_ptb
                            ws.source = "gamma-official"
                            logger.info(f"[{symbol}] Upgraded open to official Gamma PTB={official_ptb:.4f}")

                    # update live
                    ws.prices.append(price)
                    ws.last_price = price
                    ws.last_update = datetime.now(timezone.utc)

                    if seconds_passed < CONFIRM_AT or ws.signal_sent:
                        continue

                    signal = generate_signal(ws.price_to_beat, price, seconds_passed, ws.prices)
                    if is_strong_signal(signal):
                        ws.signal_sent = True
                        entry = {
                            **signal,
                            "symbol": asset.upper(),
                            "momentum_bps": signal.get("mom_short", 0),
                            "price": price,
                            "open": ws.price_to_beat,
                            "time": datetime.now(timezone.utc).isoformat(),
                            "window_start": window_start.isoformat(),
                            "window_unix": start_unix,
                            "source": ws.source,
                        }
                        ws.last_signal = entry
                        signal_history.insert(0, entry)
                        if len(signal_history) > 50:
                            signal_history.pop()
                        logger.info(
                            f"SIGNAL {symbol} {signal['side']} conf={signal['confidence']} "
                            f"score={signal.get('score')} lead={signal.get('lead_bps')} src={ws.source}"
                        )

                global_error = None if any_ok else "Немає цін"
                await asyncio.sleep(1.5)
            except Exception as e:
                global_error = str(e)
                logger.error(f"Loop: {e}")
                await asyncio.sleep(5)


async def api_status(request):
    window_start = get_current_window_start()
    seconds_passed = int((datetime.now(timezone.utc) - window_start).total_seconds())
    time_left = max(0, WINDOW_SECONDS - seconds_passed)

    markets = {}
    for symbol, ws in state.items():
        name = symbol.replace("USDT", "")
        if ws.price_to_beat is None:
            markets[name] = {"status": "waiting", "error": ws.last_error}
            continue
        # if state is from a different window — treat as waiting
        if ws.window_unix is not None and ws.window_unix != get_window_start_unix():
            markets[name] = {"status": "waiting", "error": "window_rollover"}
            continue
        current = ws.last_price or ws.price_to_beat
        lead = ((current - ws.price_to_beat) / ws.price_to_beat) * 10000
        # live preview only after CONFIRM_AT
        if seconds_passed >= CONFIRM_AT:
            live = generate_signal(ws.price_to_beat, current, seconds_passed, ws.prices)
        else:
            live = {"side": None, "confidence": 0.0, "score": 0}
        markets[name] = {
            "open": round(ws.price_to_beat, 4),
            "current": round(current, 4),
            "lead_bps": round(lead, 2),
            "signal_sent": ws.signal_sent,
            "signal": ws.last_signal if ws.signal_sent else None,
            "live_side": live.get("side"),
            "live_confidence": live.get("confidence", 0),
            "live_score": live.get("score", 0),
            "source": ws.source,
            "official_ptb": ws.official_ptb,
            "window_unix": ws.window_unix,
            "updated": ws.last_update.isoformat() if ws.last_update else None,
        }

    wins = sum(1 for h in signal_history if h.get("result") == "win")
    losses = sum(1 for h in signal_history if h.get("result") == "loss")
    total_r = wins + losses
    win_rate = round(wins / total_r * 100, 1) if total_r else None

    end_dt = datetime.fromtimestamp(get_window_start_unix() + WINDOW_SECONDS, tz=timezone.utc)
    return JSONResponse({
        "server_time": datetime.now(timezone.utc).isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": end_dt.isoformat(),
        "window_label": f"{window_start.strftime('%H:%M')}-{end_dt.strftime('%H:%M')} UTC",
        "seconds_passed": seconds_passed,
        "time_left": time_left,
        "confirm_at": CONFIRM_AT,
        "error": global_error,
        "chainlink_live": {k.replace("USDT", ""): v for k, v in live_chainlink.items()},
        "twap_live": {k.replace("USDT", ""): v for k, v in live_twap.items()},
        "markets": markets,
        "history": signal_history[:30],
        "stats": {
            "signals": len(signal_history),
            "resolved": total_r,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
        },
    })


async def health(request):
    return JSONResponse({
        "ok": True,
        "chainlink": list(live_chainlink.keys()),
        "twap": list(live_twap.keys()),
    })


HTML_PAGE = r'''<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
<meta name="apple-mobile-web-app-capable" content="yes"/>
<title>5m Signals</title>
<style>
:root{--bg:#0a0e17;--card:#121826;--border:#1c2436;--text:#e8eef8;--muted:#7b8ba5;--up:#22c55e;--down:#ef4444;--accent:#5b6cff}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:14px;padding-bottom:40px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
h1{font-size:1.15rem;font-weight:700}
.badge{font-size:.72rem;background:#1a2235;color:var(--muted);padding:4px 10px;border-radius:999px;border:1px solid var(--border)}
.badge.ok{color:var(--up);border-color:rgba(34,197,94,.3)}
.top{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.ac{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:12px 14px}
.an{font-size:.72rem;color:var(--muted)}.ap{font-size:1.05rem;font-weight:700;font-variant-numeric:tabular-nums}
.ach{font-size:.78rem;font-weight:600}.pos{color:var(--up)}.neg{color:var(--down)}
.panel{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:14px;margin-bottom:12px}
.pt{font-size:.8rem;color:var(--muted);margin-bottom:10px;display:flex;justify-content:space-between}
.mrow{display:grid;grid-template-columns:50px 1fr 1fr 90px;gap:6px;padding:9px 0;border-bottom:1px solid var(--border);font-size:.88rem;align-items:center}
.mrow.h{color:var(--muted);font-size:.7rem;text-transform:uppercase;border:none}
.bs{padding:3px 8px;border-radius:6px;font-size:.72rem;font-weight:700}
.bu{background:rgba(34,197,94,.15);color:var(--up)}.bd{background:rgba(239,68,68,.15);color:var(--down)}.bw{background:#1a2235;color:var(--muted)}
.timer{font-size:2rem;font-weight:700;font-variant-numeric:tabular-nums;margin:6px 0 12px}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.bup,.bdn{padding:13px;border-radius:12px;border:none;font-weight:700;font-size:.9rem;text-align:center;opacity:.45;transition:.15s}
.bup{background:rgba(34,197,94,.2);color:var(--up)}.bdn{background:rgba(239,68,68,.2);color:var(--down)}
.bup.on,.bdn.on{opacity:1;transform:scale(1.02)}
.confbar{height:6px;background:#1a2235;border-radius:4px;overflow:hidden;margin-top:8px}
.confbar>div{height:100%;background:var(--accent);width:0;transition:.3s}
.live{margin-top:10px}
.lr{display:flex;justify-content:space-between;padding:5px 0;font-size:.85rem;border-bottom:1px solid var(--border)}
.hrow{display:grid;grid-template-columns:70px 40px 30px 50px 1fr;gap:4px;padding:7px 0;border-bottom:1px solid var(--border);font-size:.8rem;align-items:center}
.stats{font-size:.78rem;color:var(--muted);margin-bottom:8px}
</style>
</head>
<body>
<header>
  <h1>5m Signals</h1>
  <div id="badge" class="badge">…</div>
</header>
<div class="top" id="top"></div>
<div class="panel">
  <div class="pt"><span>Ринки</span><span id="tm">—</span></div>
  <div class="mrow h"><div>Акт</div><div>Open</div><div>Now</div><div>Сигнал</div></div>
  <div id="mkt"></div>
</div>
<div class="panel">
  <div class="pt"><span>Live BTC</span><span id="ct">—</span></div>
  <div class="timer" id="timer">—</div>
  <div class="btns">
    <div id="bu" class="bup">ВГОРУ</div>
    <div id="bd" class="bdn">ВНИЗ</div>
  </div>
  <div class="confbar"><div id="cf"></div></div>
  <div class="live" id="ll"></div>
</div>
<div class="panel">
  <div class="pt"><span>Історія</span></div>
  <div class="stats" id="st">—</div>
  <div class="hrow" style="color:var(--muted);font-size:.7rem"><div>Час</div><div>Акт</div><div>Стор</div><div>Conf</div><div>Lead</div></div>
  <div id="hist"></div>
</div>
<script>
const S=["BTC","ETH","SOL","XRP"];
function f(n,d=2){return n==null?"—":Number(n).toFixed(d)}
function render(d){
  const badge=document.getElementById("badge");
  badge.textContent=d.error?"err":"live";
  badge.className="badge"+(d.error?"":" ok");
  document.getElementById("tm").textContent=(d.window_label||"")+(d.time_left!=null?" · "+d.time_left+"s":"");
  document.getElementById("timer").textContent=d.time_left!=null?Math.floor(d.time_left/60)+":"+String(d.time_left%60).padStart(2,"0"):"—";
  let top="";
  for(const n of S){
    const m=d.markets[n];
    if(!m||m.status==="waiting"){top+=`<div class="ac"><div class="an">${n}</div><div class="ap">—</div></div>`;continue}
    const c=m.lead_bps>=0?"pos":"neg";
    const src=m.source||"";
    top+=`<div class="ac"><div class="an">${n} · ${src}</div><div class="ap">$${f(m.current,n==="XRP"?4:2)}</div><div class="ach ${c}">${m.lead_bps>=0?"+":""}${m.lead_bps} bps</div></div>`;
  }
  document.getElementById("top").innerHTML=top;
  let body="";
  for(const n of S){
    const m=d.markets[n];
    if(!m||m.status==="waiting"){body+=`<div class="mrow"><div>${n}</div><div>—</div><div>—</div><div class="bs bw">…</div></div>`;continue}
    let sh=`<span class="bs bw">очікування</span>`;
    if(m.signal_sent&&m.signal){
      const s=m.signal.side;
      const sc=m.signal.score?` ${m.signal.score}/3`:"";
      sh=`<span class="bs ${s==="UP"?"bu":"bd"}">${s==="UP"?"ВГОРУ":"ВНИЗ"} ${(m.signal.confidence*100).toFixed(0)}%${sc}</span>`;
    }else if(d.seconds_passed>=d.confirm_at&&m.live_side){
      sh=`<span class="bs bw">${m.live_side} ~${(m.live_confidence*100).toFixed(0)}%</span>`;
    }else if(d.seconds_passed<d.confirm_at){
      sh=`<span class="bs bw">чекаємо ${d.confirm_at-d.seconds_passed}s</span>`;
    }
    body+=`<div class="mrow"><div style="font-weight:600">${n}</div><div>${f(m.open,n==="XRP"?4:2)}</div><div>${f(m.current,n==="XRP"?4:2)}</div><div>${sh}</div></div>`;
  }
  document.getElementById("mkt").innerHTML=body;
  const foc=d.markets.BTC||{};
  let conf=0,side=null;
  if(foc.signal_sent&&foc.signal){side=foc.signal.side;conf=foc.signal.confidence}
  else if(foc.live_side){side=foc.live_side;conf=foc.live_confidence||0}
  document.getElementById("bu").className="bup"+(side==="UP"?" on":"");
  document.getElementById("bd").className="bdn"+(side==="DOWN"?" on":"");
  document.getElementById("ct").textContent=conf?(conf*100).toFixed(1)+"%":"—";
  document.getElementById("cf").style.width=(conf*100)+"%";
  let live="";
  for(const n of S){
    const m=d.markets[n];if(!m||m.status==="waiting")continue;
    const c=m.signal_sent&&m.signal?m.signal.confidence:(m.live_confidence||0);
    const s=m.signal_sent&&m.signal?m.signal.side:m.live_side;
    live+=`<div class="lr"><span>${n}</span><span style="color:${s==="UP"?"var(--up)":s==="DOWN"?"var(--down)":"var(--muted)"}">${s||"—"} ${c?(c*100).toFixed(0)+"%":""}</span></div>`;
  }
  document.getElementById("ll").innerHTML=live;
  const st=d.stats||{};
  document.getElementById("st").textContent=`Сигналів: ${st.signals||0}  Вирішено: ${st.resolved||0}  Win: ${st.wins||0}  Loss: ${st.losses||0}`+(st.win_rate!=null?`  WR: ${st.win_rate}%`:"");
  const h=d.history||[];
  if(h.length){
    let x="";
    for(const e of h){
      const t=e.time?new Date(e.time).toLocaleTimeString():"—";
      const res=e.result==="win"?"✓":e.result==="loss"?"✗":"";
      const rc=e.result==="win"?"pos":e.result==="loss"?"neg":"";
      const sc=e.score?` ${e.score}/3`:"";
      x+=`<div class="hrow"><div>${t}</div><div>${e.symbol||"—"}</div><div style="color:${e.side==="UP"?"var(--up)":"var(--down)"}">${e.side==="UP"?"↑":"↓"}</div><div>${(e.confidence*100).toFixed(0)}%${sc}</div><div class="${e.lead_bps>=0?"pos":"neg"}">${e.lead_bps>=0?"+":""}${e.lead_bps} <span class="${rc}">${res}</span></div></div>`;
    }
    document.getElementById("hist").innerHTML=x;
  }
}
async function refresh(){try{render(await(await fetch("/api/status")).json())}catch(e){document.getElementById("badge").textContent="offline"}}
refresh();setInterval(refresh,2000);
</script>
</body>
</html>'''


async def dashboard(request):
    return HTMLResponse(HTML_PAGE)


async def on_startup():
    asyncio.create_task(rtds_loop())
    asyncio.create_task(price_loop())
    logger.info("RTDS (TWAP+instant) + price loops started")


app = Starlette(
    routes=[
        Route("/", dashboard),
        Route("/api/status", api_status),
        Route("/health", health),
    ],
    on_startup=[on_startup],
)
