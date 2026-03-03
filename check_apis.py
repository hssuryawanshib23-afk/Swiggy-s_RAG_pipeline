"""
check_apis.py — Check Groq & Gemini API status, remaining tokens, and reset times.
Run: python check_apis.py
"""

import os
import time
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def header(title):
    print(f"\n{BOLD}{CYAN}{'═'*52}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*52}{RESET}")

def ok(msg):   print(f"  {GREEN}✔  {msg}{RESET}")
def err(msg):  print(f"  {RED}✘  {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠  {msg}{RESET}")
def info(msg): print(f"  {DIM}{msg}{RESET}")

# ─────────────────────────────────────────────
# GROQ
# ─────────────────────────────────────────────
def check_groq():
    header("GROQ  (llama-3.3-70b-versatile)")

    if not GROQ_API_KEY:
        err("GROQ_API_KEY not set in .env")
        return

    # 1. List models — lightweight call, just proves the key works
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        timeout=10,
    )

    if resp.status_code == 401:
        err("Invalid API key (401 Unauthorized)")
        return
    if resp.status_code != 200:
        err(f"Models endpoint returned HTTP {resp.status_code}")
        info(resp.text[:300])
        return

    ok("API key is valid — Groq is reachable")

    # 2. Fire a tiny chat completion to read rate-limit response headers
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
    }
    t0 = time.time()
    r2 = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=15,
    )
    latency = (time.time() - t0) * 1000

    h = r2.headers

    if r2.status_code == 429:
        err("Rate-limited RIGHT NOW (429)")
        body = r2.json().get("error", {})
        warn(body.get("message", "No message"))

        # Parse retry time from message
        msg = body.get("message", "")
        import re
        m = re.search(r'try again in ([\d]+m)?([\d.]+)s', msg)
        if m:
            mins = int(m.group(1).rstrip('m')) if m.group(1) else 0
            secs = float(m.group(2))
            total_secs = mins * 60 + secs
            ready_at = datetime.fromtimestamp(time.time() + total_secs)
            warn(f"Retry after: {total_secs:.0f}s  →  available at {ready_at.strftime('%H:%M:%S')}")
    elif r2.status_code == 200:
        ok(f"Chat completions working  ({latency:.0f} ms latency)")
    else:
        err(f"Chat completions returned HTTP {r2.status_code}")
        info(r2.text[:300])

    # Rate-limit headers (present on both success and 429)
    tpm_limit     = h.get("x-ratelimit-limit-tokens")
    tpm_remaining = h.get("x-ratelimit-remaining-tokens")
    tpm_reset     = h.get("x-ratelimit-reset-tokens")
    tpd_limit     = h.get("x-ratelimit-limit-tokens-day")      # may not exist
    tpd_remaining = h.get("x-ratelimit-remaining-tokens-day")  # may not exist
    req_limit     = h.get("x-ratelimit-limit-requests")
    req_remaining = h.get("x-ratelimit-remaining-requests")
    req_reset     = h.get("x-ratelimit-reset-requests")

    print()
    if tpm_limit:
        info(f"Tokens/min  — limit: {tpm_limit:>10}  |  remaining: {tpm_remaining}  |  resets in: {tpm_reset}")
    if tpd_limit:
        info(f"Tokens/day  — limit: {tpd_limit:>10}  |  remaining: {tpd_remaining}")
    if req_limit:
        info(f"Requests    — limit: {req_limit:>10}  |  remaining: {req_remaining}  |  resets in: {req_reset}")

    if not any([tpm_limit, tpd_limit]):
        # Groq sometimes puts usage inside 429 body
        err_body = r2.json() if r2.status_code in (429, 200) else {}
        usage = err_body.get("usage") or {}
        if usage:
            info(f"Usage this request — prompt: {usage.get('prompt_tokens')}  completion: {usage.get('completion_tokens')}  total: {usage.get('total_tokens')}")
        else:
            warn("No token quota headers returned — check Groq dashboard: https://console.groq.com/settings/billing")


# ─────────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────────
def check_gemini():
    header("GEMINI  (gemini-2.0-flash-lite)")

    if not GEMINI_API_KEY:
        err("GEMINI_API_KEY not set in .env")
        return

    # List models
    url_models = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    rm = requests.get(url_models, timeout=10)
    if rm.status_code == 400:
        err("Invalid API key (400 Bad Request)")
        info(rm.text[:200])
        return
    if rm.status_code != 200:
        err(f"Models endpoint returned HTTP {rm.status_code}")
        info(rm.text[:200])
        return
    ok("API key is valid — Gemini is reachable")

    # Fire a tiny generation
    url_gen = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": "Hi"}]}],
        "generationConfig": {"maxOutputTokens": 1},
    }
    t0 = time.time()
    rg = requests.post(url_gen, json=payload, timeout=20)
    latency = (time.time() - t0) * 1000

    if rg.status_code == 429:
        err("Rate-limited RIGHT NOW (429)")
        body = rg.json()
        msg  = body.get("error", {}).get("message", "No message")
        warn(msg)
    elif rg.status_code == 200:
        ok(f"generateContent working  ({latency:.0f} ms latency)")
        # Extract token usage if available
        resp_json = rg.json()
        meta = resp_json.get("usageMetadata", {})
        if meta:
            print()
            info(f"Tokens this request — prompt: {meta.get('promptTokenCount')}  candidates: {meta.get('candidatesTokenCount')}  total: {meta.get('totalTokenCount')}")
    else:
        err(f"generateContent returned HTTP {rg.status_code}")
        info(rg.text[:300])

    # Gemini doesn't expose quota in headers — point to dashboard
    print()
    warn("Gemini quota details are only visible in Google AI Studio:")
    info("→ https://aistudio.google.com/app/apikey  (quota shown per key)")
    info("→ gemini-2.0-flash-lite free tier: 1,500 req/day, 30 req/min, resets daily at midnight PT")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{BOLD}API Status Check  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    check_groq()
    check_gemini()
    print(f"\n{DIM}{'─'*52}{RESET}\n")
