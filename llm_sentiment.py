import json
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from openai import OpenAI

# ========= CONFIG =========
API_KEY     = "API KEY"
BASE_URL    = "https://api.deepseek.com"
MODEL_NAME  = "deepseek-reasoner"   # or "deepseek-chat"
INPUT_JSON  = Path("bitcoin_news.json")
OUTPUT_JSON = Path("sentiment_llm.json")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# Control how often to save partial results (1 = after each day)
SAVE_EVERY  = 1

# process only dates >= this (YYYY-MM-DD). Set to None to process all.
START_DATE  = "2022-02-09"  # e.g., "2021-12-01" or None

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ========= PROMPTS =========
SYSTEM_PROMPT = """You are a financial sentiment annotator for Bitcoin.
Classify the sentiment TOWARD BITCOIN for each snippet as exactly one of:
- Bullish → positive for Bitcoin / likely upward pressure or favorable sentiment.
- Bearish → negative for Bitcoin / likely downward pressure or unfavorable sentiment.
- Neutral → mentions Bitcoin but no clear directional impact.

Decision rules:
- Adoption & pro-Bitcoin regulation (ETF approvals, positive policy, institutions adding BTC, merchants accepting BTC) → Bullish.
- Price action:
  • Rising/holding support/positive momentum → Bullish.
  • Falling/breaking support/negative momentum → Bearish.
  • Sideways/unclear without directional bias → Neutral.
- Mining/regulatory:
  • Restrictions, bans, punitive taxes, enforcement actions → Bearish.
  • Supportive frameworks / clarity enabling growth → Bullish.
- Mixed/ambiguous without clear net direction → Neutral.
- If Bitcoin is only background context without directional implication → Neutral.

Output format (STRICT):
Given a list of (index, text) pairs for one date, return ONLY a minified JSON object mapping each string index to one of {"Bullish","Bearish","Neutral"}.
NO explanations, NO quotes, NO markdown, NO extra fields.
Example:
{"0":"Bullish","1":"Neutral","2":"Bearish"}
"""

ALLOWED = {"Bullish", "Bearish", "Neutral"}

def build_day_payload(items: Dict[str, str]) -> str:
    lines = []
    for k in sorted(items, key=lambda x: int(x) if x.isdigit() else x):
        txt = items[k].strip().replace("\n", " ")
        lines.append(f"{k}\t{txt}")
    header = "Classify these Bitcoin snippets. Each line is '<index>\\t<text>':\n"
    return header + "\n".join(lines)

def extract_json_object(s: str) -> str:
    m = re.search(r'\{.*\}', s, flags=re.DOTALL)
    return m.group(0) if m else ""

def sanitize_labels(obj: Dict[str, str], original_items: Dict[str, str]) -> Dict[str, str]:
    out = {}
    for k in original_items.keys():
        v = obj.get(k, "Neutral")
        vv = str(v).strip().capitalize()
        out[k] = vv if vv in ALLOWED else "Neutral"
    return out

def ask_model(day_payload: str) -> Dict[str, str]:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": day_payload + '\n\nReturn ONLY a minified JSON object like {"0":"Bullish"}.'
            },
        ],
        temperature=0,
        top_p=1,
        stream=False,
        max_tokens=3000,
    )
    text = resp.choices[0].message.content.strip()
    js = extract_json_object(text) or text
    try:
        return json.loads(js)
    except Exception:
        # strict retry
        resp2 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": day_payload + '\n\nReturn ONLY valid minified JSON without comments or markdown. Example: {"0":"Bullish"}'
                },
            ],
            temperature=0,
            top_p=1,
            stream=False,
            max_tokens=3000,
        )
        text2 = resp2.choices[0].message.content.strip()
        js2 = extract_json_object(text2) or text2
        return json.loads(js2)

def parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")

def load_existing_output(path: Path) -> Dict[str, Dict[str, str]]:
    if path.exists() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text())
        except Exception:
            print(f"[WARN] Existing {path} could not be parsed; starting fresh.")
    return {}

def atomic_write_json(path: Path, data: Dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)

# ========= RUN =========
news_all = json.loads(INPUT_JSON.read_text())
existing = load_existing_output(OUTPUT_JSON)   # resume support
result = dict(existing)                        # start from existing output

# Build and filter date list
all_dates: List[str] = sorted(news_all.keys(), key=parse_date)
if START_DATE:
    start_dt = parse_date(START_DATE)
    dates = [d for d in all_dates if parse_date(d) >= start_dt]
else:
    dates = all_dates

# Skip dates that are already done (resume)
pending_dates = [d for d in dates if d not in result]

total = len(pending_dates)
if total == 0:
    print("[INFO] Nothing to do. All requested dates already processed.")
else:
    print(f"[INFO] Processing {total} dates (saving every {SAVE_EVERY}). Start at {pending_dates[0]}")

batch_counter = 0

for i, date in enumerate(pending_dates, 1):
    items = news_all[date]
    payload = build_day_payload(items)
    try:
        raw = ask_model(payload)
        result[date] = sanitize_labels(raw, items)
        print(f"[DONE] {i}/{total}  {date} → {result[date]}")
    except Exception as e:
        print(f"[WARN] {date} failed ({e}); defaulting to Neutral.")
        result[date] = {k: "Neutral" for k in items.keys()}
        print(f"[DONE] {i}/{total}  {date} → {result[date]} (fallback)")

    batch_counter += 1
    if batch_counter % SAVE_EVERY == 0:
        atomic_write_json(OUTPUT_JSON, result)
        print(f"[SAVE] Checkpoint written to {OUTPUT_JSON} at {i}/{total} dates.")

# Final save (in case the last batch < SAVE_EVERY)
atomic_write_json(OUTPUT_JSON, result)
print(f"\n[FINAL] All pending dates processed. Saved: {OUTPUT_JSON}")