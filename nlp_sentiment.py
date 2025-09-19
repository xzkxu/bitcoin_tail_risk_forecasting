import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ===== Load Input JSON =====
INPUT_JSON = "bitcoin_news.json"   # path to your JSON file
OUTPUT_JSON = "sentiment_nlp.json"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    news_data = json.load(f)

# ===== Load CryptoBert =====
MODEL_ID = "ElKulako/cryptobert"  # Bullish / Neutral / Bearish

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    attn_implementation="eager"
)

clf = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=64,
    top_k=None,
    device_map="auto"
)

# ===== Process Day by Day =====
results = {}
for day, items in tqdm(news_data.items(), desc="Processing Days"):
    day_result = {}
    for idx, text in items.items():
        pred = clf(text)[0]  # list of dicts with label & score
        probs = {d["label"]: float(d["score"]) for d in pred}
        best_label = max(probs, key=probs.get)
        day_result[idx] = best_label
    results[day] = day_result

# ===== Save to JSON =====
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)