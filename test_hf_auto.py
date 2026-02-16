# test_hf_auto.py
import os, requests, json, sys, time

# load .env if present (best-effort)
env_path = '.env'
if os.path.exists(env_path):
    print("Found .env at:", os.path.abspath(env_path))
    for line in open(env_path, 'r', encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip()
        v = v.strip()
        os.environ.setdefault(k, v)

hf_key = os.getenv('HF_API_KEY')
hf_model = os.getenv('HF_MODEL', '').strip()

if not hf_key:
    print("ERROR: HF_API_KEY not found in environment. Add it to your .env or export HF_API_KEY.")
    sys.exit(1)

candidates = []
if hf_model:
    candidates.append(hf_model)
# Add a short candidate list of commonly inference-enabled public models.
# The script will try each in order until one returns HTTP 200
candidates.extend([
    "google/flan-t5-large",
    "google/flan-t5-base",
    "bigscience/bloomz-1b1",
    "bigscience/bloom-560m",
    "tiiuae/falcon-7b-instruct",          # sometimes gated; may 404
    "HuggingFaceH4/zephyr-7b-alpha",      # try, but may be gated
    "mistralai/Mistral-7B-Instruct-v0.1", # try
    "replit/replit-code-v1-3b"            # example, if text generation
])

headers = {"Authorization": f"Bearer {hf_key}"}
payload = {"inputs": "Hello, who are you?", "parameters": {"max_new_tokens": 150}}

def try_model(model_id):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    print("\nCalling:", url)
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
    except Exception as e:
        print("Request failed:", e)
        return False, None, None
    print("HTTP status:", r.status_code)
    text = r.text[:4000]
    # try to decode json, but print raw text if fails
    try:
        jr = r.json()
        print("JSON response (truncated):")
        print(json.dumps(jr, indent=2)[:4000])
    except Exception as e:
        print("Failed to decode JSON response:", e)
        print("Raw text (truncated):", text)
    return r.status_code == 200, r, r.text

# run the check: first try configured HF_MODEL, then candidates
tried = set()
for model in candidates:
    if not model or model in tried:
        continue
    tried.add(model)
    ok, resp, txt = try_model(model)
    if ok:
        print("\n✅ Model WORKS:", model)
        print("You can set HF_MODEL =", model, "in your .env and restart your app.")
        sys.exit(0)
    else:
        # helpful hints from response text
        if resp is not None:
            if resp.status_code == 403:
                print("→ 403 Forbidden — model might be gated or your key lacks access.")
            elif resp.status_code == 404:
                print("→ 404 Not Found — model not available for Inference API (not deployed or private).")
            elif resp.status_code == 429:
                print("→ 429 Rate limit / quota exceeded.")
        time.sleep(1)

print("\n❌ No candidate model returned HTTP 200. Actions you can take next:")
print("  1) Visit the model page on huggingface.co and confirm it's marked 'Inference API (Text Generation)' or 'Inference Available'.")
print("  2) Choose a model listed on HF that has 'Inference Available' (click the filter on the Models page).")
print("  3) Make sure your HF_API_KEY has the right access (if model is gated you must request access on the model page).")
print("  4) If you want, paste the exact output (HTTP status + first 4000 chars of response) here and I will interpret it.")
sys.exit(2)
