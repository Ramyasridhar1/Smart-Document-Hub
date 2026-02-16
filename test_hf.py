import os, requests, json, sys
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv()
print('Found .env at:', env_path or '(none)')

print('OS before load: HF_MODEL =', os.environ.get('HF_MODEL'))

load_dotenv(dotenv_path=env_path, override=True)

print('OS after load:  HF_MODEL =', os.environ.get('HF_MODEL'))
print('OS after load:  HF_API_KEY present?', bool(os.environ.get('HF_API_KEY')))

hf_key = os.environ.get('HF_API_KEY')
hf_model = os.environ.get('HF_MODEL', 'NOT_SET')
if not hf_key:
    print('ERROR: HF_API_KEY not found in environment. Stop.')
    sys.exit(0)

url = f'https://api-inference.huggingface.co/models/{hf_model}'
print('Calling:', url)
headers = {'Authorization': f'Bearer {hf_key}'}
payload = {'inputs': 'Hello, who are you?', 'parameters': {'max_new_tokens': 120}}

try:
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    print('HTTP status:', r.status_code)
    try:
        jr = r.json()
        print('JSON (truncated):', json.dumps(jr, indent=2)[:1500])
    except Exception as e:
        print('Failed to decode JSON:', e)
        print('Raw text (truncated):', r.text[:2000])
except Exception as e:
    print('Request failed:', e)
