import requests

models = ["mistralai/Pixtral-12B-2409", "mistralai/Pixtral-12B-Base-2409"]

for m in models:
    print(f"\n--- {m} ---")
    
    # Check gating
    api_url = f"https://huggingface.co/api/models/{m}"
    res = requests.get(api_url)
    if res.status_code == 200:
        data = res.json()
        print(f"Gated: {data.get('gated', 'Unknown')}")
        siblings = data.get('siblings', [])
        files = [s['rfilename'] for s in siblings]
        print("All Files:", files)
    else:
        print(f"Model not found or error: {res.status_code}")

