import requests

def test_openrouter_api_key(api_key):
    url = "https://openrouter.ai/api/v1/test" 
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    print(f"Response body: {response.text}")

# Replace with your API key
api_key = "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"
test_openrouter_api_key(api_key)
