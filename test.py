import requests

# Test the root endpoint
response = requests.get('http://localhost:5000/')
print("Root endpoint:", response.text)

# Test the summarizer
test_url = ""  # Example video
response = requests.post(
    'http://localhost:5000/summarize',
    json={'url': test_url}
)
print("\nSummary endpoint:", response.json())