# Example Python script to query the MS2 Search REST API
#
# This script demonstrates how to send a query spectrum to the
# Flask server and interpret the results.
#
# To run this script:
# 1. Make sure the Flask API server (`app.py`) is running.
# 2. Install the 'requests' library if you haven't already:
#    pip install requests
# 3. Run this script from your terminal:
#    python your_query_script_name.py

import requests
import json

# --- Configuration ---
# The URL of the running Flask API server.
# If your server is on a different machine, replace '1227.0.0.1' with its IP address.
API_URL = "http://127.0.0.1:5000/search"

# The name of the library you want to search against.
# This should correspond to a file named 'my_test_library.msp' in the 'libraries' folder.
# Set this to None to search against all available libraries.
TARGET_LIBRARY = "MoNA-export-GC-MS_Spectra" 

# --- Prepare the Query Spectrum ---
# This is a sample spectrum for Caffeine, which has enough peaks to satisfy
# the server's default requirement of at least 6 matched peaks.
query_spectrum = {
    "precursorMz": 195.087,
    "peaks": [
        [55.05, 18.1],
        [67.05, 12.3],
        [82.06, 25.5],
        [109.05, 100.0],
        [138.06, 85.7],
        [166.06, 14.8],
        [195.08, 42.1]
    ]
}

# --- Construct the Full API Payload ---
payload = {
    "spectrum": query_spectrum
}

# Only add the 'library' key if a specific one is targeted.
if TARGET_LIBRARY:
    payload["library"] = TARGET_LIBRARY

# --- Send the Request and Handle the Response ---
print(f"Sending query to: {API_URL}")
print("Payload:")
# Use json.dumps for pretty printing the dictionary
print(json.dumps(payload, indent=2))
print("-" * 30)

try:
    # Send the POST request with the JSON payload
    response = requests.post(API_URL, json=payload)

    # Raise an exception for bad status codes (4xx or 5xx)
    response.raise_for_status()

    # If the request was successful, process the JSON response
    result = response.json()
    
    print("Server Response:")
    if result.get("status") == "match_found":
        print("✅ Match Found!")
        print(f"   Library: {result.get('library')}")
        print(f"   Score: {result.get('score')}")
        print("   Annotation:")
        # Pretty-print the annotation dictionary
        annotation = result.get('annotation', {})
        for key, value in annotation.items():
            print(f"     - {key}: {value}")
            
    elif result.get("status") == "no_match_found":
        print("❌ No match found in the library.")
        if 'reason' in result:
            print(f"   Reason: {result['reason']}")
            
    else:
        print("Received an unexpected response format:")
        print(result)

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
    print(f"Status Code: {response.status_code}")
    # Try to print the error message from the server's JSON response
    try:
        print(f"Server message: {response.json()}")
    except json.JSONDecodeError:
        print(f"Server response (not JSON): {response.text}")
        
except requests.exceptions.ConnectionError as conn_err:
    print(f"Connection error occurred: {conn_err}")
    print("Please ensure the Flask API server is running and accessible.")
    
except Exception as err:
    print(f"An other error occurred: {err}")
