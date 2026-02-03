import requests
import base64
import time
import json

# --- CONFIGURATION ---
API_KEY = ""
ENDPOINT_ID = "d2vbcjdj5sv40t"  # e.g., "vxxqiz..."
IMAGE_PATH = "image.png"     # The image you want to process
OUTPUT_PATH = "result.png"
# ---------------------

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def decode_save(base64_string, output_path):
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(base64_string))

def main():
    # 1. Prepare Payload
    print(f"Loading {IMAGE_PATH}...")
    b64_image = encode_image(IMAGE_PATH)
    
    payload = {
        "input": {
            "image": b64_image
        }
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 2. Send Request (Using 'runsync' for simplicity)
    # runsync waits for the job to finish (up to 90s) before returning
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    
    print("Sending request to RunPod (this might take time on first run)...")
    response = requests.post(url, json=payload, headers=headers)

    # 3. Handle Response
    try:
        data = response.json()
    except:
        print("Failed to parse JSON:", response.text)
        return

    if response.status_code != 200:
        print(f"Error {response.status_code}: {data}")
        return

    # 4. Check for Execution Status
    status = data.get('status')
    print(f"Job Status: {status}")

    if status == 'COMPLETED':
        output = data.get('output', {})
        
        # Check if our handler returned an error
        if 'error' in output:
            print("Worker Error:", output['error'])
            if 'traceback' in output:
                print(output['traceback'])
        
        # Success! Save the image
        elif 'image' in output:
            decode_save(output['image'], OUTPUT_PATH)
            print(f"✅ Success! Saved background-free image to: {OUTPUT_PATH}")
        else:
            print("Unknown output format:", output)
            
    else:
        # If it timed out or failed
        print("Job did not complete successfully.")
        print("Full Response:", data)

if __name__ == "__main__":
    main()