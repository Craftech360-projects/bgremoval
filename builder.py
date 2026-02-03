import os
import sys
import requests

# Standard cache directory
u2net_home = os.getenv('U2NET_HOME', '/root/.u2net')
model_path = os.path.join(u2net_home, "birefnet-general.onnx")

# Official BiRefNet URL
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx"

print(f"--- ⏳ DOWNLOADING OFFICIAL BIREFNET MODEL ---")
print(f"Target: {model_path}")

os.makedirs(u2net_home, exist_ok=True)

try:
    if not os.path.exists(model_path):
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("--- ✅ DOWNLOAD COMPLETE ---")
    else:
        print("--- ℹ️  FILE ALREADY EXISTS ---")

    # Verify Size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    if size_mb > 900: 
        print("✅ VERIFIED: Correct 1GB model downloaded.")
    else:
        print(f"❌ ERROR: File too small ({size_mb:.2f} MB). Download failed.")
        sys.exit(1)

except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)