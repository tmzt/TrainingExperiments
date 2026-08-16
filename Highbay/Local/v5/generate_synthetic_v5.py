import os
import json

# Define Google Drive paths
DRIVE_BASE = "/content/drive/MyDrive/HighbayGeniusTraining"
OUTPUT_DIR = os.path.join(DRIVE_BASE, "datasets", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_typed_markdown_v5.json")

# Gemini Pro Generation Logic
def generate_synthetic_data():
    print(f"Target dataset output path: {OUTPUT_FILE}")
    synthetic_dataset = [
        {
            "prompt": "Create a checkout review step that displays user().email and cart().total.",
            "target_ir": "---\nscreen: CheckoutReview\neffects: [user, cart]\n---\n- [VStack]\n  - [Text] \"Account: {{ abc user().email }}\"\n  - [Text] \"Total: {{ $ cart().total }}\""
        }
    ]
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(synthetic_dataset, f, indent=2)
    print(f"Dataset successfully written to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_synthetic_data()
