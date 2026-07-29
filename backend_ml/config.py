import os
from dotenv import load_dotenv
import torch

# Load the .env file into os.environ
load_dotenv()

# Read the paths directly as strings
MODEL_PATH = os.getenv("MODEL_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")
ISO_FOREST_PATH = os.getenv("ISO_FOREST_PATH")
METADATA_PATH = os.getenv("METADATA_PATH")

# External credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
NODE_WEBHOOK_URL = os.getenv("NODE_WEBHOOK_URL")

# Set the torch device based on string setting
dev_setting = os.getenv("HARDWARE_DEVICE", "auto").lower()
if dev_setting == "cuda":
    DEVICE = torch.device("cuda")
elif dev_setting == "cpu":
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    