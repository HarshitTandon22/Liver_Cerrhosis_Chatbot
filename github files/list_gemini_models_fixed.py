#!/usr/bin/env python3
"""List available Gemini models (with warning suppression)"""
import os
from dotenv import load_dotenv

load_dotenv()
import sys
import subprocess
import re

# Set API key

# Run the original script in a subprocess and filter warnings
result = subprocess.run(
    [sys.executable, '-c', '''
import os
from dotenv import load_dotenv

load_dotenv()
import sys
import warnings
warnings.filterwarnings("ignore")

# Suppress stderr for the import
import io
_original_stderr = sys.stderr
sys.stderr = io.StringIO()

from google import generativeai as genai

# Restore stderr
sys.stderr = _original_stderr

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
genai.configure(api_key=api_key)

print("Listing available Gemini models...")
models = genai.list_models()
for model in models:
    if "generateContent" in model.supported_generation_methods:
        print(f"- {model.name}")
'''],
    capture_output=True,
    text=True,
    env=os.environ.copy()
)

# Filter out warning lines from both stdout and stderr
output_lines = result.stdout.split('\n')
stderr_lines = result.stderr.split('\n') if result.stderr else []

filtered_lines = []
for line in output_lines:
    # Skip lines containing the warning patterns
    if not re.search(r'WARNING.*absl|E0000.*ALTS|alts_credentials', line, re.IGNORECASE):
        filtered_lines.append(line)

# Also filter stderr if any output leaked through
for line in stderr_lines:
    if not re.search(r'WARNING.*absl|E0000.*ALTS|alts_credentials', line, re.IGNORECASE):
        if line.strip():  # Only add non-empty lines
            filtered_lines.append(line)

# Print filtered output
print('\n'.join(filtered_lines))

# Exit with the same code as subprocess
sys.exit(result.returncode)
