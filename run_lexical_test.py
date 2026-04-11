import os
import re
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from inference import process_token, model, SAVED_MODEL_PATH, device
from phonological_rules import generate_variants

INPUT_PATH = os.path.join(SCRIPT_DIR, "data", "lexical_test.txt")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_lexical")
TSV_PATH = os.path.join(RESULTS_DIR, "transcriptions.tsv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def format_variants(variants_string: str) -> str:
    variants = re.findall(r"/[^/]+/", variants_string)
    return ", ".join(variants)


if not os.path.exists(SAVED_MODEL_PATH):
    raise FileNotFoundError(f"Saved model not found at: {SAVED_MODEL_PATH}")

print(f"Loading model weights from {SAVED_MODEL_PATH} …")
model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
model.eval()
print("Model loaded.\n")

with open(INPUT_PATH, encoding="utf-8") as fh:
    words = [ln.strip() for ln in fh if ln.strip()]

print(f"Words to transcribe : {len(words)}")
print(f"Results file        : {TSV_PATH}\n")

results = []
errors = []

t0 = time.time()

for i, word in enumerate(words, 1):
    try:
        archphoneme = process_token(word.lower())
        wrapped_archphoneme = f"/{archphoneme}/"

        variants_string, _ = generate_variants(wrapped_archphoneme)
        formatted_variants = format_variants(variants_string)

        if not formatted_variants:
            formatted_variants = wrapped_archphoneme

        results.append((word, formatted_variants))

    except Exception as exc:
        errors.append((word, str(exc)))
        results.append((word, "<ERROR>"))

    if i % 50 == 0 or i == len(words):
        elapsed = time.time() - t0
        print(f"  [{i:>4}/{len(words)}] elapsed {elapsed:.1f}s")

elapsed_total = time.time() - t0

with open(TSV_PATH, "w", encoding="utf-8") as fh:
    fh.write("word\ttranscriptions\n")
    for word, transcriptions in results:
        fh.write(f"{word}\t{transcriptions}\n")

print(f"\nTranscriptions saved → {TSV_PATH}")
print(f"Completed in {elapsed_total:.2f}s")
if errors:
    print(f"Words with errors: {len(errors)}")
