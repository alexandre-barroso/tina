import os
import sys
import time
from collections import Counter

# ── make sure inference.py (and its siblings) are importable ─────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# inference.py exposes the internal helpers we need
from inference import process_token, model, SAVED_MODEL_PATH, device
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = os.path.join(SCRIPT_DIR, "data", "pseudopalavras.txt")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TSV_PATH    = os.path.join(RESULTS_DIR, "transcriptions.tsv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── load model weights once ───────────────────────────────────────────────────
if not os.path.exists(SAVED_MODEL_PATH):
    raise FileNotFoundError(f"Saved model not found at: {SAVED_MODEL_PATH}")

print(f"Loading model weights from {SAVED_MODEL_PATH} …")
model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
model.eval()
print("Model loaded.\n")

# ── read input words ──────────────────────────────────────────────────────────
with open(INPUT_PATH, encoding="utf-8") as fh:
    words = [ln.strip() for ln in fh if ln.strip()]

print(f"Words to transcribe : {len(words)}")
print(f"Results directory   : {RESULTS_DIR}\n")

# ── run inference (archphonemes only, no dialectal expansion) ─────────────────
results      = []   # list of (word, transcription)
errors       = []   # list of (word, error_message)
syllable_counts = Counter()
stress_positions = Counter()   # which syllable carries stress (1-indexed from right)

t0 = time.time()

for i, word in enumerate(words, 1):
    try:
        # process_token handles hyphenated tokens; returns syllable string without slashes
        archphoneme = process_token(word.lower())
        transcription = f"/{archphoneme}/"
        results.append((word, transcription))

        # --- collect stats ---
        # count syllables (split on ".", ignore stress marker)
        syllables = [s for s in archphoneme.split(".") if s]
        syllable_counts[len(syllables)] += 1

        # locate primary stress (ˈ prefix)
        stressed = [j for j, s in enumerate(syllables, 1) if s.startswith("ˈ")]
        if stressed:
            # position from the right edge (penultimate = 2, final = 1, etc.)
            pos_from_right = len(syllables) - stressed[0] + 1
            stress_positions[pos_from_right] += 1

    except Exception as exc:
        errors.append((word, str(exc)))
        results.append((word, "<ERROR>"))

    if i % 50 == 0 or i == len(words):
        elapsed = time.time() - t0
        print(f"  [{i:>4}/{len(words)}] elapsed {elapsed:.1f}s")

elapsed_total = time.time() - t0

# ── write TSV ─────────────────────────────────────────────────────────────────
with open(TSV_PATH, "w", encoding="utf-8") as fh:
    fh.write("word\tarchphoneme\n")
    for word, transcription in results:
        fh.write(f"{word}\t{transcription}\n")

print(f"\nTranscriptions saved → {TSV_PATH}")

# ── build summary ─────────────────────────────────────────────────────────────
total_words   = len(words)
ok_words      = total_words - len(errors)
unique_trans  = len({t for _, t in results if t != "<ERROR>"})

stress_label = {1: "oxítona (final)", 2: "paroxítona (penult.)", 3: "proparoxítona (antepenult.)"}

summary_lines = [
    "=" * 60,
    "  INFERENCE SUMMARY – pseudopalavras (archphonemes only)",
    "=" * 60,
    "",
    f"  Input file        : {INPUT_PATH}",
    f"  Total words       : {total_words}",
    f"  Successful        : {ok_words}",
    f"  Errors            : {len(errors)}",
    f"  Unique outputs    : {unique_trans}",
    f"  Total time (s)    : {elapsed_total:.2f}",
    f"  Avg time/word (s) : {elapsed_total/total_words:.3f}",
    "",
    "── Syllable-count distribution ─────────────────────────────",
]
for n_syl in sorted(syllable_counts):
    count = syllable_counts[n_syl]
    pct   = 100 * count / ok_words
    bar   = "█" * int(pct / 2)
    summary_lines.append(f"  {n_syl} syl : {count:>4}  ({pct:5.1f}%)  {bar}")

summary_lines += [
    "",
    "── Stress position distribution (from right edge) ──────────",
]
total_stressed = sum(stress_positions.values())
for pos in sorted(stress_positions):
    count = stress_positions[pos]
    pct   = 100 * count / ok_words
    label = stress_label.get(pos, f"{pos} syllables from right")
    bar   = "█" * int(pct / 2)
    summary_lines.append(f"  {label:<32}: {count:>4}  ({pct:5.1f}%)  {bar}")

unstressed = ok_words - total_stressed
if unstressed > 0:
    pct = 100 * unstressed / ok_words
    summary_lines.append(f"  {'no stress marker':<32}: {unstressed:>4}  ({pct:5.1f}%)")

if errors:
    summary_lines += [
        "",
        "── Errors ──────────────────────────────────────────────────",
    ]
    for word, msg in errors:
        summary_lines.append(f"  {word:<20} {msg}")

summary_lines += ["", "=" * 60]
summary_text = "\n".join(summary_lines)

with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
    fh.write(summary_text + "\n")

print(f"Summary saved       → {SUMMARY_PATH}\n")
print(summary_text)
