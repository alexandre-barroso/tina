import os
import torch
import torch.nn.functional as F
import random
import time
import statistics
import sys
import json
import matplotlib.pyplot as plt
from collections import defaultdict

from neuralnet import LinguisticSeq2Seq, Encoder, SyllableDecoder, StressDecoder, PhonemeDecoder

# ───────────────────────────────────────────────────────────
# Paths & Hyperparams (must match training)
# ───────────────────────────────────────────────────────────

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_PATH         = os.path.join(BASE_DIR, "data", "data.txt")
MODEL_DIR         = os.path.join(BASE_DIR, "model")
SAVED_MODEL_PATH  = os.path.join(MODEL_DIR, "saved_model.pth")

SRC_LEN         = 30
MAX_PHONEME_LEN = 12

HIDDEN_DIM_NO = 256
EMB_DIM_NO = 128

ENC_EMB_DIM = EMB_DIM_NO
ENC_HID_DIM = HIDDEN_DIM_NO
SYL_EMB_DIM = EMB_DIM_NO
SYL_HID_DIM = HIDDEN_DIM_NO
PH_EMB_DIM = EMB_DIM_NO
PH_HID_DIM = HIDDEN_DIM_NO  

N_LAYERS    = 2
DROPOUT     = 0.1

device = torch.device("cpu")

# ───────────────────────────────────────────────────────────
# Vocabulary Builder (assumed consistent with training)
# ───────────────────────────────────────────────────────────

def build_vocab(data, tokenize_fn, specials):
    vocab = dict(specials)
    idx = max(specials.values()) + 1 if specials else 0
    for text in data:
        for tok in tokenize_fn(text):
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab

def tokenize_chars(text):    return list(text)
def tokenize_phonemes(text): return list(text)

class PhoneticDataset:
    def __init__(self, data_path, src_len, max_ph_len):
        self.src_len, self.max_ph_len = src_len, max_ph_len
        self.samples = []
        with open(data_path, encoding="utf-8") as f:
            for L in f:
                L = L.strip()
                if not L:
                    continue
                parts = L.split(None, 1)
                if len(parts) < 2:
                    continue
                self.samples.append(parts)  # parts[0]: word; parts[1]: transcription

        # Source vocabulary
        self.src_specials = {"<pad_char>": 0}
        self.src_vocab = build_vocab(
            [s[0] for s in self.samples],
            tokenize_chars,
            self.src_specials
        )

        # Phoneme vocabulary
        ph_specials = {"<pad_ph>": 0, "<bop>": 1, "<eop>": 2}
        all_ph = []
        for _, ph in self.samples:
            for syl in ph.strip("/").split("."):
                if syl.startswith("ˈ"):
                    syl = syl[1:]
                all_ph += tokenize_phonemes(syl)
        self.ph_vocab = build_vocab(all_ph, tokenize_phonemes, ph_specials)

        # Syllable vocabulary
        syl_specials = {"<pad_syl>": 0, "<bow>": 1, "<eow>": 2}
        all_syl = []
        for _, ph in self.samples:
            for syl in ph.strip("/").split("."):
                if syl.startswith("ˈ"):
                    syl = syl[1:]
                all_syl.append(syl)
        self.syl_vocab = build_vocab(all_syl, lambda x: [x], syl_specials)

        self.src_vocab_size = len(self.src_vocab)
        self.ph_vocab_size  = len(self.ph_vocab)
        self.syl_vocab_size = len(self.syl_vocab)

# Instantiate dataset for inference and debugging
dataset = PhoneticDataset(DATA_PATH, SRC_LEN, MAX_PHONEME_LEN)
inv_ph_vocab = {i: t for t, i in dataset.ph_vocab.items()}

# ───────────────────────────────────────────────────────────
# Build model (once) — weights will be reloaded inside debug_infer()
# ───────────────────────────────────────────────────────────

encoder = Encoder(
    input_dim=dataset.src_vocab_size,
    emb_dim=ENC_EMB_DIM,
    hid_dim=ENC_HID_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    bidirectional=True,
    pad_idx=dataset.src_specials["<pad_char>"]
)
syllable_decoder = SyllableDecoder(
    syl_vocab_size=dataset.syl_vocab_size,
    syl_emb_dim=SYL_EMB_DIM,
    syl_hid_dim=SYL_HID_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    eow_idx=dataset.syl_vocab["<eow>"],
    bow_idx=dataset.syl_vocab["<bow>"],
    pad_idx=dataset.syl_vocab["<pad_syl>"]
)
phoneme_decoder = PhonemeDecoder(
    phoneme_vocab_size=dataset.ph_vocab_size,
    ph_emb_dim=PH_EMB_DIM,
    ph_hid_dim=PH_HID_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    bop_idx=dataset.ph_vocab["<bop>"],
    eop_idx=dataset.ph_vocab["<eop>"],
    pad_idx=dataset.ph_vocab["<pad_ph>"]
)
stress_decoder = StressDecoder(
    hid_dim=ENC_HID_DIM,
    dropout=DROPOUT
)
model = LinguisticSeq2Seq(
    encoder=encoder,
    syllable_decoder=syllable_decoder,
    stress_decoder=stress_decoder,
    phoneme_decoder=phoneme_decoder,
    device=device
)
model.to(device).eval()

# ───────────────────────────────────────────────────────────
# Pre‑/post‑processing functions
# ───────────────────────────────────────────────────────────

def preprocess_input(word, src_vocab, src_len):
    chars = tokenize_chars(word)
    idxs = [src_vocab.get(c, src_vocab["<pad_char>"]) for c in chars]
    if len(idxs) < src_len:
        idxs += [src_vocab["<pad_char>"]] * (src_len - len(idxs))
    else:
        idxs = idxs[:src_len]
    return torch.tensor([idxs], dtype=torch.long, device=device)

def decode_phoneme_sequence(ph_logits, inv_vocab, pad_ph_idx=0, bop_idx=1, eop_idx=2):
    ids = ph_logits.argmax(dim=-1).tolist()
    out = []
    for tid in ids:
        if tid in {pad_ph_idx, eop_idx}:
            break
        if tid == bop_idx:
            continue
        out.append(inv_vocab.get(tid, ""))
    return "".join(out)

# ───────────────────────────────────────────────────────────
# Inference function: Runs the model and returns prediction
# ───────────────────────────────────────────────────────────

def debug_infer(word):
    # Wait until a saved model is available.
    while not os.path.exists(SAVED_MODEL_PATH):
        print(f"Saved model not found at {SAVED_MODEL_PATH}. Retrying in 5 minutes...")
        time.sleep(300)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
    src = preprocess_input(word.lower(), dataset.src_vocab, SRC_LEN)
    with torch.no_grad():
        out = model(
            src=src,
            teacher_syllable_seq=None,
            teacher_phoneme_seqs=None,
            max_phoneme_len=MAX_PHONEME_LEN,
            teacher_forcing_ratio=0.0,
            expected_syll_count=None
        )
    stress_logits = out["stress_logits"][0]       # [steps, 2]
    phoneme_outs  = out["phoneme_outputs"][0]        # [steps, ph_steps, ph_vocab]
    pred_syllables = []
    for i in range(stress_logits.size(0)):
        syl = decode_phoneme_sequence(
            phoneme_outs[i],
            inv_ph_vocab,
            pad_ph_idx=dataset.ph_vocab["<pad_ph>"],
            bop_idx=dataset.ph_vocab["<bop>"],
            eop_idx=dataset.ph_vocab["<eop>"]
        )
        if not syl:
            continue
        if torch.argmax(stress_logits[i]).item() == 1:
            syl = "ˈ" + syl
        pred_syllables.append(syl)
    transcription = "/" + ".".join(pred_syllables) + "/"
    pred_count = out["predicted_syllable_count"][0].item()
    return {"transcription": transcription, "predicted_syll_count": pred_count}

# ───────────────────────────────────────────────────────────
# Utility functions: Levenshtein distance
# ───────────────────────────────────────────────────────────

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def levenshtein_distance_list(seq1, seq2):
    if len(seq1) < len(seq2):
        return levenshtein_distance_list(seq2, seq1)
    if len(seq2) == 0:
        return len(seq1)
    prev_row = list(range(len(seq2) + 1))
    for i, token1 in enumerate(seq1):
        curr_row = [i + 1]
        for j, token2 in enumerate(seq2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (token1 != token2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

# ───────────────────────────────────────────────────────────
# Main debug_inference: Evaluate n samples and produce a FINAL REPORT
# ───────────────────────────────────────────────────────────

def debug_inference(n_samples=50, log_path="error_log.txt"):
    # Load all samples from the data file.
    all_samples = []
    n_samples_override = 75
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            all_samples.append(parts)  # [word, transcription]
    if len(all_samples) < n_samples_override:
        n_samples_override = len(all_samples)
    samples = random.sample(all_samples, n_samples_override)

    # Lists to accumulate error metrics for the current run.
    phoneme_errors = []
    syllable_errors = []
    stress_errors = []
    alignment_errors = []

    for word, gt_trans in samples:
        gt_clean = gt_trans.strip("/")
        gt_sylls = [s for s in gt_clean.split(".") if s]
        true_syll_count = len(gt_sylls)
        src_word = word.strip()

        try:
            result = debug_infer(src_word)
        except Exception as e:
            print(f"Error processing word '{src_word}':", e)
            continue

        pred_trans = result["transcription"]
        pred_count = result["predicted_syll_count"]
        pred_clean = pred_trans.strip("/")

        # Compute errors as binary indicators (0 or 1).

        # PHONEME ERROR: If the prediction does not exactly match the ground truth.
        ph_wrong = 1 if levenshtein_distance(pred_clean, gt_clean) > 0 else 0
        phoneme_errors.append(ph_wrong)

        # SYLLABLE ERROR: If the predicted syllable segmentation is not exactly the same.
        pred_sylls = [s for s in pred_clean.split(".") if s]
        syl_wrong = 1 if pred_sylls != gt_sylls else 0
        syllable_errors.append(syl_wrong)

        # STRESS ERROR: Compare the binary stress representation.
        gt_stress = "".join(["1" if s.startswith("ˈ") else "0" for s in gt_sylls])
        pred_stress = "".join(["1" if s.startswith("ˈ") else "0" for s in pred_sylls])
        stress_wrong = 1 if pred_stress != gt_stress else 0
        stress_errors.append(stress_wrong)

        # ALIGNMENT ERROR: Check if the predicted syllable count exactly matches the true count.
        alignment_wrong = 1 if round(pred_count) != true_syll_count else 0
        alignment_errors.append(alignment_wrong)

    # Compute current run summary statistics.
    avg_phoneme_err = statistics.mean(phoneme_errors) if phoneme_errors else 0.0
    avg_syllable_err = statistics.mean(syllable_errors) if syllable_errors else 0.0
    avg_stress_err = statistics.mean(stress_errors) if stress_errors else 0.0
    avg_align_err = statistics.mean(alignment_errors) if alignment_errors else 0.0

    med_phoneme_err = statistics.median(phoneme_errors) if phoneme_errors else 0.0
    med_syllable_err = statistics.median(syllable_errors) if syllable_errors else 0.0
    med_stress_err = statistics.median(stress_errors) if stress_errors else 0.0
    med_align_err = statistics.median(alignment_errors) if alignment_errors else 0.0

    report = (
        "RELATÓRIO FINAL:\n"
        "------------------------------------------------------\n"
        f"Amostras avaliadas: {n_samples_override}\n"
        f"Taxa de erro em Fonemas: {avg_phoneme_err:.3f}\n"
        f"Taxa de erro em Sílabas: {avg_syllable_err:.3f}\n"
        f"Taxa de erro em Acentuação: {avg_stress_err:.3f}\n"
        f"Taxa de erro em Alinhamento: {avg_align_err:.3f}\n"
        "------------------------------------------------------\n"
    )

    # Write the final report to a log file and print it.
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(report)

    print(report)

