import os
import torch
import torch.nn.functional as F
import itertools
import re

from neuralnet import LinguisticSeq2Seq, Encoder, SyllableDecoder, StressDecoder, PhonemeDecoder
from latent_space import PhoneticLatentSpace
from phonological_rules import generate_variants

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
                self.samples.append(parts)

        self.src_specials = {"<pad_char>": 0}
        self.src_vocab = build_vocab(
            [s[0] for s in self.samples],
            tokenize_chars,
            self.src_specials
        )

        ph_specials = {"<pad_ph>": 0, "<bop>": 1, "<eop>": 2}
        all_ph = []
        for _, ph in self.samples:
            for syl in ph.strip("/").split("."):
                if syl.startswith("ˈ"):
                    syl = syl[1:]
                all_ph += tokenize_phonemes(syl)
        self.ph_vocab = build_vocab(all_ph, tokenize_phonemes, ph_specials)

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

dataset = PhoneticDataset(DATA_PATH, SRC_LEN, MAX_PHONEME_LEN)
inv_ph_vocab = {i: t for t, i in dataset.ph_vocab.items()}

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

def preprocess_input(word, src_vocab, src_len):
    chars = tokenize_chars(word)
    idxs = [src_vocab.get(c, src_vocab["<pad_char>"]) for c in chars]
    if len(idxs) < src_len:
        idxs += [src_vocab["<pad_char>"]] * (src_len - len(idxs))
    else:
        idxs = idxs[:src_len]
    return torch.tensor([idxs], dtype=torch.long, device=device)

def decode_phoneme_sequence(ph_logits, inv_vocab,
                            pad_ph_idx=0, bop_idx=1, eop_idx=2):
    ids = ph_logits.argmax(dim=-1).tolist()
    out = []
    for tid in ids:
        if tid in {pad_ph_idx, eop_idx}:
            break
        if tid == bop_idx:
            continue
        out.append(inv_vocab.get(tid, ""))
    return "".join(out)

def _infer_single_word(word):

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

    stress_logits = out["stress_logits"][0]  
    phoneme_outs = out["phoneme_outputs"][0]   
    syllables = []

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

        syllables.append(syl)

    return ".".join(syllables)

def process_token(token):

    if '-' in token:
        parts = token.split('-')
        processed_parts = [_infer_single_word(part) for part in parts]
        return ".".join(processed_parts)
    else:
        return _infer_single_word(token)

def infer(text):

    if os.path.exists(SAVED_MODEL_PATH):
        model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
    else:
        raise FileNotFoundError(f"No saved model at {SAVED_MODEL_PATH}")

    tokens = text.strip().split()

    transcriptions = [process_token(token) for token in tokens]

    base_transcription = " ".join(transcriptions)

    wrapped_transcription = "/" + base_transcription + "/"

    final_transcription, no_variants = generate_variants(wrapped_transcription)
    return final_transcription, no_variants

def infer_with_pattern(word, pattern_idx, latent_space_module=None):

    original = _infer_single_word(word)

    try:

        src = preprocess_input(word.lower(), dataset.src_vocab, SRC_LEN)
        with torch.no_grad():

            enc_outputs, enc_hidden, _ = model.encoder(src)

            mask = (src != dataset.src_specials["<pad_char>"])

            model.syllable_decoder.set_encoder_outputs(enc_outputs, mask)
            model.stress_decoder.set_encoder_outputs(enc_outputs, mask)
            model.phoneme_decoder.set_encoder_outputs(enc_outputs, mask)

            if latent_space_module is None:
                from latent_space import PhoneticLatentSpace
                latent_space_module = PhoneticLatentSpace(dim=enc_hidden.size(-1), num_patterns=32)

                latent_space_path = os.path.join(os.path.dirname(__file__), "model", "latent_space_model.pth")
                if os.path.exists(latent_space_path):
                    try:
                        latent_space_module.load_state_dict(torch.load(latent_space_path, map_location=torch.device("cpu")))
                        print(f"Loaded latent space model from {latent_space_path}")
                    except Exception as e:
                        print(f"Could not load latent space model: {e}")

            if not (0 <= pattern_idx < latent_space_module.num_patterns):
                raise ValueError(f"Invalid pattern ID: {pattern_idx}")

            pattern_scale = 0.9 

            pattern_vector = latent_space_module.pattern_vectors[pattern_idx].detach().to(device)  

            enc_hidden_modified = enc_hidden + pattern_scale * pattern_vector.unsqueeze(0).unsqueeze(1)

            syll_outputs, syll_logits, boundary_logits = model.syllable_decoder(
                init_hidden=enc_hidden_modified,
                teacher_syllable_seq=None,
                max_steps=None,
                teacher_forcing_ratio=0.0,
                expected_syll_count=None
            )

            if syll_outputs is None or syll_outputs.size(1) == 0:
                return {
                    "original": "/" + original + "/",
                    "pattern_influenced": "/<no output>/"
                }

            dec_syl_steps = syll_outputs.size(1)

            phoneme_outs = []
            for t in range(dec_syl_steps):
                init_ph_hidden = syll_outputs[:, t, :].unsqueeze(0)
                if init_ph_hidden.size(0) < model.phoneme_decoder.n_layers:
                    init_ph_hidden = init_ph_hidden.repeat(model.phoneme_decoder.n_layers, 1, 1)

                if torch.isnan(init_ph_hidden).any():
                    continue

                try:
                    ph_out, _ = model.phoneme_decoder(
                        init_hidden=init_ph_hidden,
                        max_steps=MAX_PHONEME_LEN,
                        teacher_phoneme_seq=None,
                        teacher_forcing_ratio=0.0
                    )
                    phoneme_outs.append(ph_out)
                except Exception as e:
                    print(f"Error in phoneme decoding: {e}")
                    continue

            if not phoneme_outs:
                return {
                    "original": "/" + original + "/",
                    "pattern_influenced": "/<no phonemes>/"
                }

            try:
                stress_logits = model.stress_decoder(
                    syllable_outputs=syll_outputs,
                    phoneme_features=torch.zeros_like(syll_outputs),  
                    valid_positions=None
                )
            except Exception as e:
                print(f"Error in stress decoding: {e}")

                stress_logits = torch.zeros((syll_outputs.size(0), syll_outputs.size(1), 2), device=device)

            syllables = []

            for i in range(min(stress_logits.size(0), len(phoneme_outs))):
                try:
                    syl = decode_phoneme_sequence(
                        phoneme_outs[i][0],
                        inv_ph_vocab,
                        pad_ph_idx=dataset.ph_vocab["<pad_ph>"],
                        bop_idx=dataset.ph_vocab["<bop>"],
                        eop_idx=dataset.ph_vocab["<eop>"]
                    )

                    if not syl:
                        continue

                    if i < stress_logits.size(1) and torch.argmax(stress_logits[0][i]).item() == 1:
                        syl = "ˈ" + syl

                    syllables.append(syl)
                except Exception as e:
                    print(f"Error decoding syllable {i}: {e}")
                    continue

            if not syllables:
                pattern_transcription = "no syllables decoded"
            else:
                pattern_transcription = ".".join(syllables)

        return {
            "original": "/" + original + "/",
            "pattern_influenced": "/" + pattern_transcription + "/"
        }

    except Exception as e:
        print(f"Error in pattern-influenced inference: {e}")

        return {
            "original": "/" + original + "/",
            "pattern_influenced": f"/error: {str(e)[:30]}.../"
        }

if __name__ == "__main__":
    print("Enter words/phrases (type exit to quit):")
    while True:
        w = input("> ").strip()
        if w.lower() in {"exit", "quit"}:
            break
        if not w:
            continue
        print(infer(w))

