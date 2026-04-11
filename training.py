import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from neuralnet import LinguisticSeq2Seq, Encoder, SyllableDecoder, StressDecoder, PhonemeDecoder
from latent_space import PhoneticLatentSpace

DATA_PATH = "./data/data.txt"
MODEL_DIR = "./model"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")
SAVED_MODEL_PATH = os.path.join(MODEL_DIR, "saved_model.pth")

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS_PER_CHECKPOINT = 1
EPOCHS_TOTAL = 100

HIDDEN_DIM_NO = 256
EMB_DIM_NO = 128

ENC_EMB_DIM = EMB_DIM_NO
ENC_HID_DIM = HIDDEN_DIM_NO
SYL_EMB_DIM = EMB_DIM_NO
SYL_HID_DIM = HIDDEN_DIM_NO
PH_EMB_DIM = EMB_DIM_NO
PH_HID_DIM = HIDDEN_DIM_NO  

N_LAYERS = 2
DROPOUT = 0.1

device = torch.device("cpu")

latent_space_module = PhoneticLatentSpace(dim=ENC_HID_DIM, num_patterns=32)

def build_vocab(data, tokenize_fn, specials):
    vocab = dict(specials)
    idx = max(specials.values()) + 1 if specials else 0
    for text in data:
        tokens = tokenize_fn(text)
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def tokenize_chars(text):
    return list(text)

def tokenize_phonemes(text):
    return list(text)

class PaddedPhoneticDataset(Dataset):
    def __init__(self, data_path, max_char_len=30, max_syl_len=10, max_ph_len=12):
        self.max_char_len = max_char_len
        self.max_syl_len = max_syl_len
        self.max_ph_len  = max_ph_len
        self.samples = []

        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                ortho, phonetic = parts
                p_str = phonetic.strip("/")
                n_periods = p_str.count(".")
                expected_syll_count = n_periods + 1
                self.samples.append((ortho, phonetic, expected_syll_count))

        self.src_specials = {"<pad_char>": 0}
        src_texts = [s[0] for s in self.samples]
        self.src_vocab = build_vocab(src_texts, tokenize_chars, self.src_specials)

        self.syl_specials = {"<pad_syl>": 0, "<bow>": 1, "<eow>": 2}
        self.ph_specials  = {"<pad_ph>": 0, "<bop>": 1, "<eop>": 2}

        all_syllables = []
        all_phonemes  = []
        for _, phon_str, _ in self.samples:
            phon_str = phon_str.strip("/")
            sylls = phon_str.split(".")
            for syl in sylls:
                syl_clean = syl[1:] if syl.startswith("ˈ") else syl
                all_syllables.append(syl_clean)
                all_phonemes.extend(tokenize_phonemes(syl_clean))

        self.syl_vocab = build_vocab(all_syllables, lambda x: [x], self.syl_specials)
        self.ph_vocab  = build_vocab(all_phonemes, tokenize_phonemes, self.ph_specials)

        self.src_vocab_size = len(self.src_vocab)
        self.syl_vocab_size = len(self.syl_vocab)
        self.ph_vocab_size  = len(self.ph_vocab)

        self.pad_idx_char = self.src_specials["<pad_char>"]
        self.pad_idx_syl  = self.syl_specials["<pad_syl>"]
        self.pad_idx_ph   = self.ph_specials["<pad_ph>"]
        self.bow_idx = self.syl_specials["<bow>"]
        self.eow_idx = self.syl_specials["<eow>"]
        self.bop_idx = self.ph_specials["<bop>"]
        self.eop_idx = self.ph_specials["<eop>"]

        self.padded_samples = []
        for ortho, phon_str, exp_count in self.samples:
            char_tokens = tokenize_chars(ortho)
            char_indices = [self.src_vocab[ch] for ch in char_tokens if ch in self.src_vocab]
            char_indices = self._pad_seq(char_indices, self.max_char_len, self.pad_idx_char)

            phon_str = phon_str.strip("/")
            raw_sylls = phon_str.split(".")
            syl_seq = [self.bow_idx]
            stress_seq = []
            ph_lists = []

            for syl in raw_sylls:
                stress_flag = 1 if syl.startswith("ˈ") else 0
                syl_clean = syl[1:] if stress_flag else syl
                syl_idx = self.syl_vocab.get(syl_clean, len(self.syl_vocab))
                syl_seq.append(syl_idx)
                stress_seq.append(stress_flag)

                ph_toks = tokenize_phonemes(syl_clean)
                ph_list = [self.bop_idx]
                for tok in ph_toks:
                    if tok in self.ph_vocab:
                        ph_list.append(self.ph_vocab[tok])
                ph_list.append(self.eop_idx)
                ph_list = self._pad_seq(ph_list, self.max_ph_len, self.pad_idx_ph)
                ph_lists.append(ph_list)

            syl_seq.append(self.eow_idx)
            stress_seq.append(0)

            syl_seq = self._pad_seq(syl_seq, self.max_syl_len, self.pad_idx_syl)
            stress_seq = self._pad_seq(stress_seq, self.max_syl_len - 1, 0)

            while len(ph_lists) < self.max_syl_len - 1:
                ph_lists.append([self.pad_idx_ph] * self.max_ph_len)
            if len(ph_lists) > self.max_syl_len - 1:
                ph_lists = ph_lists[:self.max_syl_len - 1]

            self.padded_samples.append({
                "src": char_indices,
                "syl_seq": syl_seq,
                "stress_seq": stress_seq,
                "phoneme_seqs": ph_lists,
                "expected_syll_count": exp_count
            })

    def _pad_seq(self, seq, max_len, pad_idx):
        return seq + [pad_idx] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]

    def __len__(self):
        return len(self.padded_samples)

    def __getitem__(self, idx):
        return self.padded_samples[idx]

def pad_collate_fn(batch):
    src_tensors, syl_tensors, stress_tensors, ph_tensors_list, expected_counts = [], [], [], [], []
    for item in batch:
        src_tensors.append(torch.tensor(item["src"], dtype=torch.long))
        syl_tensors.append(torch.tensor(item["syl_seq"], dtype=torch.long))
        stress_tensors.append(torch.tensor(item["stress_seq"], dtype=torch.long))
        ph_tensors_list.append(torch.tensor(item["phoneme_seqs"], dtype=torch.long))
        expected_counts.append(item["expected_syll_count"])
    return {
        "src": torch.stack(src_tensors, dim=0),
        "syl_seq": torch.stack(syl_tensors, dim=0),
        "stress_seq": torch.stack(stress_tensors, dim=0),
        "phoneme_seqs": torch.stack(ph_tensors_list, dim=0),
        "expected_syll_count": torch.tensor(expected_counts, dtype=torch.long)
    }

MAX_CHAR_LEN = 30
MAX_SYL_LEN = 10
MAX_PH_LEN = 12
dataset = PaddedPhoneticDataset(DATA_PATH, max_char_len=MAX_CHAR_LEN,
                                max_syl_len=MAX_SYL_LEN, max_ph_len=MAX_PH_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)

print("Qtde. vocabulário de caracteres únicos:", dataset.src_vocab_size)
print("Qtde. vocabulário de sílabas únicas:", dataset.syl_vocab_size)
print("Qtde. vocabulário de fonemas únicos:", dataset.ph_vocab_size)

encoder = Encoder(
    input_dim=dataset.src_vocab_size,
    emb_dim=ENC_EMB_DIM,
    hid_dim=ENC_HID_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    bidirectional=True,
    pad_idx=dataset.pad_idx_char
)

syllable_decoder = SyllableDecoder(
    syl_vocab_size=dataset.syl_vocab_size,
    syl_emb_dim=SYL_EMB_DIM,
    syl_hid_dim=SYL_HID_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    eow_idx=dataset.eow_idx,
    bow_idx=dataset.bow_idx,  
    pad_idx=dataset.pad_idx_syl
)

phoneme_decoder = PhonemeDecoder(
    phoneme_vocab_size=dataset.ph_vocab_size,
    ph_emb_dim=PH_EMB_DIM,
    ph_hid_dim=PH_HID_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    bop_idx=dataset.bop_idx,
    eop_idx=dataset.eop_idx,
    pad_idx=dataset.pad_idx_ph
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
).to(device)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_TOTAL)

syllable_criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx_syl, label_smoothing=0.1)
phoneme_criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx_ph, label_smoothing=0.1)
stress_class_weights = torch.tensor([0.5, 2.0], device=device)
stress_criterion = FocalLoss(gamma=2.0, weight=stress_class_weights)
alignment_criterion = nn.MSELoss()

boundary_criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx_syl)

loss_weights = {
    'syl': 0.30,
    'stress': 0.30,
    'phon': 0.20,
    'align': 0.20
}
boundary_token_weight = 0.1

start_epoch = 1
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if os.path.exists(CHECKPOINT_PATH):
    print(f"Carregando checkpoint de {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Reiniciando treinamento da época {start_epoch}")

decay_constant = 30.0

epoch = start_epoch
while True:
    model.train()
    total_phon_loss  = 0.0
    total_str_loss   = 0.0
    total_syl_loss   = 0.0
    total_align_loss = 0.0
    batch_count      = 0

    teacher_forcing_ratio = math.exp(-epoch / decay_constant)

    latent_space_module.reset_statistics()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
    for batch_data in pbar:
        src        = batch_data["src"].to(device)
        syl_seq    = batch_data["syl_seq"].to(device)
        stress_seq = batch_data["stress_seq"].to(device)
        ph_seqs    = batch_data["phoneme_seqs"].to(device)
        exp_count  = batch_data["expected_syll_count"].to(device)

        optimizer.zero_grad()

        outputs = model(
            src=src,
            teacher_syllable_seq=syl_seq,
            teacher_phoneme_seqs=ph_seqs,
            max_phoneme_len=MAX_PH_LEN,
            teacher_forcing_ratio=teacher_forcing_ratio,
            expected_syll_count=exp_count
        )

        phoneme_outputs = outputs["phoneme_outputs"]
        stress_logits   = outputs["stress_logits"]
        syllable_logits = outputs["syllable_logits"]
        pred_count      = outputs["predicted_syllable_count"]
        pred_boundary_count = outputs["pred_boundary_count"]

        dec_syl_steps = syllable_logits.size(1)

        teacher_syl = syl_seq[:, 1:1 + dec_syl_steps]

        syl_seq_clamped = syl_seq[:, 1:dec_syl_steps + 1]
        stress_seq_clamped = stress_seq[:, :dec_syl_steps]

        teacher_boundary_seq = (syl_seq_clamped == dataset.eow_idx).long()

        syl_logits_flat = syllable_logits.reshape(-1, syllable_logits.size(-1))
        teacher_syl_flat = teacher_syl.reshape(-1)
        syl_loss = syllable_criterion(syl_logits_flat, teacher_syl_flat)

        valid_syllable_mask = (syl_seq_clamped != dataset.pad_idx_syl) & (syl_seq_clamped != dataset.eow_idx)

        B, T, C = stress_logits.shape
        stress_logits_flat = stress_logits.reshape(-1, C)
        teacher_stress_flat = stress_seq_clamped.reshape(-1)
        valid_mask = valid_syllable_mask.reshape(-1)
        if valid_mask.sum() > 0:
            valid_stress_logits = stress_logits_flat[valid_mask]
            valid_teacher_stress = teacher_stress_flat[valid_mask]
            str_loss = stress_criterion(valid_stress_logits, valid_teacher_stress)
        else:
            str_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if phoneme_outputs is not None:
            ph_out_clamped = phoneme_outputs[:, :dec_syl_steps, ...]
            B_, sylD, phL, phV = ph_out_clamped.shape
            ph_out_flat = ph_out_clamped.view(-1, phV)
            ph_seqs_clamped = ph_seqs[:, :dec_syl_steps, :]
            teacher_ph_flat = ph_seqs_clamped.reshape(-1)
            phon_loss = phoneme_criterion(ph_out_flat, teacher_ph_flat)
        else:
            phon_loss = torch.tensor(0.0, device=device)

        teacher_count = exp_count.float()
        align_loss_original = alignment_criterion(pred_count, teacher_count)
        align_loss_boundary = alignment_criterion(pred_boundary_count, teacher_count)
        align_loss = 0.5 * (align_loss_original + align_loss_boundary)

        boundary_logits_flat = outputs["boundary_logits"].reshape(-1, 2)
        teacher_boundary_flat = teacher_boundary_seq.reshape(-1)
        boundary_token_loss = boundary_criterion(boundary_logits_flat, teacher_boundary_flat)

        total_loss = (loss_weights['syl']    * syl_loss +
                      loss_weights['stress'] * str_loss +
                      loss_weights['phon']   * phon_loss +
                      loss_weights['align']  * align_loss +
                      boundary_token_weight  * boundary_token_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_syl_loss   += syl_loss.item()
        total_str_loss   += str_loss.item()
        total_phon_loss  += phon_loss.item()
        total_align_loss += align_loss.item()
        batch_count      += 1

        pbar.set_postfix({
            "L_syl": f"{syl_loss.item():.3f}",
            "L_str": f"{str_loss.item():.3f}",
            "L_phon": f"{phon_loss.item():.3f}",
            "L_align": f"{align_loss.item():.3f}",
            "L_bound": f"{boundary_token_loss.item():.3f}"
        })

        with torch.no_grad():
            _, enc_hidden, _ = model.encoder(src)
            aggregated_hidden = enc_hidden[-1]
            phoneme_seq_for_update = ph_seqs.view(ph_seqs.size(0), -1)
            latent_space_module.update_statistics(phoneme_seq_for_update, aggregated_hidden)

    avg_syl   = total_syl_loss  / batch_count if batch_count > 0 else 0.0
    avg_str   = total_str_loss  / batch_count if batch_count > 0 else 0.0
    avg_phon  = total_phon_loss / batch_count if batch_count > 0 else 0.0
    avg_align = total_align_loss/ batch_count if batch_count > 0 else 0.0

    print(f"\nResumo da época {epoch}:")
    print(f"   Perda média nível sílaba   = {avg_syl:.8f}")
    print(f"   Perda média nível acentuação = {avg_str:.8f}")
    print(f"   Perda média nível fonema    = {avg_phon:.8f}")
    print(f"   Perda média nível alinhamento = {avg_align:.8f}")
    print(f"   Teacher forcing = {teacher_forcing_ratio:.8f}")

    scheduler.step()

    if epoch % EPOCHS_PER_CHECKPOINT == 0:
        checkpoint_data = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)
        torch.save(model.state_dict(), SAVED_MODEL_PATH)
        print(f"Checkpoint e modelo salvos da época {epoch}")
        latent_space_module.save_model(epoch=epoch)

    epoch += 1
    time.sleep(0.1)
