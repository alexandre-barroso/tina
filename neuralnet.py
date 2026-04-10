import torch
import torch.nn as nn
import torch.nn.functional as F
import random

###############################################################################
# Luong-Style Attention (Multi-Head)
###############################################################################
class LuongAttention(nn.Module):

    def __init__(self, hid_dim, num_heads=4, dropout=0.1):
        super(LuongAttention, self).__init__()
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: [B, hid_dim] -> query shape: (1, B, hid_dim)
        query = decoder_hidden.unsqueeze(0)
        # encoder_outputs: [B, src_len, hid_dim] -> key and value: (src_len, B, hid_dim)
        key = encoder_outputs.transpose(0, 1)
        value = key

        # nn.MultiheadAttention expects key_padding_mask: shape [B, src_len] with True = ignore.
        key_padding_mask = ~mask if mask is not None else None

        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        # attn_output: (1, B, hid_dim) => squeeze to (B, hid_dim)
        context = attn_output.squeeze(0)
        # attn_weights: (B, 1, src_len) => squeeze to (B, src_len)
        attn_weights = attn_weights.squeeze(1)
        return context, attn_weights

###############################################################################
# Updated Encoder with Convolutional Feature Extraction, Transformer Layer,
# and Layer Normalization
###############################################################################
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim,
                 n_layers=2, dropout=0.1, bidirectional=True, pad_idx=None):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        # Convolutional Feature Extractor: captures local grapheme patterns.
        self.conv = nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.relu = nn.ReLU()

        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        # Corrected layer normalization to match GRU output shape.
        self.layer_norm = nn.LayerNorm(hid_dim * self.n_directions)
        
        # Project GRU outputs to hid_dim.
        self.fc_out = nn.Sequential(
            nn.Linear(hid_dim * self.n_directions, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        # Additional Transformer encoder layer to boost capacity.
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=4, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.pad_idx = pad_idx

        # Syllable count head using Softplus for smooth gradients.
        self.syll_count_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, 1),
            nn.Softplus()
        )

    def forward(self, src):
        # src: [B, src_len, ...]
        embedded = self.dropout(self.embedding(src))
        # Convolution expects [B, emb_dim, src_len]
        conv_input = embedded.transpose(1, 2)
        conv_out = self.relu(self.bn(self.conv(conv_input)))  # [B, emb_dim, src_len]
        # Transpose back to [B, src_len, emb_dim]
        conv_out = conv_out.transpose(1, 2)
        # Residual connection: combine conv output with original embedding.
        conv_out = conv_out + embedded
        
        # GRU produces outputs: [B, src_len, hid_dim * n_directions]
        outputs, hidden = self.rnn(conv_out)
        outputs = self.layer_norm(outputs)
        # Project to [B, src_len, hid_dim]
        proj_outputs = self.fc_out(outputs)
        # Pass through Transformer encoder layer.
        proj_outputs = self.transformer_layer(proj_outputs.transpose(0, 1)).transpose(0, 1)
        
        if self.bidirectional:
            # Combine bidirectional hidden states via summation.
            hidden = hidden.view(self.n_layers, 2, hidden.size(1), self.hid_dim)
            hidden_final = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        else:
            hidden_final = hidden

        # Predict syllable count from the mean representation.
        avg_encoded = proj_outputs.mean(dim=1)
        syll_count_pred = self.syll_count_head(avg_encoded).squeeze(-1)
        
        return proj_outputs, hidden_final, syll_count_pred

###############################################################################
# Stress Decoder (Incorporates Phoneme Features by Default)
###############################################################################
class StressDecoder(nn.Module):

    def __init__(self, hid_dim, dropout=0.1):
        super(StressDecoder, self).__init__()
        
        # Attention mechanism for encoder outputs
        self.encoder_attention = LuongAttention(hid_dim)
        
        # Bidirectional GRU to process entire syllable sequence
        self.rnn = nn.GRU(
            input_size=hid_dim,
            hidden_size=hid_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Positional encoding to help with position-sensitive tasks
        self.positional_encoding = nn.Embedding(50, hid_dim)  # Max 50 syllables
        
        # Transformer layer to capture global patterns
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hid_dim*2,  # Bidirectional output
            nhead=4,
            dropout=dropout
        )
        
        # Prediction layers
        self.syllable_context = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final stress prediction (binary: stressed/unstressed)
        self.stress_classifier = nn.Linear(hid_dim, 2)
        
        # To hold external encoder outputs and mask if provided.
        self._encoder_outputs = None
        self._encoder_mask = None
        
        # Projection to map phoneme features to the same dimension.
        self.phoneme_projector = nn.Linear(hid_dim, hid_dim)
        
    def set_encoder_outputs(self, encoder_outputs, mask=None):
        self._encoder_outputs = encoder_outputs
        self._encoder_mask = mask
        
    def forward(self, syllable_outputs, syllable_lengths=None, phoneme_features=None, valid_positions=None):
        batch_size, seq_len, hid_dim = syllable_outputs.shape
        
        # Add positional information
        positions = torch.arange(seq_len, device=syllable_outputs.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embeddings = self.positional_encoding(positions)
        syllable_outputs = syllable_outputs + pos_embeddings
        
        # Process syllable sequence with bidirectional GRU
        if syllable_lengths is not None:
            packed_seq = nn.utils.rnn.pack_padded_sequence(
                syllable_outputs, syllable_lengths, batch_first=True, enforce_sorted=False
            )
            outputs, _ = self.rnn(packed_seq)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, _ = self.rnn(syllable_outputs)
        
        # Apply transformer for global context
        transformer_in = outputs.transpose(0, 1)  # [seq_len, batch, hid_dim*2]
        transformer_out = self.transformer(transformer_in).transpose(0, 1)  # [batch, seq_len, hid_dim*2]
        
        # Project to hidden dimension
        syllable_context = self.syllable_context(transformer_out)
        
        # Integrate phoneme features (always applied)
        syllable_context = syllable_context + self.phoneme_projector(phoneme_features)
        
        # For each syllable, attend to relevant encoder parts if available
        if self._encoder_outputs is not None:
            enhanced_features = []
            for t in range(seq_len):
                context, _ = self.encoder_attention(
                    syllable_context[:, t], 
                    self._encoder_outputs,
                    mask=self._encoder_mask
                )
                # Combine context with syllable representation
                enhanced = syllable_context[:, t] + context
                enhanced_features.append(enhanced)
            
            enhanced_context = torch.stack(enhanced_features, dim=1)
        else:
            enhanced_context = syllable_context
            
        stress_logits = self.stress_classifier(enhanced_context)
        
        return stress_logits

###############################################################################
# Updated Syllable Decoder 
###############################################################################
class SyllableDecoder(nn.Module):
    def __init__(self, syl_vocab_size, syl_emb_dim, syl_hid_dim,
                 n_layers=2, dropout=0.1, eow_idx=2, bow_idx=1, pad_idx=None):
        super(SyllableDecoder, self).__init__()
        self.embedding = nn.Embedding(syl_vocab_size, syl_emb_dim, padding_idx=pad_idx)

        self.rnn = nn.GRU(
            input_size=syl_emb_dim + syl_hid_dim,
            hidden_size=syl_hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        self.attn = LuongAttention(syl_hid_dim)
        self.dropout = nn.Dropout(dropout)

        self.eow_idx = eow_idx
        self.bow_idx = bow_idx
        self.n_layers = n_layers
        self.pad_idx = pad_idx
        self.syl_vocab_size = syl_vocab_size

        # Shared base: transforms GRU output into a refined representation.
        self.fc_hidden = nn.Sequential(
            nn.Linear(syl_hid_dim, syl_hid_dim),
            nn.ReLU()
        )

        # Boundary Prediction Head for syllable segmentation.
        self.boundary_fc = nn.Linear(syl_hid_dim, 2)

        # Token branch
        self.token_branch = nn.Sequential(
            nn.Linear(syl_hid_dim, syl_hid_dim),
            nn.ReLU()
        )
        self.fc_token = nn.Linear(syl_hid_dim, syl_vocab_size)

        self._encoder_outputs = None
        self._encoder_mask = None

    def set_encoder_outputs(self, encoder_outputs, mask=None):
        self._encoder_outputs = encoder_outputs
        self._encoder_mask = mask

    def forward(self,
                init_hidden,               # [n_layers, B, syl_hid_dim]
                teacher_syllable_seq=None, # [B, syl_seq_len], optional
                max_steps=None,
                teacher_forcing_ratio=1.0,
                expected_syll_count=None):
        batch_size = init_hidden.size(1)
        shared_outputs = []
        token_logits_list = []
        boundary_logits_list = []

        input_token = torch.full((batch_size,), self.bow_idx, dtype=torch.long, device=init_hidden.device) 
        hidden = init_hidden
        finished = torch.zeros(batch_size, dtype=torch.bool, device=init_hidden.device)
        step = 0

        # Determine maximum steps.
        if teacher_syllable_seq is not None:
            global_max_steps = teacher_syllable_seq.size(1) - 1 # because we remove the <bow> token
        elif expected_syll_count is not None:
            global_max_steps = expected_syll_count.max().item()
        else:
            global_max_steps = 20

        while step < global_max_steps and not finished.all():
            embedded = self.dropout(self.embedding(input_token))
            top_hidden = hidden[-1]
            if self._encoder_outputs is not None:
                context, _ = self.attn(top_hidden, self._encoder_outputs, mask=self._encoder_mask)
            else:
                context = torch.zeros_like(top_hidden)

            rnn_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
            output, hidden = self.rnn(rnn_input, hidden)
            output = output.squeeze(1)

            shared_repr = self.fc_hidden(output)
            shared_outputs.append(shared_repr)

            # Boundary prediction (for segmentation)
            boundary_logits_list.append(self.boundary_fc(shared_repr))

            # Token branch for syllable token prediction.
            token_hidden = self.token_branch(shared_repr) + shared_repr
            token_logits_list.append(self.fc_token(token_hidden))

            # Teacher forcing / EOW logic (unchanged)
            if teacher_syllable_seq is not None and (step + 1) < teacher_syllable_seq.size(1):
                gold_next = teacher_syllable_seq[:, step + 1]
            else:
                gold_next = None

            use_gold = gold_next is not None and random.random() < teacher_forcing_ratio
            input_token = gold_next if use_gold else token_logits_list[-1].argmax(dim=1)

            if expected_syll_count is not None:
                force_eow = (step + 1) >= expected_syll_count.to(hidden.device)
                input_token = torch.where(force_eow,
                                          torch.full_like(input_token, self.eow_idx),
                                          input_token)

            finished = finished | (input_token == self.eow_idx)
            step += 1

        shared_outputs = torch.stack(shared_outputs, dim=1)            # [B, T, H]
        token_logits = torch.stack(token_logits_list, dim=1)             # [B, T, V]
        boundary_logits = torch.stack(boundary_logits_list, dim=1)         # [B, T, 2]

        return shared_outputs, token_logits, boundary_logits

###############################################################################
# Phoneme Decoder with Feature Extraction (Always Returns Features)
###############################################################################
class PhonemeDecoder(nn.Module):

    def __init__(self, phoneme_vocab_size, ph_emb_dim, ph_hid_dim,
                 n_layers=2, dropout=0.1,
                 bop_idx=0, eop_idx=1, pad_idx=None):
        super(PhonemeDecoder, self).__init__()
        self.embedding = nn.Embedding(phoneme_vocab_size, ph_emb_dim, padding_idx=pad_idx)

        self.rnn = nn.GRU(
            input_size=ph_emb_dim + ph_hid_dim,
            hidden_size=ph_hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )

        # Multi-head attention for context.
        self.attn = LuongAttention(ph_hid_dim)
        # Residual connection to enhance features.
        self.fc_hidden = nn.Sequential(
            nn.Linear(ph_hid_dim, ph_hid_dim),
            nn.ReLU()
        )
        self.fc_out = nn.Linear(ph_hid_dim, phoneme_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.phoneme_vocab_size = phoneme_vocab_size
        self.n_layers = n_layers
        self.bop_idx = bop_idx
        self.eop_idx = eop_idx
        self.pad_idx = pad_idx

        self._encoder_outputs = None
        self._encoder_mask = None

    def set_encoder_outputs(self, encoder_outputs, mask=None):
        self._encoder_outputs = encoder_outputs
        self._encoder_mask = mask

    def forward(self,
                init_hidden,         # [n_layers, B, ph_hid_dim]
                max_steps=None,
                teacher_phoneme_seq=None,  # [B, ph_seq_len]
                teacher_forcing_ratio=1.0):
        batch_size = init_hidden.size(1)
        outputs = []
        phoneme_feature_list = []

        input_token = torch.full((batch_size,), self.bop_idx,
                                 dtype=torch.long, device=init_hidden.device)
        hidden = init_hidden
        finished = torch.zeros(batch_size, dtype=torch.bool, device=init_hidden.device)
        step = 0

        if max_steps is None:
            if teacher_phoneme_seq is not None:
                max_steps = teacher_phoneme_seq.size(1)
            else:
                max_steps = 20

        while step < max_steps and not finished.all():
            embedded = self.dropout(self.embedding(input_token))
            top_hidden = hidden[-1]
            if self._encoder_outputs is not None:
                context, _ = self.attn(top_hidden, self._encoder_outputs)
            else:
                context = torch.zeros_like(top_hidden)

            rnn_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
            output, hidden = self.rnn(rnn_input, hidden)
            output = output.squeeze(1)

            # Refined representation via residual connection.
            refined = output + self.fc_hidden(output)
            token_logits = self.fc_out(refined)
            outputs.append(token_logits)
            # Always collect refined phoneme features.
            phoneme_feature_list.append(refined)

            if teacher_phoneme_seq is not None and (random.random() < teacher_forcing_ratio):
                if (step + 1) < teacher_phoneme_seq.size(1):
                    gold_next = teacher_phoneme_seq[:, step + 1]
                else:
                    gold_next = torch.full((batch_size,), self.eop_idx,
                                           dtype=torch.long, device=hidden.device)
                input_token = gold_next
            else:
                input_token = token_logits.argmax(dim=1)

            finished = finished | (input_token == self.eop_idx)
            step += 1

        if len(outputs) == 0:
            phoneme_logits = torch.zeros(batch_size, 0, self.phoneme_vocab_size, device=init_hidden.device)
            phoneme_features = torch.zeros(batch_size, 0, hidden.size(-1), device=init_hidden.device)
            return phoneme_logits, phoneme_features
        
        phoneme_logits = torch.stack(outputs, dim=1)  # [B, T, V]
        phoneme_features = torch.stack(phoneme_feature_list, dim=1)  # [B, T, ph_hid_dim]
        return phoneme_logits, phoneme_features

###############################################################################
# LinguisticSeq2Seq 
###############################################################################
class LinguisticSeq2Seq(nn.Module):
    def __init__(self, encoder, syllable_decoder, stress_decoder, phoneme_decoder, device):
        super(LinguisticSeq2Seq, self).__init__()
        self.encoder = encoder
        self.syllable_decoder = syllable_decoder
        self.stress_decoder = stress_decoder
        self.phoneme_decoder = phoneme_decoder
        self.device = device

    def forward(self,
                src,                    # [B, src_len]
                teacher_syllable_seq=None,  # [B, syl_len]
                teacher_phoneme_seqs=None,  # [B, syl_len, ph_len]
                max_phoneme_len=20,
                teacher_forcing_ratio=1.0,
                expected_syll_count=None):
        # 1) Encode
        enc_outputs, enc_hidden, syll_count_pred = self.encoder(src)
        mask = (src != self.encoder.pad_idx)
        self.syllable_decoder.set_encoder_outputs(enc_outputs, mask)
        self.stress_decoder.set_encoder_outputs(enc_outputs, mask)
        self.phoneme_decoder.set_encoder_outputs(enc_outputs, mask)

        # 2) Syllable decoding (no stress yet)
        if expected_syll_count is None:
            expected_syll_count = syll_count_pred.round().clamp(min=1, max=20).long()
        syll_max = teacher_syllable_seq.size(1) if teacher_syllable_seq is not None else None
        syll_outputs, syll_logits, boundary_logits = self.syllable_decoder(
            init_hidden=enc_hidden,
            teacher_syllable_seq=teacher_syllable_seq,
            max_steps=syll_max,
            teacher_forcing_ratio=teacher_forcing_ratio,
            expected_syll_count=expected_syll_count
        )
        dec_syl_steps = syll_outputs.size(1)

        # 3) Phoneme decoding per syllable; always extract phoneme features.
        raw_phoneme_outs = []
        phoneme_features_list = []
        for t in range(dec_syl_steps):
            # Use each syllable’s hidden state as the initial state for phoneme decoding.
            init_ph_hidden = syll_outputs[:, t, :].unsqueeze(0)
            if init_ph_hidden.size(0) < self.phoneme_decoder.n_layers:
                init_ph_hidden = init_ph_hidden.repeat(self.phoneme_decoder.n_layers, 1, 1)
            teacher_ph_seq_t = (teacher_phoneme_seqs[:, t, :]
                                if teacher_phoneme_seqs is not None and t < teacher_phoneme_seqs.size(1)
                                else None)
            ph_out, ph_features = self.phoneme_decoder(
                init_hidden=init_ph_hidden,
                max_steps=max_phoneme_len,
                teacher_phoneme_seq=teacher_ph_seq_t,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            if ph_out.size(1) < max_phoneme_len:
                ph_out = F.pad(ph_out, (0,0,0, max_phoneme_len - ph_out.size(1)))
                ph_features = F.pad(ph_features, (0,0,0, max_phoneme_len - ph_features.size(1)))
            else:
                ph_out = ph_out[:, :max_phoneme_len, :]
                ph_features = ph_features[:, :max_phoneme_len, :]
            raw_phoneme_outs.append(ph_out.unsqueeze(1))
            # Aggregate the phoneme features for this syllable (using average pooling).
            syll_phoneme_feature = ph_features.mean(dim=1)  # [B, ph_hid_dim]
            phoneme_features_list.append(syll_phoneme_feature.unsqueeze(1))
        
        phoneme_outputs = torch.cat(raw_phoneme_outs, dim=1) if raw_phoneme_outs else None
        phoneme_features = torch.cat(phoneme_features_list, dim=1) if phoneme_features_list else None

        # 4) Stress decoding (using both syllable and phoneme information)
        valid_syllable_mask = (syll_logits.argmax(dim=-1) != self.syllable_decoder.bow_idx) & \
                              (syll_logits.argmax(dim=-1) != self.syllable_decoder.eow_idx)

        # Only predict stress for valid syllable positions
        stress_logits = self.stress_decoder(
            syllable_outputs=syll_outputs,
            phoneme_features=phoneme_features,
            valid_positions=valid_syllable_mask
        )

        # 5) Compute aggregated boundary count from boundary logits for count-level supervision.
        boundary_probs = F.softmax(boundary_logits, dim=-1)[..., 1]  # [B, T]
        pred_boundary_count = boundary_probs.sum(dim=1) + 1  # +1 if starting syllable is implicit

        return {
            'stress_logits':       stress_logits,        # from StressDecoder
            'syllable_logits':     syll_logits,
            'boundary_logits':     boundary_logits,
            'phoneme_outputs':     phoneme_outputs,
            'predicted_syllable_count': syll_count_pred,
            'encoder_syll_count_pred':  syll_count_pred,
            'pred_boundary_count': pred_boundary_count
        }

