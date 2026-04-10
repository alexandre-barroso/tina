# TINA: A Hybrid Model of Brazilian Portuguese Pronunciation Structure

Preprint version, April 2026. Submitted manuscript under review.

---

## Overview

TINA is a compact hybrid system for automatic phonetic transcription of Brazilian Portuguese. It addresses the task as a structured pronunciation problem rather than a flat grapheme-to-phoneme conversion. The system predicts a syllabified, stress-marked, archiphonemic representation through a multi-output neural architecture, and then expands that representation into surface-oriented variants through explicit phonological rules.

The model is not presented as a state-of-the-art neural G2P architecture. Its purpose is to make Brazilian Portuguese pronunciation structure computationally legible, and to serve as an interpretable model of the regularities that govern syllabification, lexical stress, and dialect-sensitive realization in the language.

---

## System description

### Representation

The training lexicon contains 95,927 orthographic–transcription pairs. Transcriptions use a structured notation in which syllable boundaries are marked with periods, primary stress is marked with `'`, and certain segments are represented as uppercase archiphonemic symbols (D, H, L, R, S, T) encoding classes of realizations rather than fixed surface outputs. The target representation is therefore partially abstract by design.

Each training example is decomposed into four aligned supervision channels: a syllable token sequence, a binary stress label per syllable, a phoneme sequence within each syllable, and a syllable count derived from the transcription.

### Neural architecture

The encoder reads a character sequence through a convolutional front end, a two-layer bidirectional GRU, and a Transformer encoder layer with multi-head self-attention. It produces both time-step representations and a compressed state used to initialize decoding, as well as a scalar syllable-count prediction from the mean of the encoded sequence.

Three coordinated decoders follow. The syllable decoder predicts a sequence of whole syllabic units with Luong-style attention over the encoder. The phoneme decoder generates the segmental content of each predicted syllable, initialized from the corresponding syllable hidden state and also attending to the encoder. The stress decoder processes the syllable sequence with a bidirectional GRU and a Transformer layer, augments syllable representations with average-pooled phoneme features, attends to the encoder, and predicts a binary prominence label per syllable.

### Training

The total training objective is a weighted combination of five losses:

```
L = 0.30 L_syl + 0.30 L_stress + 0.20 L_phon + 0.20 L_align + 0.10 L_boundary
```

The syllable and phoneme losses are cross-entropy with label smoothing (0.1). The stress loss is focal loss (γ = 2.0) with class weighting. The alignment loss is mean squared error over syllable count, averaged across an encoder-based and a boundary-based estimate. The boundary loss is an auxiliary cross-entropy term on the syllable decoder's termination branch.

Training uses Adam (lr = 0.001), batch size 16, dropout 0.1, cosine annealing, and gradient clipping (max norm 1.0). Teacher forcing is applied with exponential decay across epochs.

### Rule-based expansion

After neural inference, `generate_variants()` expands the base transcription into surface-oriented variants. The expansion stages handle: epenthetic *i* insertion in coda and pre-boundary positions; consonant-plus-*s* cluster repair; reduced-vowel and affrication alternations for T and D before high vowels; and dialect-profile substitution for the archiphonemic symbols H, T, D, S, and R. Three dialect profiles are implemented: Nortista/Nordestino, Sulista/Sudestino, and Carioca.

---

## Evaluation

Evaluation proceeds along two tracks.

**Lexical track.** A held-out subset of the training lexicon is used to verify basic transcription competence. Scoring is decomposed across phoneme content, syllabification, stress placement, and syllable-count alignment, reflecting the structure of the model's own output representation.

**Pseudoword track.** 372 orthographically written pseudowords, drawn from the experimental materials in Benevides (2022, 2024), are passed through TINA without accent marks. Model outputs are compared against native-speaker stress productions reported in that benchmark. The comparison is organized by overall stress distribution, stress production by target-stress category, sensitivity to final-syllable structure (CV-CV-CV vs. CV-CV-CVC), and group-wise sensitivity to phonological similarity and lexical frequency.

The human benchmark used for direct comparison consists of 309 validated and quasi-validated items read by 34 native speakers, yielding 10,393 usable productions after exclusions. The full TINA inference batch covers all 372 pseudowords. These denominators correspond to different stages of the experimental pipeline and should not be treated as a discrepancy.

---

## Main results

On the lexical track, TINA functions as a competent structured transcriber on ordinary word forms.

On the pseudoword track, the system produces well-formed outputs for all 372 items, with trisyllabic parses dominating. The aggregate stress distribution splits almost evenly between final (49.2%) and penultimate (50.8%) stress, broadly consistent with the speaker benchmark (44.0% final, 52.3% penultimate, 3.7% antepenultimate). The model captures the main opposition between final and penultimate stress and responds strongly to final-syllable structure: open-final pseudowords are assigned predominantly penultimate stress, while closed-final pseudowords are assigned final stress categorically (100% vs. 79.4% in the speaker benchmark).

The system's limits are also systematic. Antepenultimate stress is never produced across the entire pseudoword batch, whereas it accounts for 3.7% of speaker productions overall and substantially more in conditions with high phonological similarity to antepenultimate-stressed base words. The model is insensitive to the similarity and frequency groupings that modulate speaker behavior, behaving as a structural default system rather than an analogical one.

---

## Repository contents

| Component | Description |
|---|---|
| Training lexicon | 95,927 orthographic–transcription pairs in structured notation |
| Neural model | Encoder + syllable, phoneme, and stress decoders |
| Rule expansion module | `generate_variants()` with epenthesis, alternation, and dialect profiles |
| Pseudoword inference batch | Full 372-item output set with base transcriptions and expanded variants |
| Evaluation materials | Lexical and pseudoword scoring routines aligned with the paper's metrics |
| Web interface | Structured transcription tool surfacing separate accuracy channels |

---

## Citation

> BARROSO, A. M. (2026). *TINA: A Hybrid Model of Brazilian Portuguese Pronunciation Structure*. Manuscript.

---

## Data accessibility

The corpora and files needed for methodological reconstruction are available in this repository. Repository DOI: [placeholder pending assignment].

---

## Ethics

This work reports analyses of previously published datasets and does not involve new data collection from human participants. Ethical review and consent procedures for the original data collections are described in the cited source studies.
