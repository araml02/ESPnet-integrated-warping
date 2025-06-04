# Modeling Children's Speech for a Reading Tutor

This project explores automatic speech recognition (ASR) for children's speech using the [ESPnet](https://github.com/espnet/espnet) framework. The goal is to improve reading assistance tools by adapting ASR models—typically trained on adult speech—to better recognize and understand children’s voices.

## Summary

- **Framework**: ESPnet
- **ASR Techniques**: Hybrid CTC/Attention model
- **Datasets**:
  - **CHOREC** (Children's Oral Reading Corpus)
  - **CGN** (Corpus Gesproken Nederlands)
- **Goal**: Integrated VTLN into the ASR pipeline. Estimating speaker-specific warping factors based on miminmum ASR loss

---

## 🔧 Setup

### 1. Clone ESPnet2
[ESPnet](https://github.com/espnet/espnet)

### 2. Update the following ESPnet files:
- Changes are marked as:
  - "#ARAM ============"
  - changes here
  - "#ARAM ============"
- Also some functions are adapted to accept "utt2warp"

- In order to use read "utt2warp_ref" from the data folder during decoding, we need to apply the following changes to ESPnet:
  - collate_fn.py
  - dataset.py
  - espnet_model.py

### 3. Run with asr.sh
- asr.sh:
  - Stage 12: allow variable data keys "True"

### 4. If you want to continue working on the shifted mel bins of the ESPnet frontend module:

- Apply changes to following files:
  - espnet/espnet2/asr/frontend/default.py
  - espnet/espnet2/layers/log_mel.py
In log_mel there are 2 options to shift the mel banks:
- Interpolate the librosa mel matrix and shift with f/alpha or f**alpha
    - Use warp_melmat_differentiable for this 
- Use the new differentiable mel filter 
  - Use make_slaney_vtln_mel_filterbank for this




