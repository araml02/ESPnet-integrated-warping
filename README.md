# Modeling Children's Speech for a Reading Tutor

This project explores automatic speech recognition (ASR) for children's speech using the [ESPnet](https://github.com/espnet/espnet) framework. The goal is to improve reading assistance tools by adapting ASR models—typically trained on adult speech—to better recognize and understand children’s voices.

We started exploring VTLN in Kaldi and we integrated the following methods in ESPnet:
- Post-processing the features: Warping the filter banks by using torch.nn.functional.grid_sample
- Pre-processing the features:
   - 1) Warping the librosa mel matrix by also using torch.nn.functional.grid_sample
   - 2) Creating a triangular mel filters from scratch with our own function. This technique is not on point yet.

## Summary

- **Framework**: ESPnet2
- **ASR Techniques**: Hybrid CTC/Attention model
- **Datasets we used**:
  - **CGN** (Corpus Gesproken Nederlands) -> to train adult model
  - **CHOREC** (Children's Oral Reading Corpus) -> to apply warping and decode
- **Goal**: Integrated VTLN into the ASR pipeline. Estimating speaker-specific warping factors based on miminmum ASR loss.


### How to use it
- Train model by running stage 3 - 11 in ESPnet (asr.sh file)
- Run warping script by inserting Stage 16 in asr.sh
    - This requires some new files and some changes in ESPnet, which are explained below
- Decode as usual by running Stage 12

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

- In order to use read "utt2warp_ref" from the dump data folder during decoding, we need to apply the following changes to ESPnet:
  - /espnet/espnet2/train/collate_fn.py
  - /espnet/espnet2/train/dataset.py
  - /espnet/espnet2/asr/espnet_model.py
  - /espnet/espnet2/bin/asr_inference.py
 
### 4. Add the following new files
- add /espnet/espnet2/asr/warp_layer.py
  - You can choose the warping function here: f/alpha or f**alpha
- add espnet_speaker_warping.py to your project directory (egs/...)
  
### 5. Run with asr.sh
We recommend using our asr_new.sh script
- asr.sh:
  - Stage 12: allow variable data keys "True"
  - Insert Stage 16

### 6. If you want to continue working on pre-processing the features
This is the same principle, but just different warping technique

- Changes in ESPnet to use warping in the default frontend:
  - espnet/espnet2/asr/frontend/default.py
  - espnet/espnet2/layers/log_mel.py
    
In log_mel.py there are 2 options to shift the mel banks:
- Interpolate the librosa mel matrix and shift with f/alpha or f**alpha
    - Use warp_melmat_differentiable for this 
- Use the new differentiable mel filter 
  - Use make_slaney_vtln_mel_filterbank for this




