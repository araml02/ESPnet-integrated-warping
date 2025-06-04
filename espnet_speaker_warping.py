#!/usr/bin/env python3
import os
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import torch
import kaldiio
from espnet2.bin.asr_inference import Speech2Text
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.asr.warp_layer import Filterbankwarping
from torch.cuda.amp import autocast
import shutil
import numpy as np
import subprocess
from espnet2.tasks.asr import ASRTask
import soundfile as sf
import io
import tempfile
import re
import random
import uuid
import contextlib
import sys
import random
import torchaudio

def load_speech_and_lengths(scp_reader, utt_id, raw, device):
    if not raw:
        feat = scp_reader[utt_id]
        speech = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        current_dur =  speech.shape[1] * 0.01 # assume 10ms per frame
    if raw:
        wave_path = scp_reader[utt_id]
        speech, sr = torchaudio.load(wave_path)  # speech shape: (1, T)
        speech = speech.to(torch.float32).to(device)
        current_dur = speech.shape[1] / 16000 # assume 16kHz SR
    speech_lengths = torch.tensor([speech.shape[1]], dtype=torch.long).to(device)
    return speech, speech_lengths, current_dur

def character_error_rate(hyp, ref):
    def levenshtein(a, b):
        """Simple Levenshtein distance."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1,      # deletion
                               dp[i][j - 1] + 1,      # insertion
                               dp[i - 1][j - 1] + cost)  # substitution
        return dp[m][n]

    hyp_clean = hyp.replace(" ", "")
    ref_clean = ref.replace(" ", "")
    dist = levenshtein(hyp_clean, ref_clean)
    return dist / len(ref_clean)

def get_reference_trn(ref_text, utt_id):
    ref_text = " ".join(list(ref_text.strip().replace(" ", ""))) # bcs CER
    ref_text = ref_text.replace("@", "<@>")  # Replace all "@" with "<@>"
    ref_text = ref_text.strip()
    return ref_text

def decoding(speech2text,speech, speech_length, references, alpha, model, utt_id):
    # warped_feat = model.warp_layer(speech, utt_2_warp = alpha)
    batch = {"speech": torch.tensor(speech[0]),
            "warp": alpha,
                }
    results = speech2text(**batch)
    best_result = results[0]
    text = best_result[0]
    text = text.replace("@", "<@>") 
    text = " ".join(list(text.strip().replace(" ", ""))) 
    hyp_text = text.strip()
    logging.info(f"Speech2text output: {hyp_text}")
    ref_text = references.get(utt_id)
    ref_text = get_reference_trn(ref_text, utt_id)
    logging.info(f"Ref Text: {ref_text}")
    cer = character_error_rate(ref_text, hyp_text)
    return cer

def load_raw_speech(input_data):
    for spec in input_data:
        parts = spec.split(",")
        if len(parts) >= 3 and parts[1] == "speech" and parts[2] in ["sound", "wav"]:
            wav_scp_path = parts[0]
            break
    assert wav_scp_path is not None, "No wav path (wav.scp) found."
    scp_reader = {line.split()[0]: line.split()[1] for line in open(wav_scp_path)}
    return scp_reader

def load_fbank_input(input_data):
    for spec in input_data:
        parts = spec.split(",")
        if len(parts) >= 3 and parts[1] == "speech" and parts[2] == "kaldi_ark":
            feat_path = parts[0]
            break
    assert feat_path is not None, "No feature path (feats.scp) found."
    scp_reader = kaldiio.load_scp(feat_path)
    return scp_reader


def run_gradient_descent(warp_espnet_frontend, start, max_steps, lr_init, end_limit, raw, wordtask, speech2text,spk, utt_list, scp_reader, text, model, encode_text, device, max_dur_per_speaker):
    alpha_raw = torch.nn.Parameter(torch.tensor(start, device=device))
    if warp_espnet_frontend:
        model.frontend.logmel.alpha_train = alpha_raw
        # model.warp_layer.alpha1_raw = 1.0
    else:
        model.warp_layer.alpha1_raw = alpha_raw

    optimizer = torch.optim.Adam([alpha_raw], lr=lr_init, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,        # the optimizer whose LR will be changed
        mode='min',       # because we're minimizing loss
        factor=0.5,       # halve the LR when triggered
        patience=14,      # wait 2 steps without improvement before triggering
        min_lr=1e-4       # don't go below this minimum LR
    )
    best_alpha = start
    best_loss = float("inf")
    for step in range(max_steps):
        total_loss = 0.0
        count = 0
        uttcount = 0
        dur = 0
        for utt_id in utt_list:
            if step == 1:
                logging.info(f"utt id: {utt_id}")
            if dur > max_dur_per_speaker:
                break
            uttcount +=1 
            if utt_id not in scp_reader or utt_id not in text:
                continue
            try:
                speech, speech_lengths, current_dur = load_speech_and_lengths(scp_reader, utt_id, raw, device)
                ids = encode_text(text[utt_id])
                if len(ids) == 0:
                    continue
                text_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                text_lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)
                with autocast():
                    loss, _, _ = model.forward(
                        speech=speech,
                        speech_lengths=speech_lengths,
                        text=text_tensor,
                        text_lengths=text_lengths,
                        utt_id=utt_id,
                    )
                    weight = speech_lengths.item() # optional to add weight
                    total_loss += loss 
                    count += 1
                    dur += current_dur 
            except Exception as e:
                logging.warning(f"Failed on {utt_id}: {e}")
                continue
        if count == 0:
            continue
        alpha = alpha_raw
        current_loss = total_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_alpha = alpha.item()
            trigger_times = 0
            logging.info("---Lowest Loss Until now----")
        else:
            trigger_times += 1
            if trigger_times >= end_limit:
                logging.info(f"Stopping, no improvement in loss for {end_limit} steps.")
                break # Exit the inner optimization loop
            logging.info(f"Loss did not improve for {trigger_times}/{end_limit} steps.")

        total_loss.backward()
        lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Step {step}: Speaker {spk} | Loss={total_loss.item():.2f} | Alpha={alpha.item():.4f} | LR={lr:.3f} | Grad = {alpha_raw.grad:.4f}")
        optimizer.step()
        scheduler.step(total_loss.item())
        optimizer.zero_grad()

    logging.info(f"Total duration: {dur:.2f}; Max dur: {max_dur_per_speaker} #utt: {uttcount}")
    return best_alpha

def run_speaker_grid_search(warp_espnet_frontend, raw, decode, wordtask, speech2text,spk, utt_list, scp_reader, text, model, encode_text, device, max_dur, alpha_range=(0.75, 1.20), num_points=10):
    logging.info(f"Running grid search for speaker: {spk}")
    alpha_candidates = np.linspace(*alpha_range, num=num_points)
    best_alpha = 1.0
    best_loss = float("inf")
    grad_gs = True

    for alpha_val in alpha_candidates:
        total_loss = 0.0
        count = 0
        dur = 0
        tot_cer = 0
        alpha_tensor = torch.nn.Parameter(torch.tensor(alpha_val, device=device))
        if warp_espnet_frontend:
            model.frontend.logmel.alpha_train = alpha_tensor
            # model.warp_layer.alpha1_raw = 1.0
        else:
            model.warp_layer.alpha1_raw = alpha_tensor

        for utt_id in utt_list:
            if dur > max_dur:
                continue
            if utt_id not in scp_reader or utt_id not in text:
                continue
            try:
                speech, speech_lengths, current_dur = load_speech_and_lengths(scp_reader, utt_id, raw, device)
                ids = encode_text(text[utt_id])
                if len(ids) == 0:
                    continue
                text_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                text_lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)
                # with torch.no_grad(): # Way faster then with grad
                with autocast():
                    loss, _, _ = model.forward(speech=speech, speech_lengths=speech_lengths, text=text_tensor, text_lengths=text_lengths, utt_id=utt_id)
                    total_loss += loss
                    count += 1
                    dur += current_dur
                    if decode:
                        tot_cer += decoding(speech2text,speech, speech_lengths, text, alpha_tensor, model, utt_id)
            except Exception as e:
                logging.warning(f"Failed on {utt_id}: {e}")
                continue

        if count == 0:
            continue

        log_str = f"Speaker {spk} | Alpha={alpha_val:.3f} | Total Loss={total_loss.item():.2f}"
        if decode:
            cer = tot_cer / count
            log_str += f" | CER: {cer:.2f}"
        if grad_gs:   
            total_loss.backward()
            grad = alpha_tensor.grad
            log_str += f" | Grad: {grad:.2f}"
        logging.info(log_str)
        if total_loss < best_loss:
            best_loss = total_loss.item()
            best_alpha = alpha_val
    

    logging.info(f"Best grid search alpha for {spk}: {best_alpha:.3f} (Total Loss: {best_loss:.2f})")
    logging.info(f"Spk Dur GS = {dur:.3f}")
    return best_alpha

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_data_path_and_name_and_type", action="append", required=True)
    parser.add_argument("--train_shape_file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True) 
    parser.add_argument("--fold_length", type=int)
    parser.add_argument("--bpemodel", type=str)
    parser.add_argument("--token_type", type=str)
    parser.add_argument("--token_list", type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(output_dir / "train.log"),  # Use / with Path object
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    spk2warp_path = output_dir / "spk2alpha"
    utt2warp_path = data_dir / "utt2warp_ref"
    if utt2warp_path.exists():
        os.remove(utt2warp_path)
        logging.info(f"Removed existing file: {utt2warp_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configuration
    lr_init = 0.01                  # 0.01 Is good when initiated by 5 grid search points
    grid_search = True              # Initiate grad desc with grid search
    max_steps = 30                  # For grad desc
    num_gs_points = 10              # Number of grid search points
    max_dur_per_speaker = 15        # Take at least 10 sec to avoid decrease in perf.
    end_limit = 15                  # Quit Grad Desc if no impr for 10 steps
    grad_desc = True        
    alpha_range = (0.75, 1.2)       # For grid search
    wordtask = None
    wordtask = "AVI"                # Optional if you want to prioritize some wordtasks
    decode = False                   # If you want to compute CER in grid search to plot CER(alpha)


    warp_espnet_frontend = False     # If you want to warp pre-processed futures with ESPnet frontend

    logging.info(f"Gradient Descent: {grad_desc}")
    logging.info(f"Grid Search: {grid_search}")


    # Load ASR model
    logging.info("Loading ASR model with Speech2Text...")
    speech2text = Speech2Text(
        asr_train_config=args.config,
        asr_model_file=args.model,
        token_type=args.token_type,
        bpemodel=args.bpemodel,
        lm_train_config=None,
        lm_file=None,
        lm_weight=0.0,
        device=device,
        maxlenratio=0.0,
        minlenratio=0.0,
        ctc_weight=0.5,
        beam_size=10,
        penalty=0.0,
    )
    # By default use_amp is available
    model = speech2text.asr_model
    model.to(device)
    if not hasattr(model, "warp_layer") and not warp_espnet_frontend:
        model.use_warp_layer = True
        model.warp_layer = Filterbankwarping()
    if warp_espnet_frontend == True:
        model.use_warp_frontend = True
    
    model.eval()

    # Load tokenizer and converter
    tokenizer = build_tokenizer(token_type=args.token_type, bpemodel=args.bpemodel)
    with open(args.token_list, encoding="utf-8") as f:
        token_list = [line.strip() for line in f]
    converter = TokenIDConverter(token_list)

    def encode_text(raw_text):
        tokens = tokenizer.text2tokens(raw_text)
        return converter.tokens2ids(tokens)

    
    if model.frontend is not None:
        raw = True
        scp_reader = load_raw_speech(args.train_data_path_and_name_and_type)
    # Assume Kaldi fbank pitch
    else:
        raw = False
        scp_reader = load_fbank_input(args.train_data_path_and_name_and_type)

    data_dir = Path(args.data_dir)
    utt2spk = {line.split()[0]: line.split()[1] for line in open(data_dir / "utt2spk")}
    text = {line.split()[0]: " ".join(line.split()[1:]) for line in open(data_dir / "text")}

    spk2utt = defaultdict(list)
    for utt, spk in utt2spk.items():
        spk2utt[spk].append(utt)

    # Shuffle utterances --> better alpha estimation
    if wordtask is not None:
        logging.info(f"putting {wordtask} tasks first")

    for spk in spk2utt:
        utts = spk2utt[spk]
        matched = [utt for utt in utts if any(task in utt for task in wordtask)]
        unmatched = [utt for utt in utts if all(task not in utt for task in wordtask)]
        random.shuffle(matched)
        random.shuffle(unmatched)
        spk2utt[spk] = matched + unmatched
      
    # Call this to ensure that everything is frozen
    model.warp_init
    
    for spk, utt_list in spk2utt.items():
        logging.info(f"Optimizing alpha for speaker: {spk}")
        if grid_search:
            start = run_speaker_grid_search(warp_espnet_frontend, raw, decode, wordtask, speech2text,spk, utt_list, scp_reader, text, model, encode_text, device, max_dur_per_speaker, alpha_range=alpha_range, num_points=num_gs_points)
        else:
            start = 1.0
        if grad_desc:
            best_alpha = run_gradient_descent(warp_espnet_frontend, start, max_steps, lr_init, end_limit, raw, wordtask, speech2text,spk, utt_list, scp_reader, text, model, encode_text, device, max_dur_per_speaker)
        else:
            best_alpha = start
        final_alpha = best_alpha
        with open(spk2warp_path, "a") as f:
            f.write(f"{spk} {final_alpha:.4f}\n")

        with open(utt2warp_path, "a") as f_utt2warp:
            for utt_id in utt_list:
                f_utt2warp.write(f"{utt_id} {final_alpha:.4f}\n")

    logging.info("Finished optimizing all speakers.")
    logging.shutdown()

if __name__ == "__main__":
    main()

