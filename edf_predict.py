

import mne
import numpy as np
from scipy.signal import welch
from mne_connectivity import spectral_connectivity_epochs

# Cuban convention: beta 12–19 Hz
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 19)}

# ------------------------
# Normalize channel names
# ------------------------
def normalize_channel_names(raw):
    mapping = {}
    for ch in raw.ch_names:
        base = ch.replace("-A1", "").replace("-A2", "").upper()
        if base == "T7": base = "T3"
        if base == "T8": base = "T4"
        if base == "P7": base = "T5"
        if base == "P8": base = "T6"
        if base.startswith("FP"):
            base = "Fp" + base[2:]
        elif len(base) > 1:
            base = base[0] + base[1:].lower()
        mapping[ch] = base
    raw.rename_channels(mapping)
    return raw

# ------------------------
# Preprocess EDF
# ------------------------
def prep_raw(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw = normalize_channel_names(raw)
    try:
        raw.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass
    raw.set_eeg_reference("average", projection=False)
    raw.filter(1., 40., fir_design="firwin", verbose=False)
    raw.notch_filter([50, 60], verbose=False)
    return raw

# ------------------------
# Compute relative powers per channel
# ------------------------
def compute_rel_powers(raw, bands=BANDS):
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True, verbose=False)
    data = epochs.get_data()
    sfreq = raw.info["sfreq"]
    rel = {}
    for i, ch in enumerate(epochs.ch_names):
        sig = data[:, i, :].ravel()
        f, pxx = welch(sig, fs=sfreq, nperseg=int(sfreq*1.0))
        total = np.trapz(pxx[(f >= 1) & (f <= 30)], f[(f >= 1) & (f <= 30)]) + 1e-12
        for band, (f1, f2) in bands.items():
            m = (f >= f1) & (f <= f2)
            bp = np.trapz(pxx[m], f[m])
            rel[f"Rel_{band}_{ch}"] = float(bp / total)
    return rel

# ------------------------
# Compute 9 spectral markers
# ------------------------
def compute_spectral_markers(rel):
    def safe_mean(keys):
        vals = [rel.get(k) for k in keys if k in rel]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.nanmean(vals)) if vals else np.nan

    F = ["Fp1", "Fp2", "F3", "F4", "Fz"]
    T = ["T3", "T4", "T5", "T6"]
    C = ["C3", "C4", "Cz"]
    P = ["P3", "P4", "Pz"]
    Post = ["P3", "P4", "Pz", "O1", "O2"]

    rel_theta_F = safe_mean([f"Rel_theta_{ch}" for ch in F])
    rel_theta_T = safe_mean([f"Rel_theta_{ch}" for ch in T])
    theta_FT = np.nanmean([v for v in [rel_theta_F, rel_theta_T] if not np.isnan(v)]) \
                if (not np.isnan(rel_theta_F) or not np.isnan(rel_theta_T)) else np.nan

    alpha_F = safe_mean([f"Rel_alpha_{ch}" for ch in F])
    beta_F  = safe_mean([f"Rel_beta_{ch}" for ch in F])
    beta_C  = safe_mean([f"Rel_beta_{ch}" for ch in C])
    beta_P  = safe_mean([f"Rel_beta_{ch}" for ch in P])
    beta_Post = safe_mean([f"Rel_beta_{ch}" for ch in Post])

    beta_diffuse = np.nan
    if not np.isnan(beta_C) or not np.isnan(beta_P):
        beta_diffuse = np.nanmean([v for v in [beta_C, beta_P] if not np.isnan(v)]) - (beta_F if not np.isnan(beta_F) else 0.0)

    f3 = rel.get("Rel_alpha_F3"); f4 = rel.get("Rel_alpha_F4")
    faa = np.nan
    if f3 is not None and f4 is not None and (f3 + f4) != 0:
        faa = (f3 - f4) / (f3 + f4) * 200.0

    return {
        "theta_FT": theta_FT,
        "alpha_F": alpha_F,
        "beta_F": beta_F,
        "FAA": faa,
        "beta_diffuse": beta_diffuse,
        "beta_Post": beta_Post,
        "theta_Pz": rel.get("Rel_theta_Pz"),
        "theta_P3": rel.get("Rel_theta_P3"),
        "theta_T5": rel.get("Rel_theta_T5"),
    }

# ------------------------
# Compute coherence (new API)
# ------------------------
def compute_coherence_needed(raw, needed_cols, bands=BANDS):
    ch_names = raw.ch_names
    feats = {}
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True, verbose=False)

    for col in needed_cols:
        for band, (f1, f2) in bands.items():
            if band in col.lower():
                parts = col.replace("_", ".").split(".")
                chans = [p for p in parts if p in ch_names]
                if len(chans) == 2:
                    a, b = chans
                    picks = mne.pick_channels(ch_names, include=[a, b])
                    if len(picks) == 2:
                        con = spectral_connectivity_epochs(
                            epochs, method="coh", mode="multitaper",
                            fmin=f1, fmax=f2, faverage=True,
                            sfreq=raw.info["sfreq"], verbose=False
                        )
                        feats[col] = float(np.mean(con.get_data()))
    return feats

# ------------------------
# Main extraction
# ------------------------
def extract_features_for_model(edf_path, feature_cols):
    raw = prep_raw(edf_path)
    rel_needed = [c for c in feature_cols if c.startswith("Rel_")]
    coh_needed = [c for c in feature_cols if c.startswith("COH")]
    # compute per-channel relative powers
    rel = compute_rel_powers(raw)
    # compute 9 group-level spectral markers
    spectral_markers = compute_spectral_markers(rel)
    coh = compute_coherence_needed(raw, coh_needed) if coh_needed else {}

    feats = []
    for col in feature_cols:
        if col in rel:
            feats.append(rel[col])
        elif col in spectral_markers:
            feats.append(spectral_markers[col])
        elif col in coh:
            feats.append(coh.get(col, np.nan))
        else:
            feats.append(np.nan)
    return np.array(feats, dtype=float)
