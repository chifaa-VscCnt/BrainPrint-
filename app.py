"""
BrainPrint - Streamlit demo
Run:  streamlit run app.py

Folder layout (next to this file):
    brainprint_rf_30.joblib     identification model (30 subjects)
    brainprint_auth_rf.joblib   verification model (20 enrolled people)
    demo_recordings/            a few .edf files (S001R07.edf ...)
    figures/                    PNG plots from the experiments (optional)
    auth_curves.csv             FAR/FRR curves (optional)
"""
import re
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import streamlit as st
from mne.datasets import eegbci
from scipy.signal import welch

mne.set_log_level("ERROR")

HERE = Path(__file__).parent
DEMO_DIR = HERE / "demo_recordings"
FIG_DIR = HERE / "figures"
ID_MODEL = HERE / "brainprint_rf_30.joblib"
AUTH_MODEL = HERE / "brainprint_auth_rf.joblib"
CURVES_CSV = HERE / "auth_curves.csv"

# Same settings as the training pipeline (do not change)
LOW_HZ, HIGH_HZ, WINDOW_SEC = 1.0, 40.0, 4.0
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 40)}

# Results measured in our experiments (edit here if you rerun them)
RESULTS = {
    "id_window": "98.1%", "id_recording": "98.3%", "chance": "3.3%",
    "eer_window": "2.1%", "eer_recording": "1.8%",
}


# ------------------------- SIGNAL PROCESSING -------------------------
def extract_features(windows, sfreq, ch_names):
    """Identical to the training code: band power (log + relative), mean, std."""
    freqs, psd = welch(windows, fs=sfreq, nperseg=int(sfreq * 2), axis=-1)
    df = freqs[1] - freqs[0]
    total_mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    total = psd[..., total_mask].sum(-1) * df

    feats, names = [], []
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        power = psd[..., mask].sum(-1) * df
        feats.append(np.log10(power + 1e-12))
        names += [f"{ch}_{band}_logpow" for ch in ch_names]
        feats.append(power / (total + 1e-12))
        names += [f"{ch}_{band}_relpow" for ch in ch_names]
    feats.append(windows.mean(-1))
    names += [f"{ch}_mean" for ch in ch_names]
    feats.append(windows.std(-1))
    names += [f"{ch}_std" for ch in ch_names]
    return np.concatenate(feats, axis=1), names


def process_file(path):
    """EDF file -> filtered preview signal + one feature row per 4 s window."""
    raw = mne.io.read_raw_edf(str(path), preload=True)
    eegbci.standardize(raw)
    raw.filter(LOW_HZ, HIGH_HZ, fir_design="firwin")
    sfreq = raw.info["sfreq"]
    epochs = mne.make_fixed_length_epochs(raw, duration=WINDOW_SEC, preload=True, verbose=False)
    windows = epochs.get_data() * 1e6                      # microvolts
    X, names = extract_features(windows, sfreq, raw.ch_names)
    n_prev = int(10 * sfreq)
    return {"X": X, "names": names, "ch_names": list(raw.ch_names), "sfreq": sfreq,
            "preview": raw.get_data()[:, :n_prev] * 1e6,
            "duration": raw.times[-1]}


@st.cache_data(show_spinner="Filtering the EEG and extracting features...")
def process_cached(path_str, cache_key):
    return process_file(path_str)


@st.cache_resource
def load_bundle(path_str):
    return joblib.load(path_str)


def mean_probs(bundle, X, names, n_use):
    """Average the model's probabilities over the first n_use windows."""
    df = pd.DataFrame(X[:n_use], columns=names).reindex(columns=bundle["features"])
    if df.isna().any().any():
        raise ValueError("This recording does not have the same channels as the training data.")
    return bundle["model"].predict_proba(df).mean(axis=0)


def parse_name(name):
    """'S012R07.edf' -> (12, 7); otherwise (None, None)."""
    m = re.search(r"S(\d{3})R(\d{2})", name.upper())
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


# ------------------------- UI -------------------------
def main():
    st.set_page_config(page_title="BrainPrint", page_icon="🧠", layout="wide")
    st.title("🧠 BrainPrint")
    st.caption("Can EEG brain signals help recognize a person? Student prototype on the public "
               "PhysioNet EEG Motor Movement/Imagery dataset (30 subjects).")

    if not ID_MODEL.exists() or not AUTH_MODEL.exists():
        st.error("Model files not found. Put brainprint_rf_30.joblib and "
                 "brainprint_auth_rf.joblib next to app.py.")
        st.stop()
    id_bundle = load_bundle(str(ID_MODEL))
    auth_bundle = load_bundle(str(AUTH_MODEL))

    tab_demo, tab_perf, tab_about = st.tabs(["Live demo", "Model performance", "Limits and ethics"])

    # ---------------- sidebar ----------------
    st.sidebar.header("1. Recording")
    source = st.sidebar.radio("Source", ["Demo recording", "Upload an EDF file"], key="source")
    path, display_name = None, ""
    if source == "Demo recording":
        files = sorted(DEMO_DIR.glob("*.edf")) if DEMO_DIR.exists() else []
        if not files:
            st.sidebar.warning("No file found in demo_recordings/.")
        else:
            choice = st.sidebar.selectbox("Choose a recording", [f.name for f in files], key="demo_file")
            path, display_name = DEMO_DIR / choice, choice
    else:
        up = st.sidebar.file_uploader("EDF recording", type=["edf"], key="upload")
        if up is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
            tmp.write(up.getvalue())
            tmp.close()
            path, display_name = Path(tmp.name), up.name

    st.sidebar.header("2. Mode")
    mode = st.sidebar.radio("What should the system do?",
                            ["Identification: who is this?", "Verification: is this really person X?"],
                            key="mode")

    with tab_demo:
        if path is None:
            st.info("Choose or upload a recording in the sidebar to start.")
        else:
            try:
                key = f"{display_name}-{path.stat().st_size}"
                rec = process_cached(str(path), key)
            except Exception as e:                       # noqa: BLE001
                st.error(f"Could not read this recording: {e}")
                st.stop()

            n_total = len(rec["X"])
            if n_total < 1:
                st.error("Recording too short (need at least 4 seconds).")
                st.stop()
            st.sidebar.header("3. Signal used")
            n_use = st.sidebar.slider("Number of 4 s windows used", 1, n_total, n_total, key="n_use")
            st.sidebar.caption(f"= {n_use * WINDOW_SEC:.0f} seconds of EEG")

            true_subj, true_run = parse_name(display_name)

            # ---- signal + band power
            c1, c2 = st.columns([3, 2])
            with c1:
                st.subheader("EEG signal (first 10 s, filtered 1-40 Hz)")
                wanted = [c for c in ["Fz", "Cz", "C3", "Oz"] if c in rec["ch_names"]] or rec["ch_names"][:4]
                fig, axes = plt.subplots(len(wanted), 1, sharex=True, figsize=(8, 1.5 * len(wanted) + 0.8))
                axes = np.atleast_1d(axes)
                t = np.arange(rec["preview"].shape[1]) / rec["sfreq"]
                for ax, ch in zip(axes, wanted):
                    ax.plot(t, rec["preview"][rec["ch_names"].index(ch)], lw=0.7)
                    ax.set_ylabel(f"{ch} (uV)", fontsize=8)
                axes[-1].set_xlabel("Time (s)")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with c2:
                st.subheader("Average band power")
                names = np.array(rec["names"])
                bp = {b: rec["X"][:n_use][:, [n.endswith(f"_{b}_relpow") for n in names]].mean()
                      for b in BANDS}
                st.bar_chart(pd.Series(bp, name="relative power"))
                st.caption(f"{n_total} windows in this file "
                           f"({rec['duration']:.0f} s). File: {display_name}")

            st.divider()

            # ---- identification
            if mode.startswith("Identification"):
                st.subheader("Identification: who is this?")
                try:
                    p = mean_probs(id_bundle, rec["X"], rec["names"], n_use)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                classes = id_bundle["classes"]
                order = np.argsort(p)[::-1]
                best = classes[order[0]]
                m1, m2 = st.columns(2)
                m1.metric("Predicted identity", f"Subject {best}")
                m2.metric("Confidence", f"{p[order[0]] * 100:.0f}%")
                top = pd.DataFrame({"Subject": [f"S{classes[i]:03d}" for i in order[:5]],
                                    "Confidence": p[order[:5]]}).set_index("Subject")
                st.bar_chart(top)
                if true_subj is not None:
                    if best == true_subj:
                        st.success(f"Ground truth (from file name): subject {true_subj}. Correct.")
                    else:
                        st.error(f"Ground truth (from file name): subject {true_subj}. Wrong prediction.")

            # ---- verification
            else:
                st.subheader("Verification: is this really person X?")
                enrolled = [int(s) for s in auth_bundle["enrolled"]]
                strangers = [int(s) for s in auth_bundle["strangers"]]
                v1, v2 = st.columns(2)
                claimed = v1.selectbox("Claimed identity (enrolled users)", enrolled,
                                       format_func=lambda s: f"Subject {s}", key="claimed")
                thr = v2.slider("Acceptance threshold", 0.0, 1.0, 0.30, 0.01, key="thr")
                v2.caption("Higher = stricter: fewer impostors accepted, but more real users rejected.")
                try:
                    p = mean_probs(auth_bundle, rec["X"], rec["names"], n_use)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                score = float(p[auth_bundle["classes"].index(claimed)])
                accept = score >= thr
                m1, m2, m3 = st.columns(3)
                m1.metric("Score for the claimed identity", f"{score:.2f}")
                m2.metric("Threshold", f"{thr:.2f}")
                m3.metric("Decision", "ACCEPTED" if accept else "REJECTED")
                if accept:
                    st.success(f"Access granted as subject {claimed}.")
                else:
                    st.error("Access denied.")

                if true_subj is not None:
                    if true_subj == claimed:
                        role, should = "genuine attempt (the real person)", True
                    elif true_subj in enrolled:
                        role, should = "impostor: an enrolled person claiming someone else", False
                    elif true_subj in strangers:
                        role, should = "impostor: a stranger the system has never seen", False
                    else:
                        role, should = None, None
                    if role:
                        ok = accept == should
                        msg = f"Ground truth (from file name): {role}. "
                        if ok:
                            st.info(msg + "The decision is correct.")
                        else:
                            st.warning(msg + ("False rejection (counts in FRR)." if should
                                              else "False acceptance (counts in FAR)."))
                st.caption(f"In our experiments the equal-error threshold was about "
                           f"{auth_bundle.get('threshold_recording', 0.3):.2f}.")

    # ---------------- performance tab ----------------
    with tab_perf:
        st.subheader("Measured on recordings never used for training")
        a, b, c = st.columns(3)
        a.metric("Identification accuracy", RESULTS["id_window"], f"random guess {RESULTS['chance']}",
                 delta_color="off")
        b.metric("Verification error (EER)", RESULTS["eer_window"],
                 "FAR = FRR point", delta_color="off")
        c.metric("Subjects", "30", "20 enrolled + 10 strangers", delta_color="off")
        st.markdown(
            "| Test | Result |\n|---|---|\n"
            f"| Identification, per 4 s window | {RESULTS['id_window']} |\n"
            f"| Identification, per recording | {RESULTS['id_recording']} |\n"
            f"| Verification EER, per window | {RESULTS['eer_window']} |\n"
            f"| Verification EER, per recording | {RESULTS['eer_recording']} |\n"
            "| Train on tasks, test on another task | 99.2% |\n"
            "| Train on tasks, test on eyes open | 76.4% |\n"
            "| Train on rest, test on tasks | 64.7% |\n"
            "| Train on tasks, test on eyes closed | 44.7% |")
        if CURVES_CSV.exists():
            cur = pd.read_csv(CURVES_CSV)
            lvl = st.radio("Level", ["recording", "window"], horizontal=True, key="lvl")
            cur = cur[cur["level"] == lvl].set_index("threshold")[["FRR", "FAR_all", "FAR_strangers"]] * 100
            st.line_chart(cur)
            st.caption("Error rate (%) as a function of the acceptance threshold.")
        imgs = [("confusion_matrix_30.png", "Identification confusion matrix"),
                ("auth_far_frr.png", "Verification: FAR / FRR vs threshold"),
                ("state_shift.png", "Does identity survive a change of mental state?"),
                ("acc_vs_subjects_20.png", "Accuracy vs number of subjects"),
                ("acc_vs_channels.png", "Accuracy vs number of channels"),
                ("ablation.png", "Ablation: which features carry identity?")]
        shown = [(f, t) for f, t in imgs if (FIG_DIR / f).exists()]
        for i in range(0, len(shown), 2):
            cols = st.columns(2)
            for col, (f, t) in zip(cols, shown[i:i + 2]):
                col.image(str(FIG_DIR / f), caption=t, use_container_width=True)

    # ---------------- limits tab ----------------
    with tab_about:
        st.subheader("Limits")
        st.markdown(
            "- All recordings of one person come from **one session** (same day, same cap). "
            "We cannot claim it works across days.\n"
            "- Only **30 people**, recorded under controlled conditions.\n"
            "- Accuracy drops when the **mental state changes** (eyes closed: 44.7%).\n"
            "- Strangers are harder to reject than enrolled people posing as someone else.\n"
            "- Part of the signal may reflect recording conditions (electrode contact, noise), "
            "not only the brain.")
        st.subheader("Ethics and privacy")
        st.markdown(
            "- Brain data is **highly sensitive**, and unlike a password you cannot change your brain.\n"
            "- A biometric system needs **consent** and strict storage rules.\n"
            "- Wording: EEG *may* help distinguish individuals under controlled conditions.")
        st.subheader("Two modes of biometrics")
        st.markdown(
            "- **Identification** (1 to N): who is this person among the known people?\n"
            "- **Verification** (1 to 1): is this person really who they claim to be?")


if __name__ == "__main__":
    main()