# 🧠 BrainPrint — EEG-Based Biometric Identification & Verification

**BrainPrint** is a machine learning prototype exploring whether electroencephalogram (EEG) brainwave patterns can serve as a biometric signature to identify or authenticate individuals. Built using the public [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org), the project includes a full signal-processing pipeline, ablation and stress-test experiments, and an interactive **Streamlit** web application.

---

## ✨ Key Features

* **Two Biometric Modes:**
  * **Identification (1-to-N):** Predicts which of the 30 known subjects generated an unseen EEG recording.
  * **Verification / Authentication (1-to-1):** Evaluates whether a user claiming a specific identity should be **ACCEPTED** or **REJECTED** against a tunable confidence threshold, testing both enrolled impostors and completely unknown strangers.
* **Signal Processing Pipeline:** Band-pass filtering (`1–40 Hz`), fixed-length windowing (`4.0 s`), and Welch Power Spectral Density (PSD) feature extraction across 5 frequency bands (*delta, theta, alpha, beta, gamma*) plus time-domain statistics (`768 features` across 64 channels).
* **Leak-Free Evaluation:** Models are trained strictly on **Runs 1–6** and evaluated on unseen recordings (**Runs 7–8**) as well as Leave-One-Run-Out (LORO) cross-validation.
* **Interactive Analytics:** Real-time EEG waveform preview, relative band-power visualization, adjustable signal duration slider, and interactive FAR/FRR trade-off curves.

---

## 📊 Experimental Results (30 Subjects)

All evaluation metrics were measured on recordings never seen during training:

| Task / Experiment | Metric | Result |
| :--- | :--- | :--- |
| **Identification** (per 4 s window) | Accuracy (vs. 3.3% random chance) | **98.1%** |
| **Identification** (full recording average) | Accuracy | **98.3%** |
| **Verification** (20 enrolled / 10 strangers, per window) | Equal Error Rate (EER) | **2.1% ± 0.4%** |
| **Verification** (20 enrolled / 10 strangers, per recording) | Equal Error Rate (EER) | **1.8% ± 0.7%** |
| **Mental State Shift** (Train on motor tasks → test on tasks) | Accuracy | **99.2%** |
| **Mental State Shift** (Train on motor tasks → test on eyes open) | Accuracy | **76.4%** |
| **Mental State Shift** (Train on motor tasks → test on eyes closed) | Accuracy | **44.7%** |

---

## 🚀 Quickstart & Local Installation

### 1. Clone the repository
```bash
git clone https://github.com<your-username>/BrainPrint.git
cd BrainPrint
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit demo
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`. You can select a pre-loaded recording from `demo_recordings/` or upload any PhysioNet `.edf` file (e.g., `S001R07.edf`).

*(Optional)* To reproduce the full feature extraction and model training pipeline from scratch, open `notebook.ipynb` and run all cells—MNE will automatically download the required PhysioNet `.edf` recordings.

---

## 📂 Project Structure

```text
├── app.py                      # Interactive Streamlit web application
├── brainprint_features.py      # Band-pass filtering and PSD feature extraction
├── notebook.ipynb              # End-to-end training, ablation, and stress-test experiments
├── brainprint_rf_30.joblib     # Pre-trained Random Forest identification model (30 subjects)
├── brainprint_auth_rf.joblib   # Pre-trained Random Forest verification model (20 enrolled)
├── auth_curves.csv             # False Acceptance Rate (FAR) & False Rejection Rate (FRR) summary
├── requirements.txt            # Python package dependencies
├── demo_recordings/            # Sample unseen .edf EEG recordings for testing the app
└── figures/                    # Confusion matrices, FAR/FRR curves, and ablation plots
```

---

## ⚠️ Limitations & Ethical Considerations

* **Single-Session Data:** All recordings per subject in the PhysioNet dataset were collected during a single session. Cross-day stability is not validated here.
* **Sensitivity to Mental State:** Recognition accuracy drops noticeably when a user's cognitive state changes between enrollment and testing (e.g., motor imagery vs. eyes-closed rest).
* **Biometric Privacy:** Unlike passwords or tokens, neural signals are irrevocable biological data. Any real-world EEG authentication system requires explicit user consent, local on-device processing, and strict encryption standards.

---

## 📜 Acknowledgments

* Dataset provided by **PhysioNet**: Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004). *BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.* IEEE Transactions on Biomedical Engineering, 51(6), 1034–1043.
