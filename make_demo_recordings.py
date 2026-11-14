import joblib, shutil
from pathlib import Path

# Folder that contains mne_data and your PNGs (the downloaded Drive folder).
# ".." works if brainprint_app is inside it; otherwise write the full path.
SEARCH_IN = Path("..")

aut = joblib.load("brainprint_auth_rf.joblib")
enr, strg = list(aut["enrolled"]), list(aut["strangers"])
picks = [enr[0], enr[7], enr[15], strg[0], strg[5]]   # 3 enrolled + 2 strangers

Path("demo_recordings").mkdir(exist_ok=True)
for s in picks:
    for r in (7, 8):                                   # runs never used for training
        name = f"S{s:03d}R{r:02d}.edf"
        matches = [p for p in SEARCH_IN.rglob(name) if "demo_recordings" not in p.parts]
        if matches:
            shutil.copy(matches[0], Path("demo_recordings") / name)
        else:
            print("NOT FOUND:", name)

Path("figures").mkdir(exist_ok=True)
for f in ["confusion_matrix_30.png", "auth_far_frr.png", "state_shift.png",
          "acc_vs_subjects_20.png", "acc_vs_channels.png", "ablation.png"]:
    matches = [p for p in SEARCH_IN.rglob(f) if "figures" not in p.parts]
    if matches:
        shutil.copy(matches[0], Path("figures") / f)

print("Enrolled:", sorted(picks[:3]), "| Strangers:", sorted(picks[3:]))