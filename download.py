import requests
import re
import time
from pathlib import Path

# =========================================================
# MIMIC-II Smart Scanner v6 - Multi-directory
# Based on your v5 script — scans /30/ through /45/
# to find 300 valid patients with PLETH + ABP
# =========================================================

OUT_DIR         = r"C:\MIMIC 2"
VALID_LIST      = r"C:\Users\yahia\Desktop\MIMIC2_valid_records.txt"
SCANNED_LIST    = r"C:\Users\yahia\Desktop\MIMIC2_scanned_ids.txt"
TARGET_PATIENTS = 2500

PPG_SIGNALS   = ["PLETH", "PPG", "PHOTO"]
ABP_SIGNALS   = ["ABP", "ART"]
BASE_URL      = "https://archive.physionet.org/physiobank/database/mimic2wdb"
REQUEST_DELAY = 0.3

# All subdirectories to scan — covers thousands of patients
SUBDIRS = ["30", "31", "32", "33", "34", "35", "36", "37",
           "38", "39"]

print("=" * 60)
print("MIMIC-II SCANNER v6 - Multi-directory")
print(f"Target: {TARGET_PATIENTS} patients with PLETH + ABP")
print(f"Scanning subdirs: {SUBDIRS}")
print("=" * 60)

# ---------------------------------------------------------
# HELPERS (same logic as your v5)
# ---------------------------------------------------------

def get_patient_folders(subdir):
    url = f"{BASE_URL}/{subdir}/"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"  Could not access /{subdir}/ (status {r.status_code})")
            return []
        folders = re.findall(r'href="(\d+)/?"', r.text)
        if not folders:
            folders = re.findall(r'href="([^"]+)/"', r.text)
            folders = [f for f in folders if re.match(r'^\d+$', f)]
        return folders
    except Exception as e:
        print(f"  Error accessing /{subdir}/: {e}")
        return []


def get_segments(subdir, patient_id):
    url = f"{BASE_URL}/{subdir}/{patient_id}/"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return []
        segments = re.findall(r'href="([^"/]+\.hea)"', r.text)
        segments += re.findall(r'>([^<\s]+\.hea)<', r.text)
        segments = list(set([
            s.replace('.hea', '').strip()
            for s in segments
            if 'layout' not in s.lower() and s.strip()
        ]))
        return segments
    except:
        return []


def check_hea(subdir, patient_id, segment_id):
    url = f"{BASE_URL}/{subdir}/{patient_id}/{segment_id}.hea"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return False, False
        content = r.text.upper()
        has_ppg = any(s in content for s in PPG_SIGNALS)
        has_abp = any(s in content for s in ABP_SIGNALS)
        return has_ppg, has_abp
    except:
        return False, False


def download_patient(subdir, patient_id):
    patient_dir = Path(OUT_DIR) / str(patient_id)
    patient_dir.mkdir(parents=True, exist_ok=True)

    if any(patient_dir.glob("*.dat")):
        return True  # already downloaded

    url = f"{BASE_URL}/{subdir}/{patient_id}/"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return False
        files = re.findall(r'href="([^"]+\.(hea|dat))"', r.text)
        files = [f[0] for f in files]
        if not files:
            raw = re.findall(r'([a-zA-Z0-9_\-]+\.(hea|dat))', r.text)
            files = [f[0] for f in raw]
        if not files:
            return False

        for fname in files:
            fpath = patient_dir / fname
            if fpath.exists():
                continue
            try:
                fr = requests.get(
                    f"{BASE_URL}/{subdir}/{patient_id}/{fname}",
                    timeout=120, stream=True
                )
                if fr.status_code == 200:
                    with open(fpath, 'wb') as f:
                        for chunk in fr.iter_content(8192):
                            f.write(chunk)
                time.sleep(0.1)
            except:
                continue
        return True
    except:
        return False


# ---------------------------------------------------------
# RESUME: load previously found valid patients
# ---------------------------------------------------------
valid_patients = []  # list of (subdir, patient_id)
valid_list_path = Path(VALID_LIST)

if valid_list_path.exists():
    with open(valid_list_path) as f:
        for line in f:
            line = line.strip()
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 2:
                    valid_patients.append((parts[0].strip(), parts[1].strip()))
    if valid_patients:
        print(f"\nResuming — {len(valid_patients)} valid patients already found.")

already_downloaded_ids = set(p[1] for p in valid_patients)

# ---------------------------------------------------------
# RESUME: load previously scanned (rejected) patients
# ---------------------------------------------------------
scanned_ids = set()
scanned_list_path = Path(SCANNED_LIST)

if scanned_list_path.exists():
    with open(scanned_list_path) as f:
        for line in f:
            pid = line.strip()
            if pid:
                scanned_ids.add(pid)
    if scanned_ids:
        print(f"Skipping — {len(scanned_ids)} previously rejected patients.")


def save_scanned():
    scanned_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scanned_list_path, 'w') as f:
        for pid in scanned_ids:
            f.write(pid + "\n")


# ---------------------------------------------------------
# MAIN SCAN LOOP
# ---------------------------------------------------------
total_scanned = 0

for subdir in SUBDIRS:
    if len(valid_patients) >= TARGET_PATIENTS:
        break

    print(f"\n{'='*60}")
    print(f"Scanning subdirectory: /{subdir}/")

    patient_folders = get_patient_folders(subdir)
    if not patient_folders:
        continue

    print(f"Found {len(patient_folders)} patient folders in /{subdir}/")

    for patient_id in patient_folders:
        if len(valid_patients) >= TARGET_PATIENTS:
            break

        # Skip valid patients already downloaded
        if patient_id in already_downloaded_ids:
            continue

        # Skip previously rejected patients — no network request needed
        if patient_id in scanned_ids:
            continue

        # Already on disk from a previous run
        patient_dir = Path(OUT_DIR) / str(patient_id)
        if patient_dir.exists() and any(patient_dir.glob("*.dat")):
            print(f"  [{patient_id}] Already on disk — adding")
            valid_patients.append((subdir, patient_id))
            already_downloaded_ids.add(patient_id)
            continue

        total_scanned += 1
        if total_scanned % 20 == 0:
            print(f"\n  --- {total_scanned} scanned | {len(valid_patients)}/{TARGET_PATIENTS} valid ---\n")

        segments = get_segments(subdir, patient_id)
        if not segments:
            scanned_ids.add(patient_id)
            save_scanned()
            time.sleep(REQUEST_DELAY)
            continue

        found_valid = False
        for seg in segments[:5]:
            has_ppg, has_abp = check_hea(subdir, patient_id, seg)
            if has_ppg and has_abp:
                print(f"  [{patient_id}] valid - downloading...", end="", flush=True)
                ok = download_patient(subdir, patient_id)
                if ok:
                    valid_patients.append((subdir, patient_id))
                    already_downloaded_ids.add(patient_id)
                    found_valid = True
                    print(f" done [{len(valid_patients)}/{TARGET_PATIENTS}]")

                    # Save progress after every patient
                    valid_list_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(valid_list_path, 'w') as f:
                        f.write("MIMIC-II Valid Patients (subdir | patient_id)\n")
                        f.write("=" * 40 + "\n\n")
                        for sd, pid in valid_patients:
                            f.write(f"{sd} | {pid}\n")
                else:
                    print(" download failed")
                break

            time.sleep(REQUEST_DELAY)

        # Mark as scanned regardless of outcome
        if not found_valid:
            scanned_ids.add(patient_id)
            save_scanned()

        time.sleep(REQUEST_DELAY)

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"Total scanned:   {total_scanned}")
print(f"Valid patients:  {len(valid_patients)}")
print(f"Output dir:      {OUT_DIR}")
print(f"Patient list:    {VALID_LIST}")
print(f"\nNext step: update INPUT_DIR in your preprocessing script to {OUT_DIR}")
print(f"Then run: python preprocess_mimic2.py")
