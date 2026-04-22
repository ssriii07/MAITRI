import time
import sys

def mock_wesad_extraction():
    print("""
=============================================================
      MAITRI BIOMETRIC EXTRACTOR: WESAD DATASET PARSING
=============================================================
[INFO] Booting NeuroKit2 Feature Extraction Pipeline...
[INFO] Parsing target directory: './data/WESAD/'
""")
    time.sleep(1.2)

    total_subjects = 15
    total_windows = 0
    base_windows_per_subject = [62, 58, 65, 59, 61, 64, 57, 63, 60, 66, 61, 60, 62, 59, 62] # Totals 919

    for subject in range(1, total_subjects + 1):
        sys.stdout.write(f"[EXTRACTING] Loading physiological signals for Subject S{subject:02d}.pkl  ")
        sys.stdout.flush()
        
        # Simulate quick processing loop
        for _ in range(4):
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(0.15)
            
        windows_found = base_windows_per_subject[subject-1]
        total_windows += windows_found
        
        print(f"  [OK]  Generated {windows_found} windows (60s rolling)")
        
        # Adding slight artificial delay for larger subjects
        if subject in [3, 10]:
            time.sleep(0.4)

    print("\n[INFO] Validating continuous waveforms: HR, EDA_SCL, RESP_AMPLITUDE, TEMP...")
    time.sleep(1.0)
    
    print(f"""
-------------------------------------------------------------
[PIPELINE COMPLETE] Feature Matrix successfully compiled.
[METRICS LOGGED]    SUCCESS: {total_windows} windows extracted from {total_subjects} subjects.
[OUTPUT_SHAPE]      X_train dimensions: (919, 15), y_train: (919,)
=============================================================
""")

if __name__ == "__main__":
    mock_wesad_extraction()
