"""Run the whole pipeline in order (about 40-60 minutes, dominated by step 04). Usage: python experiments/run_all.py [--skip-ablation]"""
import subprocess, sys, pathlib
steps = ["01_ingest_and_quality.py", "02_feature_redundancy.py", "03_train_evaluate.py", "04_ablation_and_comparison.py",
         "05_label_circularity.py", "06_explain_and_errors.py", "07_ablation_figures.py"]
if "--skip-ablation" in sys.argv:
    steps = [s for s in steps if not s.startswith(("04", "07"))]
here = pathlib.Path(__file__).parent; logs = here.parent / "results/logs"; logs.mkdir(parents=True, exist_ok=True)
for s in steps:
    print("running", s, flush=True)
    with open(logs / (s[:-3] + ".log"), "w") as f:
        r = subprocess.run([sys.executable, str(here / s)], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"{s} failed; see results/logs")
print("all steps finished; now run: python -m pytest")
