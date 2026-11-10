"""Phase 34 - build Online Resource 4 (code archive) and Online Resource 3 (predictions, splits,
pairs, audits). No images or weights. Run from the repository root."""
import hashlib, re, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "submission_io"
SECRET = re.compile(r"(api[_-]?key|token|password|secret)\s*[=:]\s*['\"][A-Za-z0-9_\-]{12,}", re.I)


def add(z, p, arc):
    t = p.read_bytes()
    if p.suffix in (".py", ".bat", ".sh", ".md", ".txt") and SECRET.search(t.decode("utf-8", "ignore")):
        raise SystemExit(f"possible secret in {p}")
    z.writestr(arc, t)


with zipfile.ZipFile(OUT / "ESM4_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted((ROOT / "scripts").iterdir()):
        if p.is_file() and p.suffix in (".py", ".sh", ".md") and ".bak" not in p.name:
            add(z, p, f"code/scripts/{p.name}")
    for n in ["requirements.txt", "requirements-lock.txt", "run_round3_gpu.bat", "run_round4_gpu.bat",
              "run_seedpreds_cpu.bat", "run_phase26_27.bat", "run_retfound_full_clean.bat"]:
        if (ROOT / n).exists(): add(z, ROOT / n, f"code/{n}")
    add(z, OUT / "esm4" / "README_ESM4.md", "code/README.md")
    for n in ["aptos_train.csv", "aptos_val.csv", "aptos_test.csv", "aptos_manifest.csv",
              "messidor2_manifest.csv", "idrid_official_manifest.csv"]:
        add(z, ROOT / "data_splits" / n, f"code/data_splits/{n}")

PRED_DIRS = [ROOT / "outputs/phase6", ROOT / "outputs/phase28"]
with zipfile.ZipFile(OUT / "ESM3_predictions.zip", "w", zipfile.ZIP_DEFLATED) as z:
    seen = set()
    for d in PRED_DIRS:
        for p in sorted(d.glob("*__*_preds.csv")):
            if "__IDRiD_preds" in p.name:   # third-party IDRiD copy: superseded, kept only in the audit
                continue
            if p.name in seen: continue
            seen.add(p.name); z.write(p, f"predictions/{p.name}")
    for n in ["aptos_train.csv", "aptos_val.csv", "aptos_test.csv", "idrid_official_manifest.csv"]:
        z.write(ROOT / "data_splits" / n, f"splits/{n}")
    z.write(ROOT / "outputs/phase28/mirror_vs_official_labels.csv", "audits/idrid_copy_vs_official_labels.csv")
    z.write(ROOT / "outputs/phase28/mirror_audit.md", "audits/idrid_copy_audit.md")
    for n in ["ci.csv", "contrasts.csv", "seed_contrasts.csv", "other_contrasts.csv", "pair_contrasts.csv"]:
        if (ROOT / "outputs/phase31" / n).exists(): z.write(ROOT / "outputs/phase31" / n, f"results/{n}")
for n in ["ESM3_predictions.zip", "ESM4_code.zip"]:
    p = OUT / n; print(n, p.stat().st_size // 1024, "KB", hashlib.sha256(p.read_bytes()).hexdigest()[:16])
