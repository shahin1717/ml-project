# TODO — Re-run Experiments Before Finalising Report

## Why We Need to Re-run

The figures and numbers currently in `figures/` and `report/report.tex`
were generated with **incorrect dataset sizes** that did not match the
project specification:

| Dataset | Spec Requires | Old (wrong) | Fixed |
|---|---|---|---|
| Adult Income | All 48,842 | 5,000 (subsampled) | ✅ Fixed |
| MNIST 3-vs-8 | ≥ 5,000 | 2,000 | ✅ Fixed → 6,000 |
| Breast Cancer | 569 (all) | 569 | ✅ Was already correct |

The fix is already applied in `experiments/utils.py` — **nothing else
needs to be changed**. You just need to re-run.

---

## How to Re-run (one command)

```bash
cd /home/shahin/ml-project
conda activate mlaiac
python -m src.experiments.run_all 2>&1 | tee run_all_output.log
```

Expected runtime: **~2–2.5 hours** on a standard laptop.  
The script overwrites all figures in `figures/` and prints results to `run_all_output.log`.

---

## What to Do After the Re-run

1. **Copy the updated numbers** from `run_all_output.log` into `report/report.tex`:
   - Table 2 (Baseline): search `% UPDATE AFTER RERUN` — or find Experiment 1 output in the log
   - Table 3 (Head-to-head): Experiment 4 output (mean ± std per model per dataset)
   - Table 4 (Bias-Variance): Experiment 6 output (these are already correct — ran on BC only)
   - DBSCAN ARI and K-Means ARI: Experiment 7 output

2. **Check the figures** — they will be regenerated automatically in `figures/`.

3. **Compile the PDF**:
   ```bash
   cd report/
   pdflatex report.tex && pdflatex report.tex
   ```
   > If `pdflatex` is not installed: paste `report.tex` into **Overleaf** (overleaf.com) and compile there.

---

## What Is Already Done (no changes needed)

- [x] All algorithms implemented and tested (93% coverage)
- [x] Dataset sizes corrected in `experiments/utils.py`
- [x] `download_data.py` created for clean-clone reproducibility
- [x] `report/report.tex` written with all sections, figures, math, and explanations
- [x] Bias-Variance numbers confirmed correct (ran on Breast Cancer with B=100)
- [x] All 31 figures exist (generated with old sizes — will be refreshed after re-run)
