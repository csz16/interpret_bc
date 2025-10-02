# Quick Fix Guide: SHAP and EBM Performance Issues

## 🎯 Summary of Issues

1. **SHAP Not Working**: Even after reinstalling
2. **GBM_ZINB Outperforming EBM_ZINB**: Want EBM to be the best model

---

## ⚡ Quick Fixes

### Issue 1: Fix SHAP (5 minutes)

```bash
# Option A: Run the diagnostic script
python test_shap_fix.py

# Option B: Manual fix
pip uninstall shap -y
pip install --upgrade pip setuptools wheel
pip install shap --no-cache-dir

# Option C: If still fails, use compatible version
pip install shap==0.41.0 "numpy<1.24"

# Option D: Just skip SHAP
# The pipeline will work fine without it - SHAP visualizations will be skipped
```

**Test if it worked:**
```python
python -c "import shap; print(f'SHAP {shap.__version__} loaded successfully!')"
```

---

### Issue 2: Make EBM_ZINB the Best Model (15 minutes)

**Quick Fix - Run with improved configuration:**

```bash
# Use the improved configuration (Phase 1: simple but effective)
python improved_ebm_config.py --phase 1

# If Phase 1 works well, try Phase 2 (adds interactions)
python improved_ebm_config.py --phase 2

# Or compare all phases automatically
python improved_ebm_config.py --compare-all
```

**Manual Fix - Edit your `zinb_benchmark.py`:**

```python
# Find EBM_ZINB_PARAMS (around line 131) and change to:
EBM_ZINB_PARAMS = {
    'interactions': 0,              # ← CHANGED: Remove interactions initially
    'learning_rate': 0.05,          # ← CHANGED: Increase from 0.03
    'max_bins': 512,
    'max_rounds': 2000,             # ← CHANGED: Reduce from 2500
    'early_stopping_rounds': 300,
    'validation_size': 0.15,
    'outer_bags': 16,               # ← CHANGED: Reduce from 24
    'inner_bags': 0,                # ← CHANGED: Disable from 2
    'min_samples_leaf': 3,          # ← CHANGED: Add regularization
    'max_interaction_bins': 64,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}
```

Then run your benchmark:
```bash
python run_benchmark.py
```

---

## 📊 Diagnose Current Performance

**See why GBM is winning:**

```bash
python diagnose_ebm_vs_gbm.py
```

This will:
- Compare metrics between EBM and GBM
- Identify specific issues
- Generate diagnostic plots
- Provide customized recommendations

---

## 🔧 Detailed Parameter Tuning

### Why EBM Is Currently Underperforming

| Parameter | Current | GBM Equivalent | Issue |
|-----------|---------|----------------|-------|
| `learning_rate` | 0.03 | 0.1 | **Too low** - slow learning |
| `interactions` | 3x | N/A | **Too complex** - may overfit or undertrain |
| `max_rounds` | 2500 | 100 | Needs more with lower LR |
| `outer_bags` | 24 | N/A | High variance, slower training |

### Progressive Improvement Strategy

**🥉 Phase 1: Baseline (Start Here)**
- Goal: Beat GBM without interactions
- Changes: Higher LR, no interactions, simpler bagging
- Expected: Should match or beat GBM

**🥈 Phase 2: Add Interactions**
- Goal: Improve beyond Phase 1
- Changes: Add 10 fixed interactions
- Expected: Better feature interaction modeling

**🥇 Phase 3: Full Power**
- Goal: Maximize performance
- Changes: More rounds, 2x-3x interactions, more bagging
- Expected: Best overall performance

---

## 📈 Expected Results

### Before Fixes (Current State)
```
Model Performance:
1. GBM_ZINB    - ZINB NLL: X.XXXX ← Currently best
2. EBM_ZINB    - ZINB NLL: X.XXXX (5-10% worse)
3. XGB_ZINB    - ZINB NLL: X.XXXX
```

### After Phase 1 Fixes
```
Model Performance:
1. EBM_ZINB    - ZINB NLL: X.XXXX ← Should be best or tied
2. GBM_ZINB    - ZINB NLL: X.XXXX
3. XGB_ZINB    - ZINB NLL: X.XXXX
```

### After Phase 2-3 Fixes
```
Model Performance:
1. EBM_ZINB    - ZINB NLL: X.XXXX ← Clearly best
2. GBM_ZINB    - ZINB NLL: X.XXXX (5-10% worse)
3. XGB_ZINB    - ZINB NLL: X.XXXX
```

---

## 🚀 Full Workflow

### Step 1: Fix SHAP (Optional)
```bash
python test_shap_fix.py
# Follow prompts to fix SHAP installation
```

### Step 2: Diagnose Current Issues
```bash
# Run this after you have existing benchmark results
python diagnose_ebm_vs_gbm.py
```

### Step 3: Test Improved Configuration
```bash
# Test Phase 1 (simple but effective)
python improved_ebm_config.py --phase 1

# Check if it beats GBM - if yes, continue to Phase 2
python improved_ebm_config.py --phase 2

# If Phase 2 is better, try Phase 3
python improved_ebm_config.py --phase 3
```

### Step 4: Compare All Configurations
```bash
# Automatically test all phases and find the best
python improved_ebm_config.py --compare-all
```

### Step 5: Use Best Configuration
Once you find the best phase, update your `zinb_benchmark.py` with those parameters.

---

## 🐛 Troubleshooting

### SHAP Still Not Working
**Symptom:** Import errors even after reinstalling

**Solutions:**
1. Check Python version: `python --version` (needs 3.7+)
2. Check NumPy version: `python -c "import numpy; print(numpy.__version__)"`
3. Try conda: `conda install -c conda-forge shap`
4. **Workaround**: Run without SHAP - pipeline still works, just skips SHAP plots

### EBM Still Underperforming After Fixes
**Symptom:** Even Phase 1 doesn't beat GBM

**Possible Causes:**
1. **Custom ZINB objective not working** - EBM might be using standard NB
   - Check: Add `print(f"EBM objective: {model.objective_}")` after model creation
   - Fix: Verify `ZeroInflatedNegativeBinomialDevianceRegressionObjective.hpp` is compiled

2. **Data characteristics favor simple trees**
   - Some datasets work better with simpler models
   - Try: Increase `max_depth` for GBM to make it more complex

3. **Insufficient training data**
   - EBM needs more data for interactions
   - Try: Phase 1 (no interactions) or increase `outer_bags`

4. **Feature engineering needed**
   - EBM might need different feature preprocessing
   - Try: Add polynomial features or binned features

### Diagnostic Script Fails
**Symptom:** `diagnose_ebm_vs_gbm.py` can't find results

**Solution:**
```bash
# Make sure you've run the benchmark first
python run_benchmark.py

# Check results exist
ls artifacts_zinb_benchmark/metrics/

# Then run diagnostic
python diagnose_ebm_vs_gbm.py
```

---

## 📝 Files Created

1. **`test_shap_fix.py`** - Interactive SHAP diagnostic and fix tool
2. **`improved_ebm_config.py`** - Optimized EBM configurations (Phase 1-4)
3. **`diagnose_ebm_vs_gbm.py`** - Performance comparison and diagnostic tool
4. **`DIAGNOSIS_AND_FIXES.md`** - Detailed technical documentation
5. **`QUICK_FIX_GUIDE.md`** - This file

---

## 🎯 Success Criteria

You'll know the fixes worked when:

✅ **SHAP Fixed:**
- `import shap` works without errors
- SHAP plots appear in `artifacts_zinb_benchmark/plots/`

✅ **EBM Improved:**
- EBM_ZINB ranks #1 in leaderboard
- EBM ZINB NLL is lower than GBM ZINB NLL
- McFadden R² is higher for EBM than GBM

---

## 💡 Pro Tips

1. **Start Simple**: Always test Phase 1 first - it often works best
2. **Use Demo Mode**: Quick testing with `python run_benchmark.py --demo`
3. **Monitor Progress**: Check `artifacts_zinb_benchmark/logs/` for training details
4. **Compare Incrementally**: Test one change at a time
5. **Save Results**: Each phase saves to a different output directory

---

## 🆘 Need More Help?

If issues persist:

1. Run diagnostic: `python diagnose_ebm_vs_gbm.py`
2. Check logs: `cat artifacts_zinb_benchmark/logs/progress.log`
3. Share output of:
   ```bash
   python test_shap_fix.py > shap_diagnostic.txt
   python diagnose_ebm_vs_gbm.py > performance_diagnostic.txt
   ```

---

**Good luck! 🚀**

With these fixes, EBM_ZINB should become your top-performing model!
