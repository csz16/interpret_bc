# ZINB Benchmark Issues: Diagnosis and Solutions

## Issue 1: SHAP Not Working

### Diagnosis

From `zinb_benchmark.py` (lines 86-91), SHAP import has a broad exception handler:

```python
try:
    import shap
    HAS_SHAP = True
except (ImportError, AttributeError, OSError) as e:
    HAS_SHAP = False
    print(f"Warning: SHAP not available for enhanced visualizations: {e}")
```

**Common Causes:**

1. **C++ Compilation Issues**: SHAP requires compiled extensions
2. **NumPy Version Incompatibility**: SHAP may be incompatible with your NumPy version
3. **Missing System Dependencies**: Missing C++ compiler or build tools
4. **Silent Failures**: The broad exception catching might hide the real error

### Solutions

#### Solution 1: Reinstall with proper dependencies
```bash
# Uninstall completely
pip uninstall shap -y

# Install build dependencies
pip install --upgrade pip setuptools wheel

# Reinstall SHAP
pip install shap --no-cache-dir --verbose

# Or use conda if available
conda install -c conda-forge shap
```

#### Solution 2: Check NumPy compatibility
```bash
# Check versions
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import shap; print(f'SHAP: {shap.__version__}')"

# If NumPy is too new, downgrade
pip install "numpy<1.24" shap
```

#### Solution 3: Test SHAP directly
```python
# Create a test script: test_shap.py
import numpy as np
try:
    import shap
    print(f"✅ SHAP imported successfully: {shap.__version__}")
    
    # Test basic functionality
    from sklearn.ensemble import RandomForestRegressor
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])
    print(f"✅ SHAP working correctly, shape: {shap_values.shape}")
    
except Exception as e:
    print(f"❌ SHAP test failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

#### Solution 4: Alternative - Remove SHAP dependency (temporary workaround)

If SHAP continues to fail, the pipeline should work without it. The code already has `HAS_SHAP` flag checks in `benchmark_pipeline.py` (lines 341-354). However, you might need to verify the visualization module handles this properly.

---

## Issue 2: GBM_ZINB Outperforming EBM_ZINB

### Why GBM_ZINB Might Be Better

Based on the configuration in `zinb_benchmark.py`:

**GBM_ZINB Parameters:**
```python
GBM_ZINB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
}
```

**EBM_ZINB Parameters:**
```python
EBM_ZINB_PARAMS = {
    'interactions': '3x',
    'learning_rate': 0.03,        # Lower than GBM
    'max_bins': 512,
    'max_rounds': 2500,
    'early_stopping_rounds': 300,
    'outer_bags': 24,
    'inner_bags': 2,
}
```

### Potential Issues with EBM_ZINB

1. **Learning Rate Too Low**: 0.03 vs GBM's 0.1
2. **Custom Objective Not Working**: EBM may be falling back to NB instead of ZINB
3. **Interaction Overhead**: 3x interactions might be overfitting or undertrained
4. **Bagging Strategy**: Bagging might not be optimal for this dataset
5. **Early Stopping**: Stopping too early (300 rounds out of 2500)

### Solutions to Improve EBM_ZINB Performance

#### Solution 1: Tune Learning Rate
```python
# In zinb_benchmark.py, update EBM_ZINB_PARAMS:
EBM_ZINB_PARAMS = {
    'interactions': '3x',
    'learning_rate': 0.05,  # INCREASED from 0.03
    'max_bins': 512,
    'max_rounds': 3000,     # INCREASED from 2500
    'early_stopping_rounds': 400,  # INCREASED from 300
    # ... rest of params
}
```

#### Solution 2: Verify Custom ZINB Objective
The EBM model needs the custom ZINB objective. Check if:
- `ZeroInflatedNegativeBinomialDevianceRegressionObjective.hpp` is properly compiled
- EBM is actually using it vs falling back to standard NB

Add diagnostic logging:
```python
# In model_wrappers.py, add after EBM initialization:
print(f"EBM objective: {model.objective_}")
print(f"EBM using ZINB: {'zinb' in str(model.objective_).lower()}")
```

#### Solution 3: Reduce Interactions (Test)
Too many interactions might dilute learning:
```python
# Try simpler interaction settings first
'interactions': 10,  # Fixed number instead of '3x'
# Or
'interactions': 0,   # No interactions, test main effects only
```

#### Solution 4: Optimize Bagging
```python
EBM_ZINB_PARAMS = {
    'outer_bags': 16,     # REDUCED from 24
    'inner_bags': 0,      # DISABLED - try without inner bagging
    # Or increase if training is unstable:
    'outer_bags': 32,     # INCREASED
    'inner_bags': 4,      # INCREASED
}
```

#### Solution 5: Increase Max Interaction Bins
```python
'max_interaction_bins': 128,  # INCREASED from 64
```

#### Solution 6: Add Validation Monitoring
```python
# In model_wrappers.py EBMZINBModel.fit():
# Add callbacks to monitor validation performance
# This helps diagnose if early stopping is working correctly
```

### Recommended Tuning Strategy

**Phase 1: Baseline (No Interactions)**
```python
EBM_ZINB_PARAMS = {
    'interactions': 0,              # Start simple
    'learning_rate': 0.05,          # Match GBM magnitude
    'max_bins': 512,
    'max_rounds': 2000,
    'early_stopping_rounds': 300,
    'outer_bags': 16,
    'inner_bags': 0,
    'validation_size': 0.15,
    'min_samples_leaf': 5,          # Regularization
}
```

**Phase 2: Add Interactions**
If Phase 1 beats GBM, add interactions:
```python
'interactions': 10,  # Start with fixed count
```

**Phase 3: Increase Complexity**
If still improving:
```python
'interactions': '2x',  # Then try multiplier
'learning_rate': 0.04,
'outer_bags': 24,
```

**Phase 4: Fine-tuning**
```python
'interactions': '3x',
'learning_rate': 0.03,
'max_rounds': 3000,
'outer_bags': 32,
'inner_bags': 2,
```

### Quick Test Configuration

Add this to your code to quickly test improved settings:

```python
# In zinb_benchmark.py after Config class definition:

class ImprovedEBMConfig(Config):
    """Improved EBM configuration for testing."""
    
    EBM_ZINB_PARAMS = {
        # Phase 1: Simple but effective
        'interactions': 0,              # No interactions initially
        'learning_rate': 0.05,          # Higher than before
        'max_bins': 512,
        'max_rounds': 2000,
        'early_stopping_rounds': 300,
        'validation_size': 0.15,
        'outer_bags': 16,
        'inner_bags': 0,
        'min_samples_leaf': 3,
        'max_interaction_bins': 64,
        'random_state': Config.RANDOM_SEED,
        'n_jobs': -1
    }
```

Then run:
```bash
python run_benchmark.py --config improved
```

### Diagnostic Commands

Run these to understand the performance gap:

```python
# After training, add to benchmark_pipeline.py:

def diagnose_model_performance(self):
    """Diagnose why models perform differently."""
    print("\n🔍 Model Performance Diagnosis")
    print("=" * 60)
    
    for model_name, result in self.results.items():
        summary = result['summary']
        print(f"\n{model_name}:")
        print(f"  ZINB NLL: {summary.get('zinb_nll', {}).get('mean', 'N/A'):.4f}")
        print(f"  McFadden R²: {summary.get('mcfadden_r2', {}).get('mean', 'N/A'):.4f}")
        print(f"  Zero Accuracy: {summary.get('zero_accuracy', {}).get('mean', 'N/A'):.4f}")
        
        if 'EBM' in model_name:
            # Check if using ZINB objective
            params = result['params']
            print(f"  Learning rate: {params.get('learning_rate')}")
            print(f"  Interactions: {params.get('interactions')}")
            print(f"  Outer bags: {params.get('outer_bags')}")
```

---

## Action Plan

### Step 1: Fix SHAP (Optional but Recommended)
1. Run the SHAP test script above
2. Try reinstall with verbose output
3. If fails, continue without SHAP (pipeline still works)

### Step 2: Diagnose EBM Performance
1. Add diagnostic logging to see if ZINB objective is being used
2. Check validation curves to see if early stopping is appropriate
3. Compare feature importance between EBM and GBM

### Step 3: Tune EBM Parameters
1. Start with Phase 1 (no interactions, higher LR)
2. Run benchmark and compare
3. Gradually add complexity if improving
4. Use grid search if needed:

```python
learning_rates = [0.03, 0.05, 0.07, 0.1]
interactions = [0, 10, '2x', '3x']
outer_bags = [8, 16, 24, 32]

# Grid search over combinations
```

### Step 4: Verify Model Implementation
Ensure EBMZINBModel is properly using ZINB objective, not falling back to NB.

---

## Expected Outcomes

After implementing these fixes:

1. **SHAP**: Should import without errors and generate visualizations
2. **EBM Performance**: Should match or exceed GBM performance due to:
   - Better interpretability
   - Interaction modeling
   - Proper ZINB likelihood
   - Sophisticated bagging

If EBM still underperforms, it suggests:
- Data characteristics favor simpler trees (GBM)
- ZINB objective not being used correctly
- Interactions causing overfitting
- Need for different feature engineering

Let me know the results and I can help further tune!
