#!/usr/bin/env python3
"""
SHAP Diagnostic and Fix Script
==============================

This script helps diagnose why SHAP is not working and provides fixes.
Run this before the benchmark to ensure SHAP is properly installed.
"""

import sys
import subprocess
import importlib


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    package_name = package_name or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True, version
    except Exception as e:
        print(f"❌ {package_name}: {type(e).__name__}: {e}")
        return False, None


def test_shap_basic():
    """Test basic SHAP functionality."""
    print("\n" + "="*60)
    print("Testing SHAP Basic Functionality")
    print("="*60)
    
    try:
        import shap
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        
        # Create simple dataset
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test TreeExplainer
        print("Testing TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:10])
        print(f"✅ TreeExplainer works! SHAP values shape: {shap_values.shape}")
        
        # Test summary plot (without display)
        print("Testing summary_plot...")
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode
        shap.summary_plot(shap_values, X[:10], show=False)
        plt.close()
        print("✅ summary_plot works!")
        
        return True
        
    except Exception as e:
        print(f"❌ SHAP test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def fix_shap_installation():
    """Attempt to fix SHAP installation."""
    print("\n" + "="*60)
    print("Attempting to Fix SHAP Installation")
    print("="*60)
    
    fixes = [
        ("Upgrade pip, setuptools, wheel", 
         [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]),
        
        ("Uninstall SHAP", 
         [sys.executable, "-m", "pip", "uninstall", "shap", "-y"]),
        
        ("Install SHAP with no cache", 
         [sys.executable, "-m", "pip", "install", "shap", "--no-cache-dir"]),
    ]
    
    for step_name, command in fixes:
        print(f"\n🔧 {step_name}...")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ {step_name} successful")
            else:
                print(f"⚠️  {step_name} completed with warnings")
                if result.stderr:
                    print(f"   stderr: {result.stderr[:200]}")
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            return False
    
    return True


def alternative_shap_install():
    """Try alternative installation methods."""
    print("\n" + "="*60)
    print("Trying Alternative Installation Methods")
    print("="*60)
    
    methods = [
        ("Install from conda-forge (if conda available)", 
         ["conda", "install", "-c", "conda-forge", "shap", "-y"]),
        
        ("Install specific version compatible with NumPy", 
         [sys.executable, "-m", "pip", "install", "shap==0.41.0", "numpy<1.24"]),
        
        ("Install from source", 
         [sys.executable, "-m", "pip", "install", "git+https://github.com/slundberg/shap.git"]),
    ]
    
    for method_name, command in methods:
        print(f"\n🔧 {method_name}...")
        response = input(f"   Try this method? (y/n): ").strip().lower()
        
        if response == 'y':
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    print(f"✅ {method_name} successful")
                    # Test if it works
                    if test_shap_basic():
                        return True
                else:
                    print(f"❌ {method_name} failed")
                    if result.stderr:
                        print(f"   stderr: {result.stderr[:200]}")
            except FileNotFoundError:
                print(f"⚠️  Command not found: {command[0]}")
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
        else:
            print(f"   Skipped.")
    
    return False


def create_shap_free_visualization():
    """Create a modified visualization.py that doesn't require SHAP."""
    print("\n" + "="*60)
    print("Creating SHAP-Free Visualization Module")
    print("="*60)
    
    code = '''# Modified visualization.py - SHAP-free version
# This is a backup in case SHAP cannot be installed

# In the ZINBVisualizer class, replace SHAP methods with stubs:

def plot_shap_summary(self, model, X, features, model_name):
    """Placeholder for SHAP summary plot."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=self.figsize)
    ax.text(0.5, 0.5, 'SHAP not available\\nInstall SHAP to see this visualization',
            ha='center', va='center', fontsize=14)
    ax.axis('off')
    plt.savefig(self.output_dir / f'{model_name}_shap_summary.png', 
                dpi=self.dpi, bbox_inches='tight')
    plt.close()
    print(f"⚠️  SHAP summary plot skipped (SHAP not available)")

def plot_shap_beeswarm(self, model, X, features, model_name):
    """Placeholder for SHAP beeswarm plot."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=self.figsize)
    ax.text(0.5, 0.5, 'SHAP not available\\nInstall SHAP to see this visualization',
            ha='center', va='center', fontsize=14)
    ax.axis('off')
    plt.savefig(self.output_dir / f'{model_name}_shap_beeswarm.png', 
                dpi=self.dpi, bbox_inches='tight')
    plt.close()
    print(f"⚠️  SHAP beeswarm plot skipped (SHAP not available)")

def plot_shap_bar(self, model, X, features, model_name):
    """Placeholder for SHAP bar plot."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=self.figsize)
    ax.text(0.5, 0.5, 'SHAP not available\\nInstall SHAP to see this visualization',
            ha='center', va='center', fontsize=14)
    ax.axis('off')
    plt.savefig(self.output_dir / f'{model_name}_shap_bar.png', 
                dpi=self.dpi, bbox_inches='tight')
    plt.close()
    print(f"⚠️  SHAP bar plot skipped (SHAP not available)")
'''
    
    print("📝 Add these methods to your visualization.py if SHAP continues to fail:")
    print(code)
    
    with open('shap_free_visualization_patch.py', 'w') as f:
        f.write(code)
    
    print("\n✅ Saved to shap_free_visualization_patch.py")


def main():
    """Main diagnostic routine."""
    print("🔍 SHAP Diagnostic and Fix Script")
    print("="*60)
    
    # Check dependencies
    print("\nChecking Dependencies:")
    print("-"*60)
    deps = {
        'numpy': None,
        'scipy': None,
        'scikit-learn': 'sklearn',
        'matplotlib': None,
        'pandas': None,
        'shap': None,
    }
    
    all_ok = True
    for pkg, import_name in deps.items():
        ok, version = check_import(import_name or pkg, pkg)
        if not ok and pkg != 'shap':
            all_ok = False
    
    if not all_ok:
        print("\n❌ Some required dependencies are missing!")
        print("   Install with: pip install numpy scipy scikit-learn matplotlib pandas")
        return 1
    
    # Check SHAP specifically
    shap_ok, shap_version = check_import('shap')
    
    if shap_ok:
        print("\n✅ SHAP is installed!")
        # Test functionality
        if test_shap_basic():
            print("\n🎉 SHAP is working correctly!")
            return 0
        else:
            print("\n⚠️  SHAP is installed but not working correctly")
    else:
        print("\n❌ SHAP is not installed or has import errors")
    
    # Offer fixes
    print("\nWhat would you like to do?")
    print("1. Try to fix SHAP installation (recommended)")
    print("2. Try alternative installation methods")
    print("3. Create SHAP-free visualization module (workaround)")
    print("4. Exit (run benchmark without SHAP)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        if fix_shap_installation():
            print("\n🔄 Testing SHAP after fix...")
            if test_shap_basic():
                print("\n🎉 SHAP is now working!")
                return 0
            else:
                print("\n❌ SHAP still not working. Try option 2 or 3.")
                return 1
    
    elif choice == '2':
        if alternative_shap_install():
            print("\n🎉 SHAP is now working!")
            return 0
        else:
            print("\n❌ Alternative installations failed.")
            return 1
    
    elif choice == '3':
        create_shap_free_visualization()
        print("\n✅ You can now run the benchmark without SHAP")
        return 0
    
    else:
        print("\n📝 You can run the benchmark without SHAP.")
        print("   The pipeline will skip SHAP visualizations but everything else will work.")
        return 0


if __name__ == "__main__":
    exit(main())
