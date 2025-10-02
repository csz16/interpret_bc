#!/usr/bin/env python3
"""
EBM vs GBM Performance Diagnostic Tool
======================================

This script helps diagnose why GBM_ZINB is outperforming EBM_ZINB
by analyzing their predictions, feature importance, and learning curves.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(artifacts_dir="artifacts_zinb_benchmark"):
    """Load benchmark results."""
    artifacts_dir = Path(artifacts_dir)
    
    results = {}
    metrics_dir = artifacts_dir / "metrics"
    
    for model in ['EBM_ZINB', 'GBM_ZINB', 'XGB_ZINB', 'LGB_ZINB']:
        result_file = metrics_dir / f"{model}_cv_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[model] = json.load(f)
    
    return results


def compare_metrics(results):
    """Compare key metrics between models."""
    print("\n" + "="*80)
    print("METRIC COMPARISON: EBM_ZINB vs GBM_ZINB")
    print("="*80)
    
    metrics_to_compare = [
        'zinb_nll', 'mcfadden_r2', 'zero_accuracy', 
        'mae', 'rmse', 'poisson_deviance', 'efron_r2'
    ]
    
    comparison_data = []
    
    for metric in metrics_to_compare:
        row = {'Metric': metric}
        
        for model in ['EBM_ZINB', 'GBM_ZINB']:
            if model in results and 'summary' in results[model]:
                summary = results[model]['summary']
                if metric in summary and 'mean' in summary[metric]:
                    row[f'{model}_mean'] = summary[metric]['mean']
                    row[f'{model}_std'] = summary[metric].get('std', 0)
        
        if f'EBM_ZINB_mean' in row and f'GBM_ZINB_mean' in row:
            # Calculate difference and winner
            ebm_val = row['EBM_ZINB_mean']
            gbm_val = row['GBM_ZINB_mean']
            
            # For most metrics, lower is better (except R2 metrics where higher is better)
            if 'r2' in metric.lower():
                diff = ebm_val - gbm_val
                winner = 'EBM' if ebm_val > gbm_val else 'GBM'
            else:
                diff = gbm_val - ebm_val  # Positive means GBM is worse (EBM is better)
                winner = 'EBM' if ebm_val < gbm_val else 'GBM'
            
            row['Difference'] = diff
            row['Winner'] = winner
            row['Advantage_%'] = abs(diff / gbm_val * 100) if gbm_val != 0 else 0
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Print table
    print("\nDetailed Comparison:")
    print("-"*80)
    for _, row in df.iterrows():
        metric = row['Metric']
        print(f"\n{metric.upper()}:")
        print(f"  EBM_ZINB: {row.get('EBM_ZINB_mean', 'N/A'):.6f} ± {row.get('EBM_ZINB_std', 0):.6f}")
        print(f"  GBM_ZINB: {row.get('GBM_ZINB_mean', 'N/A'):.6f} ± {row.get('GBM_ZINB_std', 0):.6f}")
        print(f"  Winner: {row.get('Winner', 'N/A')} (advantage: {row.get('Advantage_%', 0):.2f}%)")
    
    # Summary
    ebm_wins = sum(1 for row in comparison_data if row.get('Winner') == 'EBM')
    gbm_wins = sum(1 for row in comparison_data if row.get('Winner') == 'GBM')
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print(f"  EBM wins on {ebm_wins}/{len(comparison_data)} metrics")
    print(f"  GBM wins on {gbm_wins}/{len(comparison_data)} metrics")
    print("="*80)
    
    return df


def analyze_learning_patterns(results):
    """Analyze if models are learning properly."""
    print("\n" + "="*80)
    print("LEARNING PATTERN ANALYSIS")
    print("="*80)
    
    for model in ['EBM_ZINB', 'GBM_ZINB']:
        if model not in results:
            continue
        
        print(f"\n{model}:")
        print("-"*40)
        
        # Check CV consistency
        if 'fold_results' in results[model]:
            fold_results = results[model]['fold_results']
            n_folds = len(fold_results)
            
            # Collect metrics across folds
            zinb_nlls = []
            r2s = []
            
            for fold_id, fold_data in fold_results.items():
                if 'metrics' in fold_data:
                    zinb_nlls.append(fold_data['metrics'].get('zinb_nll', np.nan))
                    r2s.append(fold_data['metrics'].get('mcfadden_r2', np.nan))
            
            if zinb_nlls:
                print(f"  Cross-validation consistency:")
                print(f"    ZINB NLL range: [{min(zinb_nlls):.4f}, {max(zinb_nlls):.4f}]")
                print(f"    ZINB NLL CV: {np.std(zinb_nlls) / np.mean(zinb_nlls) * 100:.2f}%")
                print(f"    McFadden R² range: [{min(r2s):.4f}, {max(r2s):.4f}]")
                
                # Check if variance is too high (overfitting signal)
                if np.std(zinb_nlls) / np.mean(zinb_nlls) > 0.1:
                    print(f"    ⚠️  High variance across folds - possible overfitting or unstable training")
                else:
                    print(f"    ✅ Stable performance across folds")


def identify_issues(results):
    """Identify potential issues with EBM configuration."""
    print("\n" + "="*80)
    print("POTENTIAL ISSUES IDENTIFIED")
    print("="*80)
    
    if 'EBM_ZINB' not in results:
        print("❌ EBM_ZINB results not found!")
        return
    
    ebm_results = results['EBM_ZINB']
    gbm_results = results.get('GBM_ZINB', {})
    
    issues = []
    
    # Check if EBM is significantly worse
    if 'summary' in ebm_results and 'summary' in gbm_results:
        ebm_nll = ebm_results['summary'].get('zinb_nll', {}).get('mean', 0)
        gbm_nll = gbm_results['summary'].get('zinb_nll', {}).get('mean', 0)
        
        if ebm_nll > gbm_nll * 1.05:  # EBM is >5% worse
            issues.append({
                'issue': 'EBM has significantly higher NLL than GBM',
                'severity': 'HIGH',
                'possible_causes': [
                    'Learning rate too low (0.03 vs GBM 0.1)',
                    'Not enough training rounds or early stopping too aggressive',
                    'Interactions adding noise instead of signal',
                    'ZINB objective not being used (falling back to NB)',
                ],
                'recommended_fixes': [
                    'Increase learning_rate to 0.05-0.07',
                    'Increase max_rounds to 3000-4000',
                    'Try without interactions first (interactions=0)',
                    'Verify custom ZINB objective is compiled and loaded',
                ]
            })
    
    # Check for overfitting
    if 'fold_results' in ebm_results:
        fold_results = ebm_results['fold_results']
        zinb_nlls = [fold_data['metrics']['zinb_nll'] 
                     for fold_data in fold_results.values() 
                     if 'metrics' in fold_data and 'zinb_nll' in fold_data['metrics']]
        
        if zinb_nlls and np.std(zinb_nlls) / np.mean(zinb_nlls) > 0.15:
            issues.append({
                'issue': 'High variance across CV folds',
                'severity': 'MEDIUM',
                'possible_causes': [
                    'Overfitting due to too many interactions',
                    'Insufficient bagging',
                    'Small sample size in folds',
                ],
                'recommended_fixes': [
                    'Reduce interactions (from 3x to 2x or 0)',
                    'Increase outer_bags (from 24 to 32-40)',
                    'Increase min_samples_leaf (from 2 to 5)',
                    'Add L2 regularization if available',
                ]
            })
    
    # Print issues
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. {issue['issue']} [Severity: {issue['severity']}]")
            print("   Possible Causes:")
            for cause in issue['possible_causes']:
                print(f"     - {cause}")
            print("   Recommended Fixes:")
            for fix in issue['recommended_fixes']:
                print(f"     ✓ {fix}")
    else:
        print("\n✅ No major issues detected!")


def generate_recommendations(results):
    """Generate specific parameter recommendations."""
    print("\n" + "="*80)
    print("RECOMMENDED PARAMETER CHANGES")
    print("="*80)
    
    print("\n📋 Current EBM Parameters (from config):")
    print("""
    'learning_rate': 0.03,
    'max_rounds': 2500,
    'early_stopping_rounds': 300,
    'interactions': '3x',
    'outer_bags': 24,
    'inner_bags': 2,
    'max_bins': 512,
    """)
    
    print("\n🔧 Recommended Changes (Progressive Approach):")
    print("\n--- PHASE 1: Simplify and Strengthen Learning ---")
    print("""
    'learning_rate': 0.05,          # Increase from 0.03
    'max_rounds': 2000,             # Slightly reduce
    'early_stopping_rounds': 300,   # Keep same
    'interactions': 0,              # REMOVE interactions temporarily
    'outer_bags': 16,               # Reduce for faster baseline
    'inner_bags': 0,                # Disable for simplicity
    'max_bins': 512,                # Keep same
    'min_samples_leaf': 3,          # Add regularization
    """)
    
    print("\n--- PHASE 2: Add Limited Interactions ---")
    print("(Use this if Phase 1 beats GBM)")
    print("""
    'learning_rate': 0.05,
    'max_rounds': 2500,
    'early_stopping_rounds': 350,
    'interactions': 10,             # Fixed number
    'outer_bags': 20,
    'inner_bags': 0,
    'max_bins': 512,
    'min_samples_leaf': 3,
    'max_interaction_bins': 96,     # Increase from 64
    """)
    
    print("\n--- PHASE 3: Scale Up ---")
    print("(Use this if Phase 2 continues improving)")
    print("""
    'learning_rate': 0.04,
    'max_rounds': 3000,
    'early_stopping_rounds': 400,
    'interactions': '2x',           # Moderate multiplier
    'outer_bags': 24,
    'inner_bags': 2,
    'max_bins': 512,
    'min_samples_leaf': 2,
    'max_interaction_bins': 128,
    """)


def create_diagnostic_plot(results, output_path="ebm_vs_gbm_diagnosis.png"):
    """Create visual diagnostic plot."""
    print("\n" + "="*80)
    print("CREATING DIAGNOSTIC PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EBM vs GBM Performance Diagnosis', fontsize=16, fontweight='bold')
    
    models = ['EBM_ZINB', 'GBM_ZINB']
    colors = {'EBM_ZINB': '#FF6B6B', 'GBM_ZINB': '#4ECDC4'}
    
    # 1. ZINB NLL across folds
    ax = axes[0, 0]
    for model in models:
        if model in results and 'fold_results' in results[model]:
            fold_results = results[model]['fold_results']
            nlls = [fold_data['metrics'].get('zinb_nll', np.nan) 
                   for fold_data in fold_results.values() 
                   if 'metrics' in fold_data]
            if nlls:
                ax.plot(range(1, len(nlls)+1), nlls, marker='o', 
                       label=model, color=colors[model], linewidth=2, markersize=8)
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('ZINB NLL (lower is better)', fontsize=12)
    ax.set_title('ZINB Negative Log-Likelihood by Fold', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. McFadden R² across folds
    ax = axes[0, 1]
    for model in models:
        if model in results and 'fold_results' in results[model]:
            fold_results = results[model]['fold_results']
            r2s = [fold_data['metrics'].get('mcfadden_r2', np.nan) 
                  for fold_data in fold_results.values() 
                  if 'metrics' in fold_data]
            if r2s:
                ax.plot(range(1, len(r2s)+1), r2s, marker='s', 
                       label=model, color=colors[model], linewidth=2, markersize=8)
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('McFadden R² (higher is better)', fontsize=12)
    ax.set_title('McFadden R² by Fold', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Metric comparison bar chart
    ax = axes[1, 0]
    metrics = ['zinb_nll', 'mae', 'rmse']
    metric_names = ['ZINB NLL\n(lower better)', 'MAE\n(lower better)', 'RMSE\n(lower better)']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ebm_vals = []
    gbm_vals = []
    
    for metric in metrics:
        ebm_val = results.get('EBM_ZINB', {}).get('summary', {}).get(metric, {}).get('mean', 0)
        gbm_val = results.get('GBM_ZINB', {}).get('summary', {}).get(metric, {}).get('mean', 0)
        ebm_vals.append(ebm_val)
        gbm_vals.append(gbm_val)
    
    ax.bar(x - width/2, ebm_vals, width, label='EBM_ZINB', color=colors['EBM_ZINB'])
    ax.bar(x + width/2, gbm_vals, width, label='GBM_ZINB', color=colors['GBM_ZINB'])
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Key Metrics Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Performance summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = []
    for model in models:
        if model in results and 'summary' in results[model]:
            summary = results[model]['summary']
            summary_data.append([
                model,
                f"{summary.get('zinb_nll', {}).get('mean', 0):.4f}",
                f"{summary.get('mcfadden_r2', {}).get('mean', 0):.4f}",
                f"{summary.get('zero_accuracy', {}).get('mean', 0):.4f}",
            ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Model', 'ZINB NLL↓', 'McFadden R²↑', 'Zero Acc↑'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color the header
    for i in range(4):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Diagnostic plot saved to: {output_path}")
    plt.close()


def main():
    """Main diagnostic routine."""
    print("🔍 EBM vs GBM Performance Diagnostic Tool")
    
    # Load results
    results = load_results()
    
    if not results:
        print("❌ No results found! Please run the benchmark first.")
        return 1
    
    print(f"\n✅ Loaded results for: {', '.join(results.keys())}")
    
    # Run diagnostics
    comparison_df = compare_metrics(results)
    analyze_learning_patterns(results)
    identify_issues(results)
    generate_recommendations(results)
    create_diagnostic_plot(results)
    
    # Save comparison to CSV
    comparison_df.to_csv('ebm_vs_gbm_comparison.csv', index=False)
    print(f"\n✅ Detailed comparison saved to: ebm_vs_gbm_comparison.csv")
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE!")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review the recommendations above")
    print("2. Use improved_ebm_config.py to test different configurations")
    print("3. Run: python improved_ebm_config.py --phase 1")
    print("4. Compare results and iterate")
    
    return 0


if __name__ == "__main__":
    exit(main())
