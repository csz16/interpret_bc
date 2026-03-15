#!/usr/bin/env python3
"""
Improved EBM Configuration for Better Performance
================================================

This module provides optimized configurations for EBM_ZINB to outperform GBM_ZINB.
Use this by importing and replacing the default Config in your pipeline.

Usage:
    from improved_ebm_config import ImprovedConfig
    config = ImprovedConfig()
    pipeline = ZINBBenchmarkPipeline(config)
"""

from pathlib import Path
from zinb_benchmark import Config


class ImprovedConfig(Config):
    """Improved configuration with optimized EBM parameters."""
    
    # PHASE 1: Simple but Effective (Start Here)
    EBM_ZINB_PARAMS_PHASE1 = {
        'interactions': 0,              # No interactions - test main effects only
        'learning_rate': 0.05,          # Higher than default (0.03)
        'max_bins': 512,
        'max_rounds': 2000,
        'early_stopping_rounds': 300,
        'validation_size': 0.15,
        'outer_bags': 16,               # Reduced from 24
        'inner_bags': 0,                # Disabled for simplicity
        'min_samples_leaf': 3,
        'max_interaction_bins': 64,
        'random_state': Config.RANDOM_SEED,
        'n_jobs': -1
    }
    
    # PHASE 2: Add Fixed Interactions
    EBM_ZINB_PARAMS_PHASE2 = {
        'interactions': 10,             # Fixed number of top interactions
        'learning_rate': 0.05,
        'max_bins': 512,
        'max_rounds': 2500,
        'early_stopping_rounds': 350,
        'validation_size': 0.15,
        'outer_bags': 20,
        'inner_bags': 0,
        'min_samples_leaf': 3,
        'max_interaction_bins': 96,     # Increased
        'random_state': Config.RANDOM_SEED,
        'n_jobs': -1
    }
    
    # PHASE 3: Moderate Interactions with Bagging
    EBM_ZINB_PARAMS_PHASE3 = {
        'interactions': '2x',           # 2x features
        'learning_rate': 0.04,          # Slightly lower
        'max_bins': 512,
        'max_rounds': 3000,             # More rounds
        'early_stopping_rounds': 400,
        'validation_size': 0.15,
        'outer_bags': 24,
        'inner_bags': 2,                # Re-enable inner bagging
        'min_samples_leaf': 2,
        'max_interaction_bins': 128,
        'random_state': Config.RANDOM_SEED,
        'n_jobs': -1
    }
    
    # PHASE 4: Full Power (Use if Phase 3 works well)
    EBM_ZINB_PARAMS_PHASE4 = {
        'interactions': '3x',           # 3x features
        'learning_rate': 0.035,         # Balanced rate
        'max_bins': 768,                # Increased bins
        'max_rounds': 4000,
        'early_stopping_rounds': 500,
        'validation_size': 0.15,
        'outer_bags': 32,               # More bagging
        'inner_bags': 3,
        'min_samples_leaf': 2,
        'max_interaction_bins': 128,
        'random_state': Config.RANDOM_SEED,
        'n_jobs': -1
    }
    
    # AGGRESSIVE: Maximum Performance (Slow but Powerful)
    EBM_ZINB_PARAMS_AGGRESSIVE = {
        'interactions': '4x',           # 4x features
        'learning_rate': 0.02,          # Slower learning
        'max_bins': 1024,               # Maximum bins
        'max_rounds': 5000,
        'early_stopping_rounds': 600,
        'validation_size': 0.15,
        'outer_bags': 40,
        'inner_bags': 4,
        'min_samples_leaf': 1,
        'max_interaction_bins': 256,
        'random_state': Config.RANDOM_SEED,
        'n_jobs': -1
    }
    
    # Default to Phase 1 for testing
    EBM_ZINB_PARAMS = EBM_ZINB_PARAMS_PHASE1
    
    @classmethod
    def use_phase(cls, phase: int):
        """
        Switch to a specific phase configuration.
        
        Parameters:
        -----------
        phase : int
            Phase number (1-4) or 0 for aggressive
        """
        if phase == 1:
            cls.EBM_ZINB_PARAMS = cls.EBM_ZINB_PARAMS_PHASE1
        elif phase == 2:
            cls.EBM_ZINB_PARAMS = cls.EBM_ZINB_PARAMS_PHASE2
        elif phase == 3:
            cls.EBM_ZINB_PARAMS = cls.EBM_ZINB_PARAMS_PHASE3
        elif phase == 4:
            cls.EBM_ZINB_PARAMS = cls.EBM_ZINB_PARAMS_PHASE4
        elif phase == 0:
            cls.EBM_ZINB_PARAMS = cls.EBM_ZINB_PARAMS_AGGRESSIVE
        else:
            raise ValueError(f"Invalid phase {phase}. Choose 0-4.")
        
        print(f"🔧 Using EBM Phase {phase} Configuration")
        print(f"   Learning Rate: {cls.EBM_ZINB_PARAMS['learning_rate']}")
        print(f"   Interactions: {cls.EBM_ZINB_PARAMS['interactions']}")
        print(f"   Max Rounds: {cls.EBM_ZINB_PARAMS['max_rounds']}")
        print(f"   Outer Bags: {cls.EBM_ZINB_PARAMS['outer_bags']}")


class GridSearchConfig(Config):
    """Configuration for grid search over EBM parameters."""
    
    GRID_SEARCH_PARAMS = {
        'learning_rate': [0.03, 0.05, 0.07, 0.1],
        'interactions': [0, 10, '2x', '3x'],
        'outer_bags': [8, 16, 24, 32],
        'max_rounds': [2000, 3000, 4000],
        'max_bins': [256, 512, 768],
    }
    
    # Reduce CV folds for faster grid search
    CV_FOLDS = 3


# Quick access functions
def get_config(phase: int = 1):
    """
    Get improved configuration for a specific phase.
    
    Parameters:
    -----------
    phase : int, default=1
        Phase number (1-4) or 0 for aggressive
        
    Returns:
    --------
    ImprovedConfig
        Configuration object ready to use
    """
    config = ImprovedConfig()
    config.use_phase(phase)
    return config


def compare_all_phases():
    """
    Run benchmark with all phase configurations to find the best.
    
    Returns:
    --------
    dict
        Results for each phase
    """
    from benchmark_pipeline import ZINBBenchmarkPipeline
    
    results = {}
    
    for phase in range(1, 5):
        print(f"\n{'='*60}")
        print(f"Testing Phase {phase}")
        print(f"{'='*60}")
        
        config = get_config(phase)
        config.OUTPUT_DIR = Path(f"artifacts_zinb_benchmark_phase{phase}")
        
        pipeline = ZINBBenchmarkPipeline(config)
        success = pipeline.run_benchmark()
        
        if success and 'EBM_ZINB' in pipeline.results:
            ebm_summary = pipeline.results['EBM_ZINB']['summary']
            results[f'Phase_{phase}'] = {
                'zinb_nll': ebm_summary.get('zinb_nll', {}).get('mean'),
                'mcfadden_r2': ebm_summary.get('mcfadden_r2', {}).get('mean'),
                'config': config.EBM_ZINB_PARAMS
            }
    
    # Find best phase
    best_phase = min(results.items(), 
                    key=lambda x: x[1]['zinb_nll'] if x[1]['zinb_nll'] else float('inf'))
    
    print(f"\n{'='*60}")
    print(f"🏆 Best Configuration: {best_phase[0]}")
    print(f"{'='*60}")
    print(f"ZINB NLL: {best_phase[1]['zinb_nll']:.4f}")
    print(f"McFadden R²: {best_phase[1]['mcfadden_r2']:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run benchmark with improved EBM config')
    parser.add_argument('--phase', type=int, default=1, choices=[0, 1, 2, 3, 4],
                       help='Configuration phase (1-4, or 0 for aggressive)')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all phases')
    
    args = parser.parse_args()
    
    if args.compare_all:
        compare_all_phases()
    else:
        from benchmark_pipeline import ZINBBenchmarkPipeline
        
        config = get_config(args.phase)
        pipeline = ZINBBenchmarkPipeline(config)
        pipeline.run_benchmark()
