"""
MAIN OFFLINE TRAINING PIPELINE WITH TIME DECAY
===============================================
Complete end-to-end offline training

Architecture:
- Raw interactions → Embeddings & CF Model (NO time decay)
- Weighted interactions → Ranking Model Training (WITH time decay)

Run: python scripts/offline/main_offline_pipeline.py
"""

import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from recommender.common.data_loading import DataLoader
from recommender.common.data_loader import load_data, compute_statistics
from recommender.offline.embedding_generator import EmbeddingGenerator
from recommender.offline.cf_builder import CFBuilder
from recommender.offline.model_trainer import ModelTrainer
from recommender.offline.artifact_manager import ArtifactManager


def load_config(config_path='scripts/offline/config_offline.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """
    Complete offline training pipeline
    """
    print("=" * 80)
    print("🚀 OFFLINE TRAINING PIPELINE (WITH TIME DECAY)")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    # ═════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA - DUAL LOADING STRATEGY
    # ═════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING (DUAL STRATEGY)")
    print("="*80)
    
    config = load_config()
    
    # ─────────────────────────────────────────────────────────────────────
    # 1A. Load RAW data (NO time decay) - For Embeddings & CF
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n📦 Loading RAW data (for Embeddings & CF)...")
    
    data = load_data(config['data']['dir'])
    
    # Compute statistics
    user_stats, author_stats, following_dict = compute_statistics(data)
    
    print(f"✅ Raw data loaded:")
    print(f"   Users: {len(data['user']):,}")
    print(f"   Posts: {len(data['post']):,}")
    print(f"   Interactions: {len(data['postreaction']):,}")
    print(f"   Friendships: {len(data['friendship']):,}")
    
    # ─────────────────────────────────────────────────────────────────────
    # 1B. Load WEIGHTED data (WITH time decay) - For Ranking Model Training
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n📦 Loading WEIGHTED data (for Ranking Model)...")
    
    data_loader_config = {
        'lookback_days': config['data']['lookback_days'],
        'half_life_days': config.get('time_decay', {}).get('half_life_days', 7.0),
        'min_weight': config.get('time_decay', {}).get('min_weight', 0.01),
        'chunk_size': config.get('data', {}).get('chunk_size', 100000)
    }
    
    data_loader = DataLoader(
        db_connection=None,  # Use CSV for now
        config=data_loader_config,
        data_dir=config['data']['dir']
    )
    
    # Load interactions with time decay applied
    interactions_weighted = data_loader.load_and_prepare_training_data(use_csv=True)
    
    print(f"✅ Weighted data loaded: {len(interactions_weighted):,} interactions")
    print(f"   Weight range: [{interactions_weighted['weight'].min():.4f}, {interactions_weighted['weight'].max():.4f}]")
    print(f"   Mean weight: {interactions_weighted['weight'].mean():.4f}")
    
    # Create temporal splits for weighted data
    train_interactions, val_interactions, test_interactions = data_loader.create_train_test_split(
        interactions_weighted,
        test_days=config['data']['train_test_split']['test_days'],
        val_days=config['data']['train_test_split']['val_days']
    )
    
    print(f"\n📊 Temporal splits:")
    print(f"   Train: {len(train_interactions):,} ({len(train_interactions)/len(interactions_weighted)*100:.1f}%)")
    print(f"   Val: {len(val_interactions):,} ({len(val_interactions)/len(interactions_weighted)*100:.1f}%)")
    print(f"   Test: {len(test_interactions):,} ({len(test_interactions)/len(interactions_weighted)*100:.1f}%)")
    
    # ═════════════════════════════════════════════════════════════════════
    # STEP 2: GENERATE EMBEDDINGS (using RAW data)
    # ═════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("STEP 2: EMBEDDING GENERATION (using RAW interactions)")
    print("="*80)
    
    embedding_gen = EmbeddingGenerator(config)
    
    # Use RAW interactions (no time decay) for embeddings
    post_embeddings = embedding_gen.generate_post_embeddings(data['post'])
    user_embeddings = embedding_gen.generate_user_embeddings(
        data['postreaction'],  # ← RAW data, not weighted
        post_embeddings
    )
    
    embeddings = {
        'post': post_embeddings,
        'user': user_embeddings
    }
    
    print(f"✅ Embeddings generated:")
    print(f"   Posts: {len(post_embeddings):,}")
    print(f"   Users: {len(user_embeddings):,}")
    
    # ═════════════════════════════════════════════════════════════════════
    # STEP 3: BUILD COLLABORATIVE FILTERING (using RAW data)
    # ═════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("STEP 3: COLLABORATIVE FILTERING (using RAW interactions)")
    print("="*80)
    
    cf_builder = CFBuilder(config)
    
    # Use RAW interactions (no time decay) for CF
    cf_model = cf_builder.build_cf_model(data['postreaction'])  # ← RAW data
    
    print(f"✅ CF model built:")
    print(f"   Users: {len(cf_model['user_ids']):,}")
    print(f"   Posts: {len(cf_model['post_ids']):,}")
    print(f"   User similarities computed: {len(cf_model['user_similarities']):,}")
    print(f"   Item similarities computed: {len(cf_model['item_similarities']):,}")
    
    # ═════════════════════════════════════════════════════════════════════
    # STEP 4: TRAIN RANKING MODEL (using WEIGHTED data)
    # ═════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("STEP 4: RANKING MODEL TRAINING (using WEIGHTED interactions)")
    print("="*80)
    
    trainer = ModelTrainer(config)
    
    # Prepare training data using WEIGHTED interactions
    print("\n🔧 Preparing training data with time decay weights...")
    
    train_df = trainer.prepare_training_data(
        train_interactions,  # ← WEIGHTED data with 'weight' column
        data, 
        user_stats, 
        author_stats, 
        following_dict, 
        embeddings
    )
    
    val_df = trainer.prepare_training_data(
        val_interactions,  # ← WEIGHTED data
        data, 
        user_stats, 
        author_stats, 
        following_dict, 
        embeddings
    )
    
    test_df = trainer.prepare_training_data(
        test_interactions,  # ← WEIGHTED data
        data, 
        user_stats, 
        author_stats, 
        following_dict, 
        embeddings
    )
    
    print(f"\n✅ Training data prepared:")
    print(f"   Train samples: {len(train_df):,}")
    print(f"   Val samples: {len(val_df):,}")
    print(f"   Test samples: {len(test_df):,}")
    
    # Check if 'weight' column exists in training data
    if 'weight' in train_df.columns:
        print(f"\n📊 Time decay weights in training data:")
        print(f"   Mean: {train_df['weight'].mean():.4f}")
        print(f"   Median: {train_df['weight'].median():.4f}")
        print(f"   Min: {train_df['weight'].min():.4f}")
        print(f"   Max: {train_df['weight'].max():.4f}")
    
    # Train
    print("\n🔧 Training LightGBM ranking model...")
    ranking_model, ranking_scaler, ranking_feature_cols = trainer.train(train_df, val_df)
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    test_metrics = trainer.evaluate(test_df)
    
    # ═════════════════════════════════════════════════════════════════════
    # STEP 5: SAVE ARTIFACTS
    # ═════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("STEP 5: SAVING ARTIFACTS")
    print("="*80)
    
    artifact_mgr = ArtifactManager(config['output']['models_dir'])
    
    version = artifact_mgr.create_version()
    
    # Prepare metadata
    metadata = {
        'test_metrics': test_metrics,
        'n_train_samples': len(train_df),
        'n_val_samples': len(val_df),
        'n_test_samples': len(test_df),
        'config': config,
        'time_decay': {
            'enabled': True,
            'half_life_days': data_loader_config['half_life_days'],
            'min_weight': data_loader_config['min_weight'],
            'weight_stats': {
                'mean': float(interactions_weighted['weight'].mean()),
                'median': float(interactions_weighted['weight'].median()),
                'min': float(interactions_weighted['weight'].min()),
                'max': float(interactions_weighted['weight'].max())
            }
        },
        'data_stats': {
            'n_users': len(data['user']),
            'n_posts': len(data['post']),
            'n_raw_interactions': len(data['postreaction']),
            'n_weighted_interactions': len(interactions_weighted),
            'n_embeddings_post': len(post_embeddings),
            'n_embeddings_user': len(user_embeddings),
            'n_cf_users': len(cf_model['user_ids']),
            'n_cf_posts': len(cf_model['post_ids'])
        }
    }
    
    artifact_mgr.save_artifacts(
        version=version,
        embeddings=embeddings,
        cf_model=cf_model,
        ranking_model=ranking_model,
        ranking_scaler=ranking_scaler,
        ranking_feature_cols=ranking_feature_cols,
        user_stats=user_stats,
        author_stats=author_stats,
        following_dict=following_dict,
        metadata=metadata
    )
    
    # Cleanup old versions
    artifact_mgr.cleanup_old_versions(
        keep_n=config['output']['keep_last_n_versions']
    )
    
    # ═════════════════════════════════════════════════════════════════════
    # COMPLETE
    # ═════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("✅ OFFLINE TRAINING COMPLETE!")
    print("="*80)
    print(f"Version: {version}")
    print(f"\n📊 Final Metrics:")
    print(f"   Test AUC: {test_metrics['auc']:.4f}")
    print(f"   Test Precision@10: {test_metrics['precision@10']:.4f}")
    print(f"   Test Precision@20: {test_metrics['precision@20']:.4f}")
    print(f"   Test Precision@50: {test_metrics['precision@50']:.4f}")
    print(f"\n💾 Artifacts saved to: models/{version}/")
    print(f"   - Embeddings (RAW data)")
    print(f"   - CF Model (RAW data)")
    print(f"   - Ranking Model (WEIGHTED data)")
    print(f"   - Statistics & Metadata")
    print(f"\n🕐 Completed at: {datetime.now()}")
    print("="*80)


if __name__ == "__main__":
    main()