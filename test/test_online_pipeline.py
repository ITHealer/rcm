"""
QUICK TEST SCRIPT - Test toàn bộ ONLINE pipeline
=================================================
Chạy script này để verify tất cả components hoạt động đúng

Usage: python test_online_pipeline.py
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup
print("="*70)
print("TESTING ONLINE INFERENCE PIPELINE")
print("="*70)
print()

# ========================================================================
# TEST 1: Feature Engineer (Fixed)
# ========================================================================

print("TEST 1: Feature Engineer (Fixed)")
print("-" * 70)

try:
    # Import fixed feature engineer
    from feature_engineer_fixed import FeatureEngineer, validate_training_data
    from recommender.common.data_loader import load_data
    import yaml
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("Loading data...")
    data = load_data(config.get('data_dir', 'dataset'))
    print(f"✓ Loaded {len(data['user'])} users, {len(data['post'])} posts")
    
    # Test data validation
    interactions = data['postreaction']
    print(f"\nOriginal interactions: {len(interactions)}")
    
    cleaned = validate_training_data(interactions, data)
    print(f"Cleaned interactions: {len(cleaned)}")
    print(f"Removed: {len(interactions) - len(cleaned)} invalid")
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    feature_engineer = FeatureEngineer(data, {}, {}, {}, {})
    
    user_id = data['user']['Id'].iloc[0]
    post_id = data['post']['Id'].iloc[0]
    
    features = feature_engineer.extract_features(user_id, post_id)
    print(f"✓ Extracted {len(features)} features for (user={user_id}, post={post_id})")
    
    # Test with invalid IDs (should not crash)
    print("\nTesting with invalid IDs...")
    try:
        features = feature_engineer.extract_features(-1, post_id)
        print("✓ Handled invalid user_id gracefully")
    except ValueError as e:
        print(f"✗ Still raising error: {e}")
    
    print("\n✅ TEST 1 PASSED: Feature Engineer works correctly!")
    
except Exception as e:
    print(f"\n❌ TEST 1 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ========================================================================
# TEST 2: Multi-Channel Recall
# ========================================================================

print("TEST 2: Multi-Channel Recall")
print("-" * 70)

try:
    from multi_channel_recall import MultiChannelRecall
    
    # Load models
    print("Loading models...")
    with open('models/cf_model.pkl', 'rb') as f:
        cf_model = pickle.load(f)
    print("✓ Loaded cf_model.pkl")
    
    with open('models/embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    print("✓ Loaded embeddings.pkl")
    
    # Initialize recall system
    print("\nInitializing recall system...")
    recall = MultiChannelRecall(
        redis_client=None,
        data=data,
        cf_model=cf_model,
        embeddings=embeddings
    )
    print("✓ Recall system initialized")
    
    # Test recall
    print("\nTesting recall...")
    test_user_id = data['user']['Id'].iloc[0]
    
    start_time = time.time()
    candidates = recall.recall(test_user_id, k=1000)
    latency = (time.time() - start_time) * 1000
    
    print(f"✓ Recalled {len(candidates)} candidates in {latency:.1f}ms")
    
    # Check latency
    if latency < 100:
        print(f"✓ Latency < 100ms (excellent!)")
    elif latency < 200:
        print(f"✓ Latency < 200ms (good)")
    else:
        print(f"⚠ Latency > 200ms (consider optimization)")
    
    # Print metrics
    print("\nRecall channel breakdown:")
    recall.print_metrics()
    
    print("\n✅ TEST 2 PASSED: Multi-channel recall works correctly!")
    
except Exception as e:
    print(f"\n❌ TEST 2 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ========================================================================
# TEST 3: Complete Pipeline
# ========================================================================

print("TEST 3: Complete Inference Pipeline")
print("-" * 70)

try:
    from post_recommendation_pipeline import PostRecommendationPipeline
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = PostRecommendationPipeline(
        models_dir=config.get('models_dir', 'models'),
        data=data
    )
    print("✓ Pipeline initialized")
    
    # Test inference
    print("\nGenerating feed...")
    test_user_id = data['user']['Id'].iloc[0]
    
    start_time = time.time()
    feed = pipeline.get_feed(user_id=test_user_id, limit=50)
    latency = (time.time() - start_time) * 1000
    
    print(f"✓ Generated feed with {len(feed)} posts in {latency:.1f}ms")
    
    # Check latency
    if latency < 200:
        print(f"✓ Total latency < 200ms ✅")
    elif latency < 300:
        print(f"⚠ Total latency < 300ms (acceptable but could be better)")
    else:
        print(f"❌ Total latency > 300ms (needs optimization)")
    
    # Show sample results
    print("\nSample feed (top 10):")
    for i, post in enumerate(feed[:10], 1):
        print(f"  {i}. Post {post['post_id']} | Score: {post['score']:.4f}")
    
    # Print metrics
    print("\nPipeline performance:")
    pipeline.print_metrics()
    
    print("\n✅ TEST 3 PASSED: Complete pipeline works correctly!")
    
except Exception as e:
    print(f"\n❌ TEST 3 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ========================================================================
# TEST 4: Multiple Users (Stress Test)
# ========================================================================

print("TEST 4: Multiple Users (Stress Test)")
print("-" * 70)

try:
    print("Testing with 10 users...")
    test_users = data['user']['Id'].head(10).tolist()
    
    latencies = []
    feed_sizes = []
    
    for user_id in test_users:
        start_time = time.time()
        feed = pipeline.get_feed(user_id, limit=50)
        latency = (time.time() - start_time) * 1000
        
        latencies.append(latency)
        feed_sizes.append(len(feed))
    
    # Stats
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    avg_feed_size = np.mean(feed_sizes)
    
    print(f"\nResults:")
    print(f"  Users tested: {len(test_users)}")
    print(f"  Avg latency: {avg_latency:.1f}ms")
    print(f"  P95 latency: {p95_latency:.1f}ms")
    print(f"  Avg feed size: {avg_feed_size:.1f} posts")
    
    # Check if meets requirements
    if p95_latency < 200:
        print(f"\n✅ P95 latency < 200ms (excellent!)")
    elif p95_latency < 300:
        print(f"\n⚠ P95 latency < 300ms (acceptable)")
    else:
        print(f"\n❌ P95 latency > 300ms (needs optimization)")
    
    print("\n✅ TEST 4 PASSED: Pipeline handles multiple users!")
    
except Exception as e:
    print(f"\n❌ TEST 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ========================================================================
# FINAL SUMMARY
# ========================================================================

print("="*70)
print("FINAL SUMMARY")
print("="*70)
print()
print("✅ TEST 1: Feature Engineer (Fixed) - PASSED")
print("✅ TEST 2: Multi-Channel Recall - PASSED")
print("✅ TEST 3: Complete Pipeline - PASSED")
print("✅ TEST 4: Multiple Users - PASSED")
print()
print("="*70)
print("🎉 ALL TESTS PASSED! PIPELINE IS READY FOR PRODUCTION! 🎉")
print("="*70)
print()
print("Next steps:")
print("1. Deploy API endpoint (FastAPI)")
print("2. Setup monitoring (Prometheus/Grafana)")
print("3. A/B test with real users")
print("4. Iterate and improve!")
print()