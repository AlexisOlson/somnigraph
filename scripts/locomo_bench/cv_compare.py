import sys, pickle, warnings, numpy as np
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')

from locomo_bench.train_locomo_reranker import (
    FEATURE_NAMES, FEATURES_PATH, train_cv, summarize_metrics
)

with open(FEATURES_PATH, 'rb') as f:
    saved = pickle.load(f)

acf = saved['conv_features']
acd = saved['conv_data']
asd = saved['search_data']

def fi(names):
    return sorted([FEATURE_NAMES.index(n) for n in names])

base13 = ['fts_rank', 'fts_bm25', 'vec_dist', 'theme_overlap', 'query_coverage',
          'speaker_match', 'query_length', 'neighbor_density', 'score_percentile',
          'entity_overlap', 'theme_complementarity', 'graph_edge_count', 'graph_synthetic_score']

base15_no_session = base13 + ['vec_rank', 'has_temporal_expr']

configs = {
    # Round 3: neighbor/session/coref swaps
    '15_no_neighbor': fi([f for f in base13 if f != 'neighbor_density'] + ['vec_rank', 'has_temporal_expr', 'session_cooccurrence']),
    '14_ses_coref':   fi([f for f in base15_no_session if f != 'neighbor_density'] + ['graph_coref_hits']),
    '16_coref_no_ses': fi(base15_no_session + ['graph_coref_hits']),
    '16_rrf_no_ses':   fi(base15_no_session + ['phase1_rrf_score']),
    '13_no_ses_vr_nb': fi([f for f in base15_no_session if f not in ('vec_rank', 'neighbor_density')]),
}

seeds = [42, 123, 69, 420]

# Collect results
results = {}  # config -> {metric -> [values per seed]}
for cname, indices in configs.items():
    print(f"\n=== {cname} ({len(indices)} features) ===", flush=True)
    results[cname] = {'r@10': [], 'ndcg@10': [], 'mrr': [], 'r@20': []}
    for rs in seeds:
        metrics, _ = train_cv(acf, acd, asd, indices, n_estimators=300, random_state=rs)
        for m in ['r@10', 'ndcg@10', 'mrr', 'r@20']:
            val = np.mean([q[m] for q in metrics])
            results[cname][m].append(val)
        print(f"  seed {rs}: R@10={results[cname]['r@10'][-1]:.4f}  NDCG={results[cname]['ndcg@10'][-1]:.4f}  MRR={results[cname]['mrr'][-1]:.4f}  R@20={results[cname]['r@20'][-1]:.4f}", flush=True)

# Summary table
print(f"\n{'='*90}")
print(f"{'Config':<20s} {'R@10 mean':>10s} {'R@10 std':>9s} {'NDCG mean':>10s} {'MRR mean':>10s} {'R@20 mean':>10s}")
print(f"{'-'*90}")
for cname in configs:
    r = results[cname]
    print(f"{cname:<20s} {np.mean(r['r@10']):10.4f} {np.std(r['r@10']):9.4f} {np.mean(r['ndcg@10']):10.4f} {np.mean(r['mrr']):10.4f} {np.mean(r['r@20']):10.4f}")
