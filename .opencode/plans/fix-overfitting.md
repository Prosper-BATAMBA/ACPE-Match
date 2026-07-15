# Fix Overfitting in CatBoost Ranker

## Problem
1. `full_evaluation_v2.py` evaluates on **identical candidates** as `train_ranker.py` (100% overlap)
2. `train_ranker.py` Hit Rate is computed globally (all pairs pooled), not per-query → trivially 100%

## Fix 1: `train_ranker.py` — Per-Query Hit Rate

**File:** `backend/train_ranker.py`
**Lines:** 379-403

Replace the broken evaluation block:

```python
    print(f"\n[5/5] Evaluation du modele...")

    test_preds = model.predict(test_pool)

    test_sorted_indices = np.argsort(test_preds)[::-1]
    test_sorted_labels = y_test[test_sorted_indices]

    hr = {}
    ndcg = {}
    for k in [1, 3, 5, 10]:
        top_k = test_sorted_labels[:k]
        hr[k] = 1 if any(l == 1 for l in top_k) else 0

        dcg = sum(l / math.log2(i + 2) for i, l in enumerate(top_k))
        ideal = sorted(test_sorted_labels, reverse=True)[:k]
        idcg = sum(l / math.log2(i + 2) for i, l in enumerate(ideal))
        ndcg[k] = dcg / idcg if idcg > 0 else 0

    print(f"\n  --- Hit Rate (test set) ---")
    for k in [1, 3, 5, 10]:
        print(f"  Hit Rate@{k}: {hr[k]:.4f}")

    print(f"\n  --- NDCG (test set) ---")
    for k in [1, 3, 5, 10]:
        print(f"  NDCG@{k}: {ndcg[k]:.4f}")
```

With this:

```python
    print(f"\n[5/5] Evaluation du modele (per-query)...")

    test_preds = model.predict(test_pool)

    from collections import defaultdict
    query_indices = defaultdict(list)
    for i, qid in enumerate(test_qids):
        query_indices[qid].append(i)

    hr = {k: [] for k in [1, 3, 5, 10]}
    ndcg = {k: [] for k in [1, 3, 5, 10]}

    for qid, indices in query_indices.items():
        preds = test_preds[indices]
        labels = y_test[indices]
        sorted_idx = np.argsort(preds)[::-1]
        sorted_labels = labels[sorted_idx]
        for k in [1, 3, 5, 10]:
            top_k = sorted_labels[:k]
            hr[k].append(1 if any(l == 1 for l in top_k) else 0)
            dcg = sum(l / math.log2(i + 2) for i, l in enumerate(top_k))
            ideal = sorted(labels, reverse=True)[:k]
            idcg = sum(l / math.log2(i + 2) for i, l in enumerate(ideal))
            ndcg[k].append(dcg / idcg if idcg > 0 else 0)

    print(f"\n  --- Hit Rate (test set, {len(query_indices)} queries) ---")
    for k in [1, 3, 5, 10]:
        print(f"  Hit Rate@{k}: {np.mean(hr[k]):.4f}")

    print(f"\n  --- NDCG (test set) ---")
    for k in [1, 3, 5, 10]:
        print(f"  NDCG@{k}: {np.mean(ndcg[k]):.4f}")
```

## Fix 2: `full_evaluation_v2.py` — Disjoint Candidates

**File:** `backend/full_evaluation_v2.py`
**Line:** 251

Replace:

```python
    selected = selected[:N_CANDIDATES]
```

With:

```python
    selected = selected[N_CANDIDATES:N_CANDIDATES * 2]
```

This uses candidates 5000-10000 (never seen during training).

## Steps

1. Apply Fix 1 to `train_ranker.py`
2. Apply Fix 2 to `full_evaluation_v2.py`
3. Run `cd backend && python train_ranker.py`
4. Run `cd backend && python full_evaluation_v2.py`
5. Report new metrics
