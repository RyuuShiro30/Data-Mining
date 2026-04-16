## Association Rule Mining — Apriori vs FP-Growth
## Dataset: slide dosen (TID 100-500)
## ============================================================

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# ── Dataset dari slide ──────────────────────────────────────
dataset = [
    ['1', '3', '4'],       # TID 100
    ['2', '3', '5'],       # TID 200
    ['1', '2', '3', '5'],  # TID 300
    ['2', '5'],            # TID 400
    ['1', '3', '5']        # TID 500
]

# ── Encode ke one-hot ────────────────────────────────────────
te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_array, columns=te.columns_)

print("=" * 55)
print("DATASET (One-Hot Encoding)")
print("=" * 55)
print(df.to_string())

# ── Setting ──────────────────────────────────────────────────
# min support count 2 dari 5 transaksi = 2/5 = 0.4
MIN_SUPPORT    = 0.4
MIN_CONFIDENCE = 0.5

# ── APRIORI (sesuai slide dosen) ─────────────────────────────
fi_apriori = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
fi_apriori['support_count'] = (fi_apriori['support'] * len(df)).astype(int)
fi_apriori = fi_apriori.sort_values('support', ascending=False).reset_index(drop=True)

rules_apriori = association_rules(fi_apriori, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules_apriori = rules_apriori[['antecedents','consequents','support','confidence','lift']]
rules_apriori = rules_apriori.sort_values('confidence', ascending=False).reset_index(drop=True)

# ── FP-GROWTH (sesuai Altair AI Studio) ──────────────────────
fi_fp = fpgrowth(df, min_support=MIN_SUPPORT, use_colnames=True)
fi_fp['support_count'] = (fi_fp['support'] * len(df)).astype(int)
fi_fp = fi_fp.sort_values('support', ascending=False).reset_index(drop=True)

rules_fp = association_rules(fi_fp, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules_fp = rules_fp[['antecedents','consequents','support','confidence','lift']]
rules_fp = rules_fp.sort_values('confidence', ascending=False).reset_index(drop=True)

# ── Tampilkan hasil berdampingan ──────────────────────────────
print("\n" + "=" * 55)
print("FREQUENT ITEMSETS")
print("=" * 55)

print("\n--- Apriori (slide dosen) ---")
print(fi_apriori.to_string())

print("\n--- FP-Growth (Altair AI Studio) ---")
print(fi_fp.to_string())

print("\n" + "=" * 55)
print("ASSOCIATION RULES")
print("=" * 55)

print("\n--- Apriori (slide dosen) ---")
print(rules_apriori.to_string())

print("\n--- FP-Growth (Altair AI Studio) ---")
print(rules_fp.to_string())

# ── Cek apakah frequent itemset sama ─────────────────────────
set_apriori = set(
    frozenset(x) for x in fi_apriori['itemsets']
)
set_fp = set(
    frozenset(x) for x in fi_fp['itemsets']
)

print("\n" + "=" * 55)
print("PERBANDINGAN")
print("=" * 55)
if set_apriori == set_fp:
    print("Frequent itemset SAMA antara Apriori dan FP-Growth")
else:
    print("Hanya di Apriori :", set_apriori - set_fp)
    print("Hanya di FP-Growth:", set_fp - set_apriori)

print(f"Jumlah rules Apriori   : {len(rules_apriori)}")
print(f"Jumlah rules FP-Growth : {len(rules_fp)}")