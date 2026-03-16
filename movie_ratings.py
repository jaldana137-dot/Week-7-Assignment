import pandas as pd
import os

# ─────────────────────────────────────────────
# 1. Load the CSV into a DataFrame
# ─────────────────────────────────────────────
# Build the path relative to wherever THIS script lives,
# so it works no matter what directory you run it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Movie_Ratings_by_Friends.csv")

df = pd.read_csv(CSV_PATH, index_col=0, encoding="utf-8-sig")

print("=" * 60)
print("ORIGINAL RATINGS")
print("=" * 60)
print(df)

# ─────────────────────────────────────────────
# 2. Average ratings — original data
# ─────────────────────────────────────────────
print("\n--- Average rating per USER (original) ---")
print(df.mean(axis=1).round(2))

print("\n--- Average rating per MOVIE (original) ---")
print(df.mean(axis=0).round(2))

# ─────────────────────────────────────────────
# 3. Normalized ratings  (min-max per user)
#    formula: (x - min) / (max - min)
#    scales each user's ratings to [0, 1]
# ─────────────────────────────────────────────
def normalize_row(row):
    row_min = row.min()
    row_max = row.max()
    if row_max == row_min:          # avoid division by zero
        return row.apply(lambda x: 0.0 if pd.isna(x) else 0.5)
    return (row - row_min) / (row_max - row_min)

df_normalized = df.apply(normalize_row, axis=1)

print("\n" + "=" * 60)
print("NORMALIZED RATINGS  (min-max per user, range 0-1)")
print("=" * 60)
print(df_normalized.round(4))

print("\n--- Average rating per USER (normalized) ---")
print(df_normalized.mean(axis=1).round(4))

print("\n--- Average rating per MOVIE (normalized) ---")
print(df_normalized.mean(axis=0).round(4))

# ─────────────────────────────────────────────
# 4. Text-based conclusion
# ─────────────────────────────────────────────
conclusion = """
========================================================
CONCLUSION: Normalized vs. Actual Ratings
========================================================

ADVANTAGES of normalized ratings:
  - Removes personal bias in scale usage. For example, a user
    who always rates between 3-5 and a user who uses the full
    1-5 range are brought onto the same footing.
  - Makes it easier to compare which movies each user relatively
    preferred, regardless of their overall rating tendencies.
  - Useful for recommendation algorithms that need consistent
    input ranges across users.

DISADVANTAGES of normalized ratings:
  - We lose absolute information. A normalized score of 1.0
    might correspond to a 3/5 for one user and a 5/5 for another
    -- the actual enthusiasm level is hidden.
  - Sensitive to outliers: one extreme rating can distort all
    other normalized values for that user.
  - Missing values complicate the min/max calculation, so users
    with few ratings may have less reliable normalized scores.
  - Average normalized ratings per movie become harder to
    interpret in real-world terms.
========================================================
"""
print(conclusion)

# ─────────────────────────────────────────────
# 5. [Extra Credit] Standardized ratings (Z-score per user)
#    formula: (x - mean) / std
#    centers each user at 0 with std deviation of 1
# ─────────────────────────────────────────────
def standardize_row(row):
    row_mean = row.mean()
    row_std  = row.std()
    if pd.isna(row_std) or row_std == 0:
        return row.apply(lambda x: 0.0 if pd.notna(x) else float("nan"))
    return (row - row_mean) / row_std

df_standardized = df.apply(standardize_row, axis=1)

print("=" * 60)
print("STANDARDIZED RATINGS  (Z-score per user, mean=0 std=1)")
print("=" * 60)
print(df_standardized.round(4))

print("\n--- Average rating per USER (standardized) ---")
print(df_standardized.mean(axis=1).round(4))

print("\n--- Average rating per MOVIE (standardized) ---")
print(df_standardized.mean(axis=0).round(4))
