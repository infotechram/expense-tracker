"""
train_model.py
Uses scikit-learn — no torch, no transformers, no DLL errors.

CSV columns : Merchant, Category
Saves to    : <folder>/ExpenseModel/expense_model.pkl

Run:
    pip install scikit-learn pandas joblib
    python train_model.py --folder <your_folder>
"""
import argparse
import pandas as pd
import joblib
import json
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "DefaultTrainingData", "training_data.csv")

# ── Paths ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, help="Base folder containing TrainingData/training_data.csv")
args = parser.parse_args()

DATA_FOLDER = args.folder
CSV_FILE    = os.path.join(DATA_FOLDER, "TrainingData/training_data.csv")
MODEL_DIR   = os.path.join(DATA_FOLDER, "ExpenseModel")
MODEL_PATH  = os.path.join(MODEL_DIR, "expense_model.pkl")
LABEL_PATH  = os.path.join(MODEL_DIR, "label_map.json")


# ── 1. Load CSV ────────────────────────────────────────────────────
dfs = []

if os.path.exists(CSV_FILE):
    print(f"📄 Using training data from {CSV_FILE}")
    dfs.append(pd.read_csv(CSV_FILE).dropna())

if os.path.exists(DEFAULT_CSV):
    print(f"📄 Using default training data: {DEFAULT_CSV}")
    dfs.append(pd.read_csv(DEFAULT_CSV).dropna())

if not dfs:
    print(f"❌ No training data found at {CSV_FILE} or {DEFAULT_CSV}")
    exit(1)

df = pd.concat(dfs, ignore_index=True).dropna()

# ── 2. Auto-detect column names ───────────────────────────────────
cols = df.columns.tolist()
print(f"\n📄 Columns found in CSV: {cols}")

TEXT_ALIASES  = ["merchant", "text", "description", "transaction"]
LABEL_ALIASES = ["category", "label", "cat"]

text_col  = None
label_col = None

for col in cols:
    if col.lower() in TEXT_ALIASES:
        text_col = col
    if col.lower() in LABEL_ALIASES:
        label_col = col

if not text_col or not label_col:
    print(f"\n❌ Could not find required columns.")
    print(f"   Your CSV has: {cols}")
    print(f"   Expected columns named like: Merchant/Category or text/label")
    exit(1)

print(f"   Using → input: '{text_col}'  |  label: '{label_col}'\n")

# ── 3. Clean data ──────────────────────────────────────────────────
df = df[df[text_col].str.strip()  != ""]
df = df[df[label_col].str.strip() != ""]

print(f"✅ Loaded {len(df)} training examples\n")

# ── 4. Show category breakdown ─────────────────────────────────────
CATEGORIES = sorted(df[label_col].unique().tolist())
counts     = df[label_col].value_counts()

print("─" * 50)
print(f"  {'Category':<25} {'Count':>5}")
print("─" * 50)
for cat, n in sorted(counts.items()):
    bar = "█" * (n // 2)
    print(f"  {cat:<25} {n:>5}  {bar}")
print("─" * 50)
print(f"  {'TOTAL':<25} {len(df):>5}")
print("─" * 50)

# ── 5. Prepare data ────────────────────────────────────────────────
X = df[text_col].tolist()
y = df[label_col].tolist()

# ── 6. Build model pipeline ────────────────────────────────────────
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=10000,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver="lbfgs",
    ))
])

# ── 7. Evaluate with cross-validation ─────────────────────────────
print("📊 Running 5-fold cross-validation...")
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
accuracy = scores.mean()
print(f"🎯 Cross-validated Accuracy: {accuracy:.1%} ± {scores.std():.1%}\n")

# ── 8. Train on full data ──────────────────────────────────────────
print("🚀 Training model on full dataset...")
model.fit(X, y)
print("✅ Training complete!\n")

# ── 9. Test with sample merchants ─────────────────────────────────
test_merchants = [
    "Swiggy",
    "Zomato",
    "Uber",
    "Ola",
    "Amazon",
    "Flipkart",
    "BESCOM electricity",
    "Airtel postpaid",
    "Apollo Pharmacy",
    "BookMyShow",
    "To Rahul",
    "DMart",
    "Petrol bunk",
    "Netflix",
    "ATM withdrawal",
    "Paid to MITHRAN SUPER STORES",
    "Paid to SUPER SARAVANA STORES TEX CHROMPET",
    "Paid to AMARAVATHY SWEET STALL",
    "Paid to PIZZA HUT"
]

print("\n🧪 Predictions on sample merchants:")
print("─" * 62)
print(f"  {'Merchant':<28} {'Predicted':<22} Confidence")
print("─" * 62)

preds  = model.predict(test_merchants)
probas = model.predict_proba(test_merchants)

for merchant, pred, prob in zip(test_merchants, preds, probas):
    conf = max(prob)
    print(f"  {merchant:<28} {pred:<22} {conf:.0%}")

print("─" * 62)

# ── 10. Delete old model then save new ────────────────────────────
print("\n🗑️  Deleting old model...")

if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print(f"   Deleted: {MODEL_PATH}")

if os.path.exists(LABEL_PATH):
    os.remove(LABEL_PATH)
    print(f"   Deleted: {LABEL_PATH}")

# ── 11. Save model and label map ───────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_PATH)

label_map = {
    "categories": CATEGORIES,
    "label2id":   {l: i for i, l in enumerate(CATEGORIES)},
    "id2label":   {i: l for i, l in enumerate(CATEGORIES)},
}
json.dump(label_map, open(LABEL_PATH, "w"), indent=2)

print(f"""
{'='*55}
  Model saved successfully!

  Model file  →  {MODEL_PATH}
  Label map   →  {LABEL_PATH}
  Accuracy    →  {accuracy:.1%}
{'='*55}
""")