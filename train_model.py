"""
train_model.py
Uses scikit-learn — no torch, no transformers, no DLL errors.

CSV columns : Merchant, Category
Saves to    : results/expense_model.pkl

Run:
    pip install scikit-learn pandas joblib
    python train_model.py
"""
import argparse
import pandas as pd
import joblib
import json
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "DefaultTrainingData", "training_data.csv")

# ── Paths ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, help="Base folder containing TrainingData/training_data.csv")
args = parser.parse_args()

DATA_FOLDER   = args.folder
CSV_FILE      = os.path.join(DATA_FOLDER, "TrainingData/training_data.csv")
MODEL_DIR     = os.path.join(DATA_FOLDER, "ExpenseModel")
MODEL_PATH    = os.path.join(MODEL_DIR, "expense_model.pkl")
LABEL_PATH    = os.path.join(MODEL_DIR, "label_map.json")


# ── 1. Load CSV ────────────────────────────────────────────────────
dfs = []

if os.path.exists(CSV_FILE):
    print(f"📄 Using training data from {CSV_FILE}")
    dfs.append(pd.read_csv(CSV_FILE).dropna())

if os.path.exists(DEFAULT_CSV):
    print(f"⚠️ Using default training data: {DEFAULT_CSV}")
    dfs.append(pd.read_csv(DEFAULT_CSV).dropna())

if not dfs:
    print(f"❌ No training data found at {CSV_FILE} or {DEFAULT_CSV}")
    exit(1)

df = pd.concat(dfs, ignore_index=True).dropna()

# ── Auto-detect column names ───────────────────────────────────────
# Handles: Merchant/Category, text/label, description/category etc.
cols = df.columns.tolist()
print(f"\n📄 Columns found in CSV: {cols}")

# Map whatever columns exist to text_col and label_col
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

# ── 2. Clean data ──────────────────────────────────────────────────
df = df[df[text_col].str.strip()  != ""]
df = df[df[label_col].str.strip() != ""]

print(f"✅ Loaded {len(df)} training examples\n")

# ── 3. Show category breakdown ─────────────────────────────────────
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

# ── 4. Split into train / test ─────────────────────────────────────
X = df[text_col].tolist()
y = df[label_col].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n📊 Train : {len(X_train)} samples")
print(f"   Test  : {len(X_test)}  samples\n")

# ── 5. Build model pipeline ────────────────────────────────────────
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

# ── 6. Train ───────────────────────────────────────────────────────
print("🚀 Training model...")
model.fit(X_train, y_train)
print("✅ Training complete!\n")

# ── 7. Evaluate ────────────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = (pd.Series(y_pred) == pd.Series(y_test)).mean()

print(f"🎯 Overall Accuracy : {accuracy:.1%}\n")
print("📋 Per category results:")
print(classification_report(y_test, y_pred, zero_division=0))

# ── 8. Test with sample merchants ─────────────────────────────────
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

# ── 8. Training succeeded → now delete old model → save new ───────
#
#   This block runs ONLY if training completed without errors.
#   If training failed above, exit(1) was called and we never reach here.
#   So the old model is always safe until we are 100% ready to replace it.
#
print("\n🗑️  Deleting old model...")
 
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print(f"   Deleted: {MODEL_PATH}")
 
if os.path.exists(LABEL_PATH):
    os.remove(LABEL_PATH)
    print(f"   Deleted: {LABEL_PATH}")

# ── 9. Save model and label map ────────────────────────────────────

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

  Next steps:
  1. Upload results/expense_model.pkl to GitHub
  2. process_pdf.py will load it automatically

  To retrain later with more data:
    python collect_training_data.py
    python train_model.py
{'='*55}
""")