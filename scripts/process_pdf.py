"""
process_pdf.py
Extracts transactions from a GPay PDF statement and categorizes them
using a trained scikit-learn model.

Run:
    pip install scikit-learn pdfplumber joblib pymupdf rapidfuzz
    python process_pdf.py --folder "C:/MyExpenseApp" "march_statement.pdf"
"""

import sys
import json
import os
import argparse
from datetime import datetime
import pdfplumber
import re
import uuid
import joblib
import re  # Ensure this is imported

# YOU MUST PASTE THIS EXACTLY AS IT IS IN YOUR TRAINING SCRIPT
def clean_descriptions(texts):
    cleaned = []
    for text in texts:
        t = str(text).upper()
        # Remove transaction noise
        t = re.sub(r'\b(PAID TO|SENT TO|TO|PAYMENT FROM|UPI|PVT|LTD|LIMITED|TECHNOLOGIES)\b', '', t)
        # Remove numbers and special characters
        t = re.sub(r'[^A-Z\s]', ' ', t)
        cleaned.append(" ".join(t.split()))
    return cleaned

# ─── Optional Dependencies ────────────────────────────────────────────────────

try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    print("⚠️  PyMuPDF (fitz) not installed. Falling back to pdfplumber.")

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    RAPIDF_AVAILABLE = True
except ImportError:
    RAPIDF_AVAILABLE = False


# ─── Argument Parsing ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, help="Base folder containing ExpenseModel/")
parser.add_argument("pdf", help="Path to the PDF file to process")
args = parser.parse_args()

MODEL_PATH = os.path.join(args.folder, "ExpenseModel", "expense_model.pkl")
LABEL_PATH = os.path.join(args.folder, "ExpenseModel", "label_map.json")


# ─── Load sklearn Model ───────────────────────────────────────────────────────

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print(f"   Please train the model first using train_model.py --folder {args.folder}")
    sys.exit(1)

if not os.path.exists(LABEL_PATH):
    print(f"❌ Label map not found: {LABEL_PATH}")
    sys.exit(1)

print(f"✅ Loading model from: {MODEL_PATH}")
sk_model = joblib.load(MODEL_PATH)

with open(LABEL_PATH) as f:
    label_map = json.load(f)

CATEGORIES = label_map["categories"]
print(f"   Categories: {CATEGORIES}\n")


# ─── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_description(text):
    """
    Clean extracted transaction description.

    Fixes:
    - Normalizes 'Paid to' → 'To <name>' instead of stripping
    - Removes timestamps (12:30 PM, 08AM)
    - Removes dates (03Mar,2026 | 03 Mar 2026 | 03Mar2026)
    - Removes UPI / transaction IDs
    - Removes long numeric IDs (10+ digits)
    - Splits CamelCase words  (SwiggyFood → Swiggy Food)
    - Splits letter+digit boundaries (Swiggy123 → Swiggy 123)
    - Splits digit+letter boundaries (123Swiggy → 123 Swiggy)
    """
    if not text:
        return ""

    # Normalize "Paid to" prefix instead of removing
    text = re.sub(r'^Pait?do?\s*', 'To ', text, flags=re.IGNORECASE)
    text = re.sub(r'^Paid\s+to\s*', 'To ', text, flags=re.IGNORECASE)

    # Remove timestamps: 12:30 PM, 08:00AM, 8AM
    text = re.sub(r'\b\d{1,2}:\d{2}\s*[AP]M\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{2}[AP]M\b', '', text, flags=re.IGNORECASE)

    # Remove dates: 03Mar,2026 | 03 Mar 2026 | 03Mar2026
    text = re.sub(r'\b\d{1,2}\s*[A-Za-z]{3}\s*,?\s*\d{4}\b', '', text)

    # Remove UPI transaction IDs
    text = re.sub(r'UPI\s*Transaction\s*ID[:\s]*\w+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Transaction\s*ID[:\s]*\w+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Ref\s*No[:\s]*\w+', '', text, flags=re.IGNORECASE)

    # Remove long numeric IDs (10+ digits)
    text = re.sub(r'\b\d{10,}\b', '', text)

    # CamelCase split: swiggyFood → swiggy Food
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # ALL-CAPS prefix split: "SBIBank" → "SBI Bank"
    text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', text)

    # Letter → digit boundary: "Swiggy123" → "Swiggy 123"
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    # Digit → letter boundary: "123Swiggy" → "123 Swiggy"
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)

    # Remove leftover leading numbers / punctuation
    text = re.sub(r'^[\d\s,.\-/]+', '', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ─── PDF Extraction ────────────────────────────────────────────────────────────

SKIP_KEYWORDS = [
    'transaction', 'statement', 'period', 'date', 'time',
    'details', 'amount', 'balance', 'opening', 'closing', 'total',
    'summary', 'march', 'january', 'february',
    'april', 'may', 'june', 'july', 'august', 'september',
    'october', 'november', 'december'
]


def extract_date_info(raw_line: str) -> dict:
    """Extract date and day of week from raw transaction line."""
    match = re.search(r'(\d{1,2})\s*([A-Za-z]{3})\s*,?\s*(\d{4})', raw_line)
    if match:
        try:
            date_str = f"{match.group(1)} {match.group(2)} {match.group(3)}"
            dt = datetime.strptime(date_str, "%d %b %Y")
            return {
                "date": dt.strftime("%d-%b-%Y"),
                "day_of_week": dt.strftime("%A")
            }
        except ValueError:
            pass
    return {"date": None, "day_of_week": None}


def _is_header_row(text: str) -> bool:
    text_lower = text.lower()

    if "transaction statement period" in text_lower:
        return True

    if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', text_lower):
        if not re.search(r'\bto\b|\bfrom\b', text_lower):
            return True

    if re.search(r'\b(sent|received)\b', text_lower):
        if not re.search(r'\bto\b|\bfrom\b', text_lower):
            return True

    if any(kw in text_lower for kw in SKIP_KEYWORDS):
        if not re.search(r'₹\s*[\d,]+(?:\.\d{2})?', text):
            return True

    return False


def is_received_transaction(raw_line: str) -> bool:
    """Returns True if the transaction is money received (not spent)."""
    text = raw_line.lower()
    return bool(
        re.search(r'\breceived\s+from\b', text) or
        re.search(r'^from\s+[a-z]', text) or
        re.search(r'\bcredited\b', text)
    )


def extract_with_fitz(pdf_path: str) -> list:
    """Layout-aware extraction using PyMuPDF dict mode."""
    transactions = []
    doc = fitz.open(pdf_path)

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        rows: dict[float, list[str]] = {}
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                y = round(line["bbox"][1] / 5) * 5
                for span in line.get("spans", []):
                    span_text = span["text"].strip()
                    if span_text:
                        rows.setdefault(y, []).append(span_text)

        for y_pos in sorted(rows.keys()):
            row_text = " ".join(rows[y_pos]).strip()

            if not row_text or len(row_text) < 5:
                continue
            if _is_header_row(row_text):
                continue

            amount_match = re.search(r'₹\s*([\d,]+(?:\.\d{2})?)', row_text)
            if not amount_match:
                continue

            date_info = extract_date_info(row_text)
            description = re.sub(r'₹\s*[\d,]+(?:\.\d{2})?', '', row_text).strip()
            description = clean_description(description)

            if amount_match and description:
                if is_received_transaction(row_text):
                    continue
                transactions.append({
                    'description': description,
                    'amount': amount_match.group(1),
                    'date': date_info['date'],
                    'day_of_week': date_info['day_of_week'],
                    'raw_line': row_text
                })

    doc.close()
    return transactions


def extract_with_pdfplumber(pdf_path: str) -> list:
    """Fallback extraction using pdfplumber (table-first, then text)."""
    transactions = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()

            if tables:
                for table in tables:
                    for row in table:
                        if not row or len(row) < 2:
                            continue
                        row_text = " ".join(
                            str(cell).strip() for cell in row if cell
                        )
                        if _is_header_row(row_text):
                            continue

                        amount_match = re.search(r'₹\s*([\d,]+(?:\.\d{2})?)', row_text)
                        if not amount_match:
                            continue

                        date_info = extract_date_info(row_text)
                        description = re.sub(r'₹\s*[\d,]+(?:\.\d{2})?', '', row_text).strip()
                        description = clean_description(description)

                        if description and len(description) > 3:
                            if is_received_transaction(row_text):
                                continue
                            transactions.append({
                                'description': description,
                                'amount': amount_match.group(1),
                                'date': date_info['date'],
                                'day_of_week': date_info['day_of_week'],
                                'raw_line': row_text
                            })
            else:
                text = page.extract_text()
                if not text:
                    continue

                for line in text.split('\n'):
                    line = line.strip()
                    if not line or len(line) < 5:
                        continue
                    if _is_header_row(line):
                        continue

                    amount_match = re.search(r'₹\s*([\d,]+(?:\.\d{2})?)', line)
                    if not amount_match:
                        continue

                    date_info = extract_date_info(line)
                    description = re.sub(r'₹\s*[\d,]+(?:\.\d{2})?', '', line).strip()
                    description = clean_description(description)

                    if description and len(description) > 3:
                        if is_received_transaction(line):
                            continue
                        transactions.append({
                            'description': description,
                            'amount': amount_match.group(1),
                            'date': date_info['date'],
                            'day_of_week': date_info['day_of_week'],
                            'raw_line': line
                        })

    return transactions


def extract_pdf(pdf_path: str) -> list:
    """Try PyMuPDF first (layout-aware), fall back to pdfplumber."""
    if FITZ_AVAILABLE:
        try:
            print("📄 Extracting with PyMuPDF (layout-aware dict mode)...")
            transactions = extract_with_fitz(pdf_path)
            if transactions:
                print(f"   ✅ Found {len(transactions)} transactions via PyMuPDF")
                return transactions
            print("   ⚠️  No transactions found via PyMuPDF, trying pdfplumber...")
        except Exception as e:
            print(f"   ⚠️  PyMuPDF failed: {e}")

    print("📄 Extracting with pdfplumber...")
    transactions = extract_with_pdfplumber(pdf_path)
    print(f"   ✅ Found {len(transactions)} transactions via pdfplumber")
    return transactions


# ─── Categorization ────────────────────────────────────────────────────────────

def categorize(transactions: list) -> list:
    """Categorize all transactions using the trained sklearn model."""
    categorized = []

    descriptions = [t['description'] for t in transactions]

    # Batch predict — fast single call for all transactions
    preds  = sk_model.predict(descriptions)
    probas = sk_model.predict_proba(descriptions)

    for i, trans in enumerate(transactions):
        conf = max(probas[i])
        categorized.append({
            'description': trans['description'],
            'amount':      trans['amount'],
            'date':        trans.get('date'),
            'day_of_week': trans.get('day_of_week'),
            'category':    preds[i],
            'confidence':  f"{conf:.0%}",
            'source':      'sklearn',
            'raw_line':    trans.get('raw_line', ''),
            'user_editable': True
        })

    return categorized


# ─── Main ──────────────────────────────────────────────────────────────────────

def main(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n🚀 Processing: {pdf_path}")

    process_id        = str(uuid.uuid4())[:8]
    process_timestamp = datetime.now().isoformat()

    pdf_filename    = os.path.splitext(os.path.basename(pdf_path))[0]
    output_filename = f"{pdf_filename}_expenses.json"
    results_dir = os.path.join(args.folder, 'results')
    output_path = os.path.join(results_dir, output_filename)
    os.makedirs(results_dir, exist_ok=True)

    # ── Extract ───────────────────────────────────────────────────────────
    transactions = extract_pdf(pdf_path)
    if not transactions:
        print("❌ No transactions found in PDF. Check if the PDF has selectable text.")
        sys.exit(1)

    print(f"\n📊 Found {len(transactions)} transactions")

    # ── Categorize ────────────────────────────────────────────────────────
    categorized = categorize(transactions)

    # ── Summary ───────────────────────────────────────────────────────────
    summary: dict[str, float] = {}
    by_day:  dict[str, float] = {}
    DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    total = 0.0

    for t in categorized:
        amount = float(t['amount'].replace(',', ''))
        cat    = t['category']
        day    = t.get('day_of_week') or 'Unknown'
        summary[cat] = summary.get(cat, 0.0) + amount
        by_day[day]  = by_day.get(day, 0.0) + amount
        total += amount

    by_day_sorted = {d: round(by_day[d], 2) for d in DAY_ORDER if d in by_day}
    if 'Unknown' in by_day:
        by_day_sorted['Unknown'] = round(by_day['Unknown'], 2)

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "meta": {
            "process_id":   process_id,
            "processed_at": process_timestamp,
            "pdf_file":     os.path.basename(pdf_path),
            "source_path":  pdf_path,
            "model_path":   MODEL_PATH
        },
        "status": "success",
        "transactions": categorized,
        "summary": {
            "total_spent":       round(total, 2),
            "by_category":       {k: round(v, 2) for k, v in summary.items()},
            "by_day_of_week":    by_day_sorted,
            "transaction_count": len(categorized)
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Processing log ────────────────────────────────────────────────────
    log_file = os.path.join(args.folder, 'results', 'processing_log.json')
    log_entry = {
        "process_id":       process_id,
        "processed_at":     process_timestamp,
        "pdf_file":         os.path.basename(pdf_path),
        "output_file":      output_filename,
        "transaction_count": len(categorized),
        "total_amount":     round(total, 2)
    }

    if os.path.exists(log_file):
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = {"processing_history": []}

    log["processing_history"].append(log_entry)
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  💰 EXPENSE SUMMARY")
    print(f"{'='*55}")
    for cat, amt in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25} ₹{amt:>10,.2f}")
    print(f"{'─'*55}")
    print(f"  {'TOTAL':<25} ₹{total:>10,.2f}")
    print(f"{'='*55}")
    print(f"\n✅ Saved  → {output_path}")
    print(f"📋 ID     → {process_id}")
    print(f"📝 Log    → {log_file}\n")


if __name__ == "__main__":
    main(args.pdf)