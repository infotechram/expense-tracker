import sys
import json
import os
from datetime import datetime
import pdfplumber
import re
import uuid

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

try:
    import transformers
    print(f"\n{'='*60}")
    print(f"🔍 Hugging Face Cache Location:")
    print(f"   {transformers.utils.TRANSFORMERS_CACHE}")
    print(f"{'='*60}\n")
except Exception:
    pass

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ─── Cache Info ───────────────────────────────────────────────────────────────

def show_cache_info():
    """Display Hugging Face cache details."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")

    if os.path.exists(cache_dir):
        print(f"\n📦 Cache Directory: {cache_dir}")

        total_size = 0
        model_count = 0
        for dirpath, _, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
                model_count += 1

        size_gb = total_size / (1024 ** 3)
        print(f"💾 Total Cache Size: {size_gb:.2f} GB")
        print(f"📊 Total Files: {model_count}")

        print(f"\n📂 Cached Models:")
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path):
                print(f"   ✅ {item}")


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

    # ✅ CamelCase split: swiggyFood → swiggy Food
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # ✅ ALL-CAPS prefix split: "SBIBank" → "SBI Bank"
    text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', text)

    # ✅ Letter → digit boundary: "Swiggy123" → "Swiggy 123"
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    # ✅ Digit → letter boundary: "123Swiggy" → "123 Swiggy"
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


def is_summary_row(text: str) -> bool:
    """
    Detects summary/header rows like 'Sent ₹...' or 'Received ₹...'
    but allows peer-to-peer transfers with 'to'/'from'.
    """
    return (
        re.search(r'\b(sent|received)\b', text, re.IGNORECASE)
        and not re.search(r'\bto\b|\bfrom\b', text, re.IGNORECASE)
        and len(text.split()) <= 3
    )

def _is_header_row(text: str) -> bool:
    text_lower = text.lower()

    # Skip obvious statement period headers
    if "transaction statement period" in text_lower:
        return True

    # Skip summary rows like "Sent ₹..." or "Received ₹..."
    if re.search(r'\b(sent|received)\b', text_lower):
        # If no counterparty mentioned, treat as summary
        if not re.search(r'\bto\b|\bfrom\b', text_lower):
            return True

    # Skip other header keywords only if no amount present
    if any(kw in text_lower for kw in SKIP_KEYWORDS):
        if not re.search(r'₹\s*[\d,]+(?:\.\d{2})?', text):
            return True

    return False



def extract_with_fitz(pdf_path: str) -> list:
    """
    Layout-aware extraction using PyMuPDF dict mode.
    """
    transactions = []
    doc = fitz.open(pdf_path)

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        rows: dict[float, list[str]] = {}
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                # Use tolerance bucket (±5px) for better row grouping
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

            description = re.sub(r'₹\s*[\d,]+(?:\.\d{2})?', '', row_text).strip()
            description = clean_description(description)

           # Keep transactions if they have an amount and at least some description
            if amount_match and description:
                transactions.append({
                    'description': description,
                    'amount': amount_match.group(1),
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

                        description = re.sub(r'₹\s*[\d,]+(?:\.\d{2})?', '', row_text).strip()
                        description = clean_description(description)

                        if description and len(description) > 3:
                            transactions.append({
                                'description': description,
                                'amount': amount_match.group(1),
                                'raw_line': row_text
                            })
            else:
                # No tables found — fall back to raw text
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

                    description = re.sub(r'₹\s*[\d,]+(?:\.\d{2})?', '', line).strip()
                    description = clean_description(description)

                    if description and len(description) > 3:
                        transactions.append({
                            'description': description,
                            'amount': amount_match.group(1),
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

CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Transportation",
    "Shopping",
    "Bills & Utilities",
    "Entertainment",
    "Healthcare",
    "Transfer",
    "Other"
]

# Contact-to-category memory (persisted across runs)
CONTACT_MAP_FILE = "results/contact_map.json"


def load_contact_map() -> dict:
    if os.path.exists(CONTACT_MAP_FILE):
        with open(CONTACT_MAP_FILE) as f:
            return json.load(f)
    return {}


def save_contact_map(contact_map: dict):
    os.makedirs("results", exist_ok=True)
    with open(CONTACT_MAP_FILE, "w") as f:
        json.dump(contact_map, f, indent=2)


def extract_contact_name(description: str):
    """Try to extract a person's name from a 'paid to <name>' description."""
    match = re.search(
        r'(?:to|paid)\s+([A-Za-z][A-Za-z\s]{2,30}?)(?:\s+via|\s+on|\s+ref|$)',
        description, re.IGNORECASE
    )
    if match:
        name = match.group(1).strip().lower()
        # Filter out merchant-like names (contain digits or are too long)
        if not re.search(r'\d', name) and len(name.split()) <= 4:
            return name
    return None


def is_personal_transfer(description: str) -> bool:
    """Detect if the transaction looks like a peer-to-peer payment."""
    personal_patterns = [
        r'\bsent\s+to\b', r'\bpaid\s+to\b', r'\btransfer(red)?\s+to\b',
        r'\bvia\s+(gpay|google\s*pay|upi|phonepe|paytm)\b'
    ]
    desc_lower = description.lower()
    return any(re.search(p, desc_lower) for p in personal_patterns)


def categorize_hf_batch(classifier, descriptions: list) -> list:
    """
    ✅ Batch zero-shot — runs all at once instead of one-by-one.
    multi_label=False fixes the deprecated multi_class warning.
    batch_size=16 speeds up CPU inference significantly.
    """
    results = classifier(descriptions, CATEGORIES, multi_label=False, batch_size=16)
    if isinstance(results, dict):   # single item returned as dict
        results = [results]
    return [
        {
            'category': r['labels'][0],
            'confidence': f"{r['scores'][0]:.0%}",
            'source': 'zero-shot-AI'
        }
        for r in results
    ]


def simple_categorize_one(description: str) -> dict:
    """Keyword + optional fuzzy matching fallback."""
    keywords = {
        "Food & Dining": [
            "restaurant", "food", "cafe", "coffee", "pizza", "lunch", "dinner",
            "hotel", "tavern", "bistro", "diner", "bakery", "sweet", "biryani",
            "dhaba", "eatery", "snack", "swiggy", "zomato", "burger", "barbeque"
        ],
        "Groceries": [
            "supermarket", "grocery", "market", "mart", "store", "fresh",
            "organic", "reliance smart", "dmart", "big bazaar", "more"
        ],
        "Transportation": [
            "uber", "taxi", "auto", "travel", "bus", "train", "petrol",
            "fuel", "parking", "metro", "ola", "cab", "rapido", "irctc", "flight"
        ],
        "Shopping": [
            "amazon", "flipkart", "retail", "clothing", "apparel", "bazaar",
            "myntra", "ajio", "mall", "nykaa", "meesho"
        ],
        "Bills & Utilities": [
            "bill", "electric", "electricity", "water", "gas", "internet",
            "phone", "mobile", "airtel", "jio", "vodafone", "bsnl",
            "broadband", "dth", "bescom", "tneb", "postpaid", "recharge"
        ],
        "Healthcare": [
            "doctor", "hospital", "medicine", "pharmacy", "health",
            "clinic", "medical", "apollo", "diagnostic", "lab", "dental"
        ],
        "Entertainment": [
            "movie", "cinema", "spotify", "netflix", "game", "ticket",
            "theatre", "show", "pvr", "inox", "bookmyshow", "hotstar", "prime"
        ],
        "Transfer": [
            "received", "transfer", "sent", "deposit", "withdraw", "wallet"
        ],
    }

    desc_lower = description.lower()

    # Exact substring match
    for cat, kw_list in keywords.items():
        if any(kw in desc_lower for kw in kw_list):
            return {'category': cat, 'confidence': 'keyword-match', 'source': 'keyword'}

    # Fuzzy match via rapidfuzz
    if RAPIDF_AVAILABLE:
        flat = [(kw, cat) for cat, kw_list in keywords.items() for kw in kw_list]
        choices = [k for k, _ in flat]
        try:
            res = rf_process.extractOne(
                desc_lower, choices,
                scorer=rf_fuzz.token_sort_ratio,
                score_cutoff=60
            )
            if res:
                match_text, score, idx = res[0], res[1], res[2] if len(res) > 2 else choices.index(res[0])
                _, matched_cat = flat[idx]
                return {
                    'category': matched_cat,
                    'confidence': f'fuzzy-{int(score)}%',
                    'source': 'fuzzy'
                }
        except Exception:
            pass

    return {'category': 'Other', 'confidence': 'no-match', 'source': 'fallback'}


def categorize(transactions: list) -> list:
    contact_map = load_contact_map()
    categorized = []

    hf_classifier = None
    if HF_AVAILABLE:
        try:
            hf_classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-MiniLM2-L6-H768",
                device=-1
            )
        except Exception as e:
            print(f"⚠️ HF model load failed: {e}. Using fallback.")

    # Batch AI classification if available
    if hf_classifier:
        batch_descs = [t['description'] for t in transactions]
        batch_output = hf_classifier(batch_descs, CATEGORIES, multi_label=False)

        if isinstance(batch_output, dict):
            batch_output = [batch_output]

    for i, trans in enumerate(transactions):
        desc = trans['description']
        contact = extract_contact_name(desc)

        # Contact map override
        if contact and contact in contact_map:
            result = {
                'category': contact_map[contact],
                'confidence': '100% (learned)',
                'source': 'contact-map'
            }
        elif hf_classifier:
            res = batch_output[i]
            result = {
                'category': res['labels'][0],
                'confidence': f"{res['scores'][0]:.0%}",
                'source': 'zero-shot-AI'
            }
        else:
            # Fallback only if AI unavailable
            result = simple_categorize_one(desc)

        categorized.append({
            'description': trans['description'],
            'amount': trans['amount'],
            'category': result['category'],
            'confidence': result['confidence'],
            'source': result['source'],
            'raw_line': trans.get('raw_line', ''),
            'user_editable': True
        })

    return categorized



# ─── Main ──────────────────────────────────────────────────────────────────────

def main(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n🚀 Processing: {pdf_path}")

    process_id = str(uuid.uuid4())[:8]
    process_timestamp = datetime.now().isoformat()

    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_filename = f"{pdf_filename}_expenses.json"
    output_path = os.path.join('results', output_filename)
    os.makedirs('results', exist_ok=True)

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
    total = 0.0
    for t in categorized:
        amount = float(t['amount'].replace(',', ''))
        cat = t['category']
        summary[cat] = summary.get(cat, 0.0) + amount
        total += amount

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "meta": {
            "process_id": process_id,
            "processed_at": process_timestamp,
            "pdf_file": os.path.basename(pdf_path),
            "source_path": pdf_path
        },
        "status": "success",
        "transactions": categorized,
        "summary": {
            "total_spent": round(total, 2),
            "by_category": {k: round(v, 2) for k, v in summary.items()},
            "transaction_count": len(categorized)
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Processing log ────────────────────────────────────────────────────
    log_file = 'results/processing_log.json'
    log_entry = {
        "process_id": process_id,
        "processed_at": process_timestamp,
        "pdf_file": os.path.basename(pdf_path),
        "output_file": output_filename,
        "transaction_count": len(categorized),
        "total_amount": round(total, 2)
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
    if len(sys.argv) < 2:
        print("Usage: python gpay_expense_categorizer.py <path_to_pdf>")
        sys.exit(1)
    main(sys.argv[1])