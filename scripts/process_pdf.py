import sys
import json
import os
from datetime import datetime
import pdfplumber
import re
import uuid

try:
    import fitz
    FITZ_AVAILABLE = True
except:
    FITZ_AVAILABLE = False

try:
    import transformers
    print(f"\n{'='*60}")
    print(f"🔍 Hugging Face Cache Location:")
    print(f"   {transformers.utils.TRANSFORMERS_CACHE}")
    print(f"{'='*60}\n")
except:
    pass

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False

def show_cache_info():
    """Display cache details"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    
    if os.path.exists(cache_dir):
        print(f"\n📦 Cache Directory: {cache_dir}")
        
        total_size = 0
        model_count = 0
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
                model_count += 1
        
        size_gb = total_size / (1024**3)
        print(f"💾 Total Cache Size: {size_gb:.2f} GB")
        print(f"📊 Total Files: {model_count}")
        
        print(f"\n📂 Cached Models:")
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path):
                print(f"   ✅ {item}")

def clean_description(text):
    """Clean extracted text by adding spaces between concatenated words"""
    if not text:
        return ""
    
    # Remove "Paitdo" prefix (transaction type)
    text = re.sub(r'^Paitdo\s*', '', text, flags=re.IGNORECASE)
    
    # Remove times (with various formats)
    text = re.sub(r'[\d]{1,2}:\d{2}\s*[AP]M', '', text)
    text = re.sub(r'[\d]{2}[AP]M', '', text)
    
    # Remove UPI/Transaction IDs
    text = re.sub(r'UPI\s*Transaction\s*ID[:\s]*[\w\d]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Transaction\s*ID[:\s]*[\w\d]+', '', text, flags=re.IGNORECASE)
    
    # Remove dates (handles formats like "03Mar2026" and "03Mar,2026")
    text = re.sub(r'^\d{1,2}\s*[A-Za-z]{3}\s*,?\s*\d{4}', '', text).strip()
    text = re.sub(r'\d{1,2}[A-Za-z]{3},?\d{4}', '', text).strip()
    
    # Add spaces before capital letters in mixed case (CamelCase)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Split all-caps sequences (VELIDAINSDHRRAESTAURANT → VELIDA INSDHRRA ESTAURANT)
    text = re.sub(r'([A-Z]{3,})([A-Z][a-z])', r'\1 \2', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove trailing numbers (like IDs)
    text = re.sub(r'\s+\d+$', '', text).strip()
    
    return text

def extract_pdf(pdf_path):
    """Extract transactions from PDF using PyMuPDF (fitz) for better text extraction"""
    transactions = []
    
    # Keywords to skip (header/summary rows)
    skip_keywords = ['transaction', 'statement', 'period', 'sent', 'received', 'date', 'time', 'details', 'amount']
    
    # Try PyMuPDF first for better text extraction
    if FITZ_AVAILABLE:
        try:
            print("📄 Using PyMuPDF for text extraction...")
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                # Extract text with layout preservation
                text = page.get_text()
                
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        
                        # Skip empty or short lines
                        if not line or len(line) < 5:
                            continue
                        
                        # Skip header rows
                        if any(keyword.lower() in line.lower() for keyword in skip_keywords):
                            continue
                        
                        # Look for amount (₹ symbol)
                        amount = re.search(r'₹\s*([\d,]+(?:\.\d{2})?)', line)
                        if amount and amount.group(1):
                            # Remove amount from description
                            description = re.sub(r'₹\s*[\d,]+(?:\.\d{2})?', '', line).strip()
                            
                            # Clean the description
                            description = clean_description(description)
                            
                            if description and len(description) > 3:
                                transactions.append({
                                    'description': description,
                                    'amount': amount.group(1),
                                    'raw_line': line
                                })
            
            doc.close()
            if transactions:
                return transactions
        except Exception as e:
            print(f"⚠️  PyMuPDF extraction failed: {e}, falling back to pdfplumber...")
    
    # Fallback to pdfplumber
    print("📄 Using pdfplumber for text extraction...")
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Try to extract tables first (more reliable for structured data)
            tables = page.extract_tables()
            
            if tables:
                for table in tables:
                    for row in table:
                        if row and len(row) >= 2:
                            # Convert row to strings and clean
                            row_text = [str(cell).strip() if cell else "" for cell in row]
                            row_str = " ".join(row_text)
                            
                            # Skip header rows
                            if any(keyword.lower() in row_str.lower() for keyword in skip_keywords):
                                continue
                            
                            # Look for amount (₹ symbol)
                            amount = re.search(r'₹\s*([\d,]+(?:\.\d{2})?)', row_str)
                            if amount and amount.group(1):
                                # Remove amount from description to clean it up
                                description = re.sub(r'₹\s*[\d,]+(?:\.\d{2})?', '', row_str).strip()
                                
                                # Clean the description
                                description = clean_description(description)
                                
                                if description and len(description) > 3:
                                    transactions.append({
                                        'description': description,
                                        'amount': amount.group(1),
                                        'raw_line': row_str
                                    })
            else:
                # Fallback: Extract text if no tables found
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        line = line.strip()
                        
                        # Skip short lines or headers
                        if not line or len(line) < 5:
                            continue
                        if any(keyword.lower() in line.lower() for keyword in skip_keywords):
                            continue
                        
                        # Look for amount
                        amount = re.search(r'₹\s*([\d,]+(?:\.\d{2})?)', line)
                        if amount and amount.group(1):
                            description = re.sub(r'₹\s*[\d,]+(?:\.\d{2})?', '', line).strip()
                            
                            # Clean the description
                            description = clean_description(description)
                            
                            if description and len(description) > 3:
                                transactions.append({
                                    'description': description,
                                    'amount': amount.group(1),
                                    'raw_line': line
                                })
    
    return transactions

def categorize(transactions):
    """Categorize using zero-shot AI"""
    categorized = []
    
    # Define relevant expense categories
    categories = [
        "Food & Dining",
        "Transportation",
        "Shopping",
        "Bills & Utilities",
        "Entertainment",
        "Healthcare",
        "Transfer",
        "Other"
    ]
    
    if HF_AVAILABLE:
        try:
            print("📥 Loading model from cache...")
            classifier = pipeline("zero-shot-classification", 
                                model="facebook/bart-large-mnli", device=-1)
            
            print("✅ Model loaded successfully!")
            show_cache_info()
            
            for trans in transactions:
                result = classifier(trans['description'], 
                                  categories, multi_class=False)
                
                categorized.append({
                    'description': trans['description'],
                    'amount': trans['amount'],
                    'category': result['labels'][0],
                    'confidence': f"{result['scores'][0]:.0%}",
                    'user_editable': True
                })
        except Exception as e:
            print(f"Error: {e}")
            return simple_categorize(transactions)
    else:
        return simple_categorize(transactions)
    
    return categorized

def simple_categorize(transactions):
    """Simple fallback with keyword matching"""
    keywords = {
        "Food & Dining": [
            "restaurant", "food", "cafe", "coffee", "pizza", "lunch", "dinner", "hotel", "tavern", 
            "bistro", "diner", "bar", "bakery", "bakehouse", "estaurant", "velida", "thalap", 
            "myjio", "esserliv", "saarth", "sweet", "biryani", "dhaba", "eatery", "snack"
        ],
        "Groceries": [
            "supermarket", "grocery", "market", "super", "mart", "store", "shop", "wallmark", 
            "costco", "fresh", "organic", "daily", "provisions"
        ],
        "Transportation": [
            "uber", "taxi", "auto", "travel", "bus", "train", "gas", "petrol", "fuel", "parking", 
            "metro", "ola", "cab", "commute", "transit"
        ],
        "Shopping": [
            "amazon", "mall", "flipkart", "retail", "clothing", "apparel", "store", "shop", 
            "purchase", "buy", "mart", "bazaar", "emporium"
        ],
        "Bills & Utilities": [
            "bill", "electric", "water", "gas", "internet", "phone", "mobile", "utility", 
            "telephone", "airtel", "jio", "vi", "vodafone", "bsnl", "broadband", "dth"
        ],
        "Healthcare": [
            "doctor", "hospital", "medicine", "pharmacy", "health", "clinic", "medical", 
            "doctor", "medic", "pharmacy", "care", "wellness"
        ],
        "Entertainment": [
            "movie", "cinema", "spotify", "netflix", "game", "ticket", "theatre", "theater", 
            "show", "concert", "music", "streaming"
        ],
        "Transfer": [
            "received", "transfer", "sent", "paid", "deposit", "withdraw"
        ],
    }
    
    categorized = []
    for trans in transactions:
        desc_lower = trans['description'].lower()
        category = "Other"
        
        # Match against keywords (prioritize specific matches)
        for cat, keywords_list in keywords.items():
            if any(keyword in desc_lower for keyword in keywords_list):
                category = cat
                break
        
        categorized.append({
            'description': trans['description'],
            'amount': trans['amount'],
            'category': category,
            'confidence': 'Manual (Keyword)',
            'user_editable': True
        })
    
    return categorized

def main(pdf_path):
    print(f"Processing: {pdf_path}")
    
    # Generate unique process ID and timestamp
    process_id = str(uuid.uuid4())[:8]
    process_timestamp = datetime.now().isoformat()
    
    # Extract PDF filename without extension
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_filename = f"{pdf_filename}_expenses.json"
    output_path = os.path.join('results', output_filename)
    
    transactions = extract_pdf(pdf_path)
    print(f"Found {len(transactions)} transactions")
    
    categorized = categorize(transactions)
    
    summary = {}
    total = 0
    for t in categorized:
        amount = float(t['amount'].replace(',', ''))
        cat = t['category']
        summary[cat] = summary.get(cat, 0) + amount
        total += amount
    
    results = {
        "meta": {
            "process_id": process_id,
            "processed_at": process_timestamp,
            "pdf_file": os.path.basename(pdf_path),
            "source_path": pdf_path
        },
        "status": "success",
        "timestamp": process_timestamp,
        "cache_location": os.path.expanduser("~/.cache/huggingface/hub/"),
        "transactions": categorized,
        "summary": {
            "total_spent": total,
            "by_category": summary,
            "transaction_count": len(categorized)
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Add to processing log for multi-user tracking
    log_entry = {
        "process_id": process_id,
        "processed_at": process_timestamp,
        "pdf_file": os.path.basename(pdf_path),
        "output_file": output_filename,
        "transaction_count": len(categorized),
        "total_amount": total
    }
    
    log_file = 'results/processing_log.json'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log = json.load(f)
    else:
        log = {"processing_history": []}
    
    log["processing_history"].append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"✅ Saved to {output_path}")
    print(f"📋 Process ID: {process_id}")
    print(f"📝 Tracking added to {log_file}")

if __name__ == "__main__":
    main(sys.argv[1])