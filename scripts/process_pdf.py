import sys
import json
import os
from datetime import datetime
import pdfplumber
import re
import uuid

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

def extract_pdf(pdf_path):
    """Extract transactions from PDF"""
    transactions = []
    
    # Keywords to skip (header/summary rows)
    skip_keywords = ['Transaction statement period', 'Sent', 'Received', 'Date & time', 'Transaction details', 'Amount']
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            current_description = []
            
            for line in text.split('\n'):
                line = line.strip()
                
                # Skip header/summary rows
                if any(keyword.lower() in line.lower() for keyword in skip_keywords):
                    continue
                
                # Skip empty lines
                if not line or len(line) < 5:
                    continue
                
                amount = re.search(r'₹\s*([\d,]+(?:\.\d{2})?)', line)
                
                if amount:
                    # This line contains an amount - it's likely the amount line of a transaction
                    # Look for description in previous lines
                    description = ' '.join(current_description) if current_description else line
                    
                    # Clean up the description - remove UPI/Transaction IDs
                    description = re.sub(r'UPI Transaction ID:.*', '', description).strip()
                    
                    if description and amount.group(1):
                        transactions.append({
                            'description': description,
                            'amount': amount.group(1),
                            'raw_line': line
                        })
                    
                    current_description = []
                else:
                    # This might be part of transaction description
                    if line and not re.search(r'^\d{2}:\d{2}', line):  # Not a time
                        current_description.append(line)
    
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
        "Food & Dining": ["restaurant", "food", "cafe", "pizza", "lunch", "dinner", "coffee"],
        "Transportation": ["uber", "taxi", "auto", "travel", "bus", "train", "gas", "petrol"],
        "Shopping": ["amazon", "store", "shop", "buy", "purchase", "mall", "flipkart"],
        "Bills & Utilities": ["bill", "electric", "water", "gas", "internet", "phone", "mobile"],
        "Healthcare": ["doctor", "hospital", "medicine", "pharmacy", "health", "clinic"],
        "Entertainment": ["movie", "cinema", "spotify", "netflix", "game", "ticket"],
        "Transfer": ["received", "transfer", "sent"],
    }
    
    categorized = []
    for trans in transactions:
        desc_lower = trans['description'].lower()
        category = "Other"
        
        # Match against keywords
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