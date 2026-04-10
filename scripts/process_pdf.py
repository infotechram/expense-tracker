import sys
import json
import os
from datetime import datetime
import pdfplumber
import re

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
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            for line in text.split('\n'):
                amount = re.search(r'₹\s*([\d,]+(?:\.\d{2})?)', line)
                if amount and len(line.strip()) > 5:
                    transactions.append({
                        'description': line.strip(),
                        'amount': amount.group(1)
                    })
    
    return transactions

def categorize(transactions):
    """Categorize using zero-shot AI"""
    categorized = []
    
    if HF_AVAILABLE:
        try:
            print("📥 Loading model from cache...")
            classifier = pipeline("zero-shot-classification", 
                                model="facebook/bart-large-mnli", device=-1)
            
            print("✅ Model loaded successfully!")
            show_cache_info()
            
            for trans in transactions:
                result = classifier(trans['description'], 
                                  ["expense"], multi_class=False)
                
                categorized.append({
                    'description': trans['description'],
                    'amount': trans['amount'],
                    'category': 'Expense',
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
    """Simple fallback"""
    return [
        {
            'description': t['description'],
            'amount': t['amount'],
            'category': 'Uncategorized',
            'confidence': 'Manual',
            'user_editable': True
        }
        for t in transactions
    ]

def main(pdf_path):
    print(f"Processing: {pdf_path}")
    
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
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "pdf_file": os.path.basename(pdf_path),
        "cache_location": os.path.expanduser("~/.cache/huggingface/hub/"),
        "transactions": categorized,
        "summary": {
            "total_spent": total,
            "by_category": summary,
            "transaction_count": len(categorized)
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/expenses.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved to results/expenses.json")

if __name__ == "__main__":
    main(sys.argv[1])