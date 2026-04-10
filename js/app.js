const GITHUB_REPO = 'infotechram/expense-tracker';
const RAW_URL = `https://raw.githubusercontent.com/${GITHUB_REPO}/main`;

let results = null;
let edits = {};
let uploadedFileName = null;

async function upload() {
    const file = document.getElementById('fileInput').files[0];
    const token = document.getElementById('tokenInput').value;
    
    if (!file || !token) {
        alert('Please select file and enter token');
        return;
    }

    // Generate a timestamp-based filename (same as what we'll upload)
    const timestamp = Date.now();
    uploadedFileName = String(timestamp);

    const name = `uploads/${timestamp}.pdf`;
    const reader = new FileReader();
    
    reader.onload = async (e) => {
        const base64 = e.target.result.split(',')[1];
        
        const res = await fetch(
            `https://api.github.com/repos/${GITHUB_REPO}/contents/${name}`,
            {
                method: 'PUT',
                headers: {
                    'Authorization': `token ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: `Upload PDF: ${file.name}`,
                    content: base64
                })
            }
        );

        if (res.ok) {
            document.getElementById('uploadSection').style.display = 'none';
            document.getElementById('processingSection').style.display = 'block';
            pollResults();
        } else {
            alert('Upload failed');
        }
    };
    
    reader.readAsDataURL(file);
}

async function pollResults() {
    const expectedFileName = `${uploadedFileName}_expenses.json`;
    
    for (let i = 0; i < 120; i++) {
        const res = await fetch(`${RAW_URL}/results/${expectedFileName}?t=${Date.now()}`);
        
        if (res.ok) {
            results = await res.json();
            showResults();
            return;
        }
        
        await new Promise(r => setTimeout(r, 3000));
    }
    
    alert('Timeout: Could not find processed file or processing took too long');
}

function showResults() {
    document.getElementById('processingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'block';
    
    const summary = results.summary;
    document.getElementById('totalSpent').textContent = summary.total_spent.toFixed(2);
    document.getElementById('totalTrans').textContent = summary.transaction_count;
    document.getElementById('totalCats').textContent = Object.keys(summary.by_category).length;
    
    drawChart(summary.by_category);
    fillTable(results.transactions);
}

function drawChart(data) {
    const ctx = document.getElementById('chart').getContext('2d');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: Object.keys(data),
            datasets: [{
                data: Object.values(data),
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
            }]
        }
    });
}

function fillTable(transactions) {
    const table = document.getElementById('table');
    
    transactions.forEach((t, i) => {
        const row = table.insertRow();
        row.innerHTML = `
            <td>${i + 1}</td>
            <td>${t.description}</td>
            <td>₹${t.amount}</td>
            <td><input type="text" value="${t.category}" 
                onchange="edits[${i}]=this.value" style="width:100px;"></td>
        `;
    });
}

function save() {
    Object.keys(edits).forEach(i => {
        results.transactions[i].category = edits[i];
    });
    
    const summary = {};
    let total = 0;
    results.transactions.forEach(t => {
        const amt = parseFloat(t.amount);
        summary[t.category] = (summary[t.category] || 0) + amt;
        total += amt;
    });
    
    results.summary = {
        total_spent: total,
        by_category: summary,
        transaction_count: results.transactions.length
    };
    
    alert('✅ Changes saved locally');
    location.reload();
}

function download() {
    const data = JSON.stringify(results, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${uploadedFileName}_expenses.json`;
    a.click();
}