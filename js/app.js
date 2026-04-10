const GITHUB_REPO = 'infotechram/expense-tracker';
const RAW_URL = `https://raw.githubusercontent.com/${GITHUB_REPO}/main`;

let results = null;
let edits = {};
let uploadedFileName = null;
let chartInstance = null;

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// ─── Upload ───────────────────────────────────────────────────────────────────

async function upload() {
    const file = document.getElementById('fileInput').files[0];
    const token = document.getElementById('tokenInput').value;

    if (!file || !token) {
        alert('Please select file and enter token');
        return;
    }

    const uniqueId = `${Date.now()}-${generateUUID()}`;
    uploadedFileName = uniqueId;

    const name = `uploads/${uniqueId}.pdf`;
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
// ─── Polling ──────────────────────────────────────────────────────────────────

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

// ─── Results ──────────────────────────────────────────────────────────────────

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

// ─── Chart ────────────────────────────────────────────────────────────────────

function refreshChart() {
    const groupBy = document.getElementById('groupBy').value;

    document.getElementById('dayBreakdown').style.display = 'none';
    document.getElementById('viewAllBtn').style.display = 'none';
    fillTable(results.transactions);

    if (groupBy === 'day_of_week') {
        drawChart(results.summary.by_day_of_week);
    } else {
        drawChart(results.summary.by_category);
    }
}

function drawChart(data) {
    const canvas = document.getElementById('chart');
    const ctx = canvas.getContext('2d');

    if (chartInstance) {
        chartInstance.destroy();
    }

    if (canvas._clickHandler) {
        canvas.removeEventListener('click', canvas._clickHandler);
    }

    chartInstance = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: Object.keys(data),
            datasets: [{
                data: Object.values(data),
                backgroundColor: [
                    '#FF6384', '#36A2EB', '#FFCE56',
                    '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF'
                ]
            }]
        },
        options: {
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            return ` ${label}: \u20B9${value.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });

    canvas._clickHandler = function(event) {
        const elements = chartInstance.getElementsAtEventForMode(
            event, 'nearest', { intersect: true }, false
        );
        if (elements.length === 0) return;

        const index = elements[0].index;
        const label = Object.keys(data)[index];
        const groupBy = document.getElementById('groupBy').value;

        if (groupBy === 'day_of_week') {
            showDayTransactions(label);
        } else {
            showCategoryTransactions(label);
        }
    };

    canvas.addEventListener('click', canvas._clickHandler);
}

// ─── Filters ──────────────────────────────────────────────────────────────────

function showDayTransactions(day) {
    const filtered = results.transactions.filter(t => t.day_of_week === day);
    document.getElementById('dayBreakdown').textContent =
        `Showing ${filtered.length} transactions for ${day}`;
    document.getElementById('dayBreakdown').style.display = 'inline';
    document.getElementById('viewAllBtn').style.display = 'inline-block';
    fillTable(filtered);
}

function showCategoryTransactions(category) {
    const filtered = results.transactions.filter(t => t.category === category);
    document.getElementById('dayBreakdown').textContent =
        `Showing ${filtered.length} transactions for ${category}`;
    document.getElementById('dayBreakdown').style.display = 'inline';
    document.getElementById('viewAllBtn').style.display = 'inline-block';
    fillTable(filtered);
}

function viewAll() {
    document.getElementById('dayBreakdown').style.display = 'none';
    document.getElementById('viewAllBtn').style.display = 'none';
    fillTable(results.transactions);
}

// ─── Table ────────────────────────────────────────────────────────────────────

function esc(str) {
    return String(str || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function fillTable(transactions) {
    const table = document.getElementById('table');

    while (table.rows.length > 1) {
        table.deleteRow(1);
    }

    transactions.forEach((t, i) => {
        const row = table.insertRow();
        row.innerHTML = `
            <td>${i + 1}</td>
            <td>${esc(t.description)}</td>
            <td>&#8377;${esc(t.amount)}</td>
            <td>
                <input type="text" value="${esc(t.category)}"
                    onchange="edits[${i}]=this.value" style="width:120px;">
            </td>
            <td>${esc(t.date) || '-'}</td>
            <td>${esc(t.day_of_week) || '-'}</td>
        `;
    });
}

// ─── Save & Download ──────────────────────────────────────────────────────────

function save() {
    Object.keys(edits).forEach(i => {
        results.transactions[i].category = edits[i];
    });

    const summary = {};
    let total = 0;
    results.transactions.forEach(t => {
        const amt = parseFloat(String(t.amount).replace(',', ''));
        summary[t.category] = (summary[t.category] || 0) + amt;
        total += amt;
    });

    results.summary = {
        total_spent: total,
        by_category: summary,
        transaction_count: results.transactions.length
    };

    alert('Changes saved locally');
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