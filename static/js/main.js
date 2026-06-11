const uploadArea = document.getElementById('uploadArea');
const uploadCard = document.getElementById('uploadCard');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImg = document.getElementById('previewImg');
const previewName = document.getElementById('previewName');
const btnReset = document.getElementById('btnReset');
const btnDetect = document.getElementById('btnDetect');
const btnTryAgain = document.getElementById('btnTryAgain');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultSection = document.getElementById('resultSection');

let selectedFile = null;

// DRAG & DROP 
uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));

uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    const allowed = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowed.includes(file.type)) {
        alert('Format tidak didukung. Gunakan JPG atau PNG.');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        previewName.textContent = file.name;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'flex';
        resultSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// RESET
btnReset.addEventListener('click', resetAll);
btnTryAgain.addEventListener('click', resetAll);

function resetAll() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    previewArea.style.display = 'none';
    uploadArea.style.display = 'flex';
    uploadCard.style.display = 'block';
    resultSection.style.display = 'none';
    loadingOverlay.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// DETECT
btnDetect.addEventListener('click', async () => {
    if (!selectedFile) return;

    previewArea.style.display = 'none';
    uploadCard.style.display = 'none';
    loadingOverlay.style.display = 'block';
    resultSection.style.display = 'none';

    animateLoadingSteps();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.error) {
            alert('Error: ' + data.error);
            loadingOverlay.style.display = 'none';
            uploadCard.style.display = 'block';
            previewArea.style.display = 'flex';
            return;
        }

        setTimeout(() => {
            loadingOverlay.style.display = 'none';
            renderResults(data);
            resultSection.style.display = 'block';
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 3200);

    } catch (err) {
        alert('Terjadi kesalahan. Pastikan server Flask berjalan.');
        loadingOverlay.style.display = 'none';
        uploadCard.style.display = 'block';
        previewArea.style.display = 'flex';
    }
});

function animateLoadingSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    steps.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('active');
    });
    document.getElementById('step1').classList.add('active');

    setTimeout(() => {
        document.getElementById('step1').classList.remove('active');
        document.getElementById('step2').classList.add('active');
    }, 900);

    setTimeout(() => {
        document.getElementById('step2').classList.remove('active');
        document.getElementById('step3').classList.add('active');
    }, 1800);

    setTimeout(() => {
        document.getElementById('step3').classList.remove('active');
        const s4 = document.getElementById('step4');
        if (s4) s4.classList.add('active');
    }, 2600);
}

// RENDER RESULTS
function renderResults(data) {
    // Gambar preprocessing
    document.getElementById('imgOriginal').src = 'data:image/jpeg;base64,' + data.images.original;
    document.getElementById('imgFiltered').src = 'data:image/jpeg;base64,' + data.images.filtered;
    document.getElementById('imgSegmented').src = 'data:image/jpeg;base64,' + data.images.segmented;

    // Fitur GLCM
    renderGLCM(data.glcm_features);

    // Kartu klasifikasi
    renderCard('knn', data.knn, data.class_info.knn);
    renderCard('nb', data.nb, data.class_info.nb);
}

// Render tabel fitur GLCM 
function renderGLCM(glcmFeatures) {
    const grid = document.getElementById('glcmGrid');
    if (!grid || !glcmFeatures) return;
    grid.innerHTML = '';

    // threshold: nilai di atas threshold dianggap "tinggi"
    const glcmInfo = {
        'Contrast': {
            short: 'Perbedaan intensitas antarpiksel yang berdekatan.',
            high: 'Tekstur kasar — kemungkinan terdapat bercak atau lesi',
            low: 'Tekstur halus — permukaan daun rata dan seragam',
            threshold: 10
        },
        'Dissimilarity': {
            short: 'Ketidakseragaman warna antarpiksel yang berdekatan.',
            high: 'Warna daun tidak merata — indikasi adanya penyakit',
            low: 'Warna daun merata — daun terlihat sehat',
            threshold: 2
        },
        'Homogeneity': {
            short: 'Keseragaman keseluruhan tekstur permukaan daun.',
            high: 'Tekstur seragam — daun cenderung sehat',
            low: 'Tekstur tidak merata — indikasi bercak atau lesi',
            threshold: 0.5
        },
        'Energy': {
            short: 'Keteraturan pola tekstur pada permukaan daun.',
            high: 'Pola tekstur teratur dan konsisten',
            low: 'Pola tekstur tidak beraturan — tekstur bervariasi',
            threshold: 0.05
        },
        'Correlation': {
            short: 'Konsistensi hubungan intensitas antarpiksel yang berdekatan.',
            high: 'Pola tekstur kuat dan berulang',
            low: 'Tidak ada pola dominan — tekstur acak',
            threshold: 0.9
        },
        'ASM': {
            short: 'Keseragaman distribusi intensitas piksel pada tekstur.',
            high: 'Distribusi intensitas seragam — sedikit variasi',
            low: 'Distribusi intensitas beragam — banyak variasi tekstur',
            threshold: 0.01
        }
    };

    Object.entries(glcmFeatures).forEach(([label, val]) => {
        const info = glcmInfo[label] || { short: '', high: '', low: '', threshold: 0.5 };
        const numVal = typeof val === 'number' ? val.toFixed(6) : val;
        const numRaw = typeof val === 'number' ? val : parseFloat(val);

        const isHigh = numRaw >= info.threshold;
        const interpHTML = isHigh
            ? `<span class="glcm-tag high">↑ Tinggi: ${info.high}</span>`
            : `<span class="glcm-tag low">↓ Rendah: ${info.low}</span>`;

        const item = document.createElement('div');
        item.className = 'glcm-item';
        item.innerHTML = `
            <div class="glcm-item-label">${label}</div>
            <div class="glcm-item-short">${info.short}</div>
            <div class="glcm-item-val">${numVal}</div>
            <div class="glcm-item-interp">
                ${interpHTML}
            </div>
        `;
        grid.appendChild(item);
    });
}

// kartu hasil klasifikasi
function renderCard(prefix, result, info) {
    const card = document.getElementById(prefix + 'Card');

    const classMap = { 'Healthy': 'healthy', 'Leaf Spot': 'leafspot', 'Powdery Mildew': 'mildew' };
    card.className = 'result-card ' + (classMap[result.label] || '');

    document.getElementById(prefix + 'Label').textContent = result.label;
    document.getElementById(prefix + 'Icon').textContent = info.icon;
    document.getElementById(prefix + 'Conf').textContent = result.confidence + '%';
    document.getElementById(prefix + 'Desc').textContent = info.desc;
    document.getElementById(prefix + 'Rec').textContent = '💡 ' + info.recommendation;

    setTimeout(() => {
        document.getElementById(prefix + 'Bar').style.width = result.confidence + '%';
    }, 100);

    const breakdown = document.getElementById(prefix + 'Breakdown');
    breakdown.innerHTML = '';
    Object.entries(result.all_probs).forEach(([cls, prob]) => {
        const row = document.createElement('div');
        row.className = 'prob-row';
        const isTop = cls === result.label;
        row.innerHTML = `
            <span class="prob-name">${cls}</span>
            <span class="prob-val ${isTop ? 'highlight' : ''}">${prob}%</span>
        `;
        breakdown.appendChild(row);
    });
}