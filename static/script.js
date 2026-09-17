// const dropzone = document.getElementById('dropzone');
// const fileInput = document.getElementById('file-input');
// const emptyState = document.getElementById('empty-state');
// const previewImg = document.getElementById('preview-img');
// const removeBtn = document.getElementById('remove-btn');
// const predictBtn = document.getElementById('predict-btn');
// const btnSpinner = document.getElementById('btn-spinner');
// const btnLabel = document.getElementById('btn-label');
// const resultBox = document.getElementById('result');
// const verdictText = document.getElementById('verdict-text');
// const pctText = document.getElementById('pct-text');
// const gaugeFill = document.getElementById('gauge-fill');
// const resultNote = document.getElementById('result-note');
// const resetBtn = document.getElementById('reset-btn');
// const errorBox = document.getElementById('error-box');

// let selectedFile = null;

// dropzone.setAttribute('tabindex', '0');
// dropzone.setAttribute('role', 'button');
// dropzone.setAttribute('aria-label', 'Upload a patch image');

// dropzone.addEventListener('click', () => fileInput.click());
// dropzone.addEventListener('keydown', (e) => {
//   if (e.key === 'Enter' || e.key === ' ') {
//     e.preventDefault();
//     fileInput.click();
//   }
// });

// ['dragenter', 'dragover'].forEach((evt) => {
//   dropzone.addEventListener(evt, (e) => {
//     e.preventDefault();
//     dropzone.classList.add('is-dragover');
//   });
// });

// ['dragleave', 'drop'].forEach((evt) => {
//   dropzone.addEventListener(evt, (e) => {
//     e.preventDefault();
//     dropzone.classList.remove('is-dragover');
//   });
// });

// dropzone.addEventListener('drop', (e) => {
//   const file = e.dataTransfer.files[0];
//   if (file) handleFile(file);
// });

// fileInput.addEventListener('change', (e) => {
//   const file = e.target.files[0];
//   if (file) handleFile(file);
// });

// removeBtn.addEventListener('click', (e) => {
//   e.stopPropagation();
//   clearImage();
// });

// resetBtn.addEventListener('click', () => {
//   clearImage();
//   hideResult();
// });

// function handleFile(file) {
//   selectedFile = file;
//   hideError();
//   hideResult();

//   const url = URL.createObjectURL(file);
//   previewImg.src = url;
//   previewImg.hidden = false;
//   emptyState.hidden = true;
//   removeBtn.hidden = false;
//   predictBtn.disabled = false;
// }

// function clearImage() {
//   selectedFile = null;
//   fileInput.value = '';
//   previewImg.hidden = true;
//   previewImg.src = '';
//   removeBtn.hidden = true;
//   emptyState.hidden = false;
//   predictBtn.disabled = true;
//   hideError();
// }

// predictBtn.addEventListener('click', async () => {
//   if (!selectedFile) return;

//   setLoading(true);
//   hideError();
//   hideResult();

//   try {
//     const formData = new FormData();
//     formData.append('file', selectedFile);

//     const res = await fetch('/api/predict', {
//       method: 'POST',
//       body: formData,
//     });

//     const data = await res.json();

//     if (!res.ok) {
//       throw new Error(data.detail || 'Something went wrong.');
//     }

//     showResult(data);
//   } catch (err) {
//     showError(err.message);
//   } finally {
//     setLoading(false);
//   }
// });

// function setLoading(isLoading) {
//   predictBtn.disabled = isLoading;
//   btnSpinner.hidden = !isLoading;
//   btnLabel.textContent = isLoading ? 'Analyzing' : 'Analyze patch';
// }

// function showResult(data) {
//   const pct = (data.probability * 100).toFixed(1);
//   const isPositive = data.label === 1;

//   verdictText.textContent = isPositive
//     ? 'Flagged: IDC positive'
//     : 'Clear: IDC negative';
//   verdictText.className = 'result__verdict ' + (isPositive ? 'is-positive' : 'is-negative');

//   pctText.textContent = pct + '% probability';

//   resultNote.textContent = isPositive
//     ? 'Likely IDC positive.'
//     : 'Likely healthy tissue.';

//   resultBox.hidden = false;

//   // animate the gauge on next frame so the transition actually plays
//   gaugeFill.style.width = '0%';
//   requestAnimationFrame(() => {
//     requestAnimationFrame(() => {
//       gaugeFill.style.width = pct + '%';
//     });
//   });
// }

// function hideResult() {
//   resultBox.hidden = true;
//   gaugeFill.style.width = '0%';
// }

// function showError(message) {
//   errorBox.textContent = message;
//   errorBox.hidden = false;
// }

// function hideError() {
//   errorBox.hidden = true;
// }
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const emptyState = document.getElementById('empty-state');
const previewImg = document.getElementById('preview-img');
const removeBtn = document.getElementById('remove-btn');
const predictBtn = document.getElementById('predict-btn');
const btnSpinner = document.getElementById('btn-spinner');
const btnLabel = document.getElementById('btn-label');
const resultBox = document.getElementById('result');
const verdictText = document.getElementById('verdict-text');
const pctText = document.getElementById('pct-text');
const gaugeFill = document.getElementById('gauge-fill');
const resultNote = document.getElementById('result-note');
const resetBtn = document.getElementById('reset-btn');
const errorBox = document.getElementById('error-box');
const warningBanner = document.getElementById('warning-banner');
const warningText = document.getElementById('warning-text');

let selectedFile = null;

dropzone.setAttribute('tabindex', '0');
dropzone.setAttribute('role', 'button');
dropzone.setAttribute('aria-label', 'Upload a patch image');

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

['dragenter', 'dragover'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('is-dragover');
  });
});

['dragleave', 'drop'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('is-dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

removeBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearImage();
});

resetBtn.addEventListener('click', () => {
  clearImage();
  hideResult();
});

function handleFile(file) {
  selectedFile = file;
  hideError();
  hideResult();

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.hidden = false;
  emptyState.hidden = true;
  removeBtn.hidden = false;
  predictBtn.disabled = false;
}

function clearImage() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.hidden = true;
  previewImg.src = '';
  removeBtn.hidden = true;
  emptyState.hidden = false;
  predictBtn.disabled = true;
  hideError();
}

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  setLoading(true);
  hideError();
  hideResult();

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'Something went wrong.');
    }

    showResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  btnSpinner.hidden = !isLoading;
  btnLabel.textContent = isLoading ? 'Analyzing' : 'Analyze patch';
}

function showResult(data) {
  const pct = (data.probability * 100).toFixed(1);
  const isPositive = data.label === 1;

  if (data.warning) {
    warningText.textContent = data.warning;
    warningBanner.hidden = false;
  } else {
    warningBanner.hidden = true;
  }

  verdictText.textContent = isPositive
    ? 'Flagged: IDC positive'
    : 'Clear: IDC negative';
  verdictText.className = 'result__verdict ' + (isPositive ? 'is-positive' : 'is-negative');

  pctText.textContent = pct + '% probability';

  resultNote.textContent = isPositive
    ? 'Likely IDC positive.'
    : 'Likely healthy tissue.';

  resultBox.hidden = false;

  // animate the gauge on next frame so the transition actually plays
  gaugeFill.style.width = '0%';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      gaugeFill.style.width = pct + '%';
    });
  });
}

function hideResult() {
  resultBox.hidden = true;
  gaugeFill.style.width = '0%';
  warningBanner.hidden = true;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}
