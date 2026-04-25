const uploadBox  = document.getElementById('upload-box');
const fileInput  = document.getElementById('file-input');
const analyzeBtn = document.getElementById('analyze-btn');
const preview    = document.getElementById('upload-preview');
const content    = document.getElementById('upload-content');
const previewName = document.getElementById('preview-name');
const previewSize = document.getElementById('preview-size');
const loader     = document.getElementById('loader');
const form       = document.getElementById('upload-form');

if (uploadBox) {
  uploadBox.addEventListener('click', () => fileInput.click());

  uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
  });

  uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
  });

  uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
}

function handleFile(file) {
  const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
  previewName.textContent = file.name;
  previewSize.textContent = sizeMB + ' MB';
  content.style.display = 'none';
  preview.style.display = 'flex';
  analyzeBtn.disabled = false;

  // Transfer file to input
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;
}

if (form) {
  form.addEventListener('submit', () => {
    if (loader) {
      loader.style.display = 'flex';
      form.style.display   = 'none';
    }
  });
}

// Auto hide flash messages
setTimeout(() => {
  document.querySelectorAll('.flash').forEach(f => f.style.display = 'none');
}, 4000);