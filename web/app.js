const MAX_FILE_SIZE = 5 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
const ALLOWED_EXTENSIONS = /\.(png|jpe?g|webp)$/i;

// Keep all DOM lookups together so the UI dependencies are easy to audit.
const elements = {
  uploadArea: document.querySelector("#upload-area"),
  fileInput: document.querySelector("#file-input"),
  errorMessage: document.querySelector("#error-message"),
  resultPanel: document.querySelector("#result-panel"),
  previewImage: document.querySelector("#preview-image"),
  loadingState: document.querySelector("#loading-state"),
  caption: document.querySelector("#caption"),
  retryButton: document.querySelector("#retry-button"),
  chooseButton: document.querySelector("#choose-button"),
};

let lastResult = null;
let selectedFile = null;
let previewUrl = null;
let activeController = null;

function showError(message) {
  elements.errorMessage.textContent = message;
  elements.errorMessage.hidden = false;
}

function clearError() {
  elements.errorMessage.textContent = "";
  elements.errorMessage.hidden = true;
}

function formatCaption(text) {
  const caption = String(text || "").trim();
  if (!caption) {
    return "No caption was generated.";
  }
  const capitalized = caption.charAt(0).toUpperCase() + caption.slice(1);
  return /[.!?]$/.test(capitalized) ? capitalized : `${capitalized}.`;
}

function isAllowedFile(file) {
  return ALLOWED_TYPES.has(file.type) || (
    !file.type && ALLOWED_EXTENSIONS.test(file.name)
  );
}

function setLoading(loading) {
  elements.loadingState.hidden = !loading;
  elements.retryButton.hidden = loading || !selectedFile;
  elements.chooseButton.hidden = loading && !selectedFile;
  if (loading) {
    elements.caption.textContent = "";
  }
}

function resetPage() {
  if (activeController) {
    activeController.abort();
    activeController = null;
  }
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  selectedFile = null;
  lastResult = null;
  elements.fileInput.value = "";
  elements.previewImage.removeAttribute("src");
  elements.resultPanel.hidden = true;
  elements.loadingState.hidden = true;
  elements.caption.textContent = "";
  elements.retryButton.hidden = true;
  elements.chooseButton.hidden = true;
  clearError();
}

async function readError(response) {
  try {
    const body = await response.json();
    return body.detail || body.error || body.message || `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}

async function generateCaption(file) {
  if (activeController) {
    activeController.abort();
  }
  activeController = new AbortController();
  const timeoutId = window.setTimeout(() => activeController.abort(), 60000);
  const formData = new FormData();
  formData.append("file", file);

  try {
    const apiBase = window.APP_CONFIG?.API_BASE || "";
    const response = await fetch(`${apiBase}/api/caption`, {
      method: "POST",
      body: formData,
      signal: activeController.signal,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    lastResult = await response.json();
    elements.caption.textContent = formatCaption(lastResult.caption);
    elements.retryButton.hidden = true;
  } catch (error) {
    if (error.name === "AbortError") {
      showError("The request took too long. Please try again.");
    } else {
      showError(error.message || "We could not generate a caption. Please try again.");
    }
    elements.retryButton.hidden = false;
  } finally {
    window.clearTimeout(timeoutId);
    activeController = null;
    elements.loadingState.hidden = true;
  }
}

function handleFile(file) {
  clearError();
  if (!file) {
    return;
  }
  if (!isAllowedFile(file)) {
    showError("Please choose a PNG, JPEG, or WEBP image.");
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    showError("That image is too large. Please choose a file no bigger than 5 MB.");
    return;
  }

  selectedFile = file;
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  elements.previewImage.src = previewUrl;
  elements.resultPanel.hidden = false;
  elements.retryButton.hidden = true;
  elements.chooseButton.hidden = false;
  setLoading(true);
  generateCaption(file);
}

elements.fileInput.addEventListener("change", (event) => {
  handleFile(event.target.files[0]);
});

elements.uploadArea.addEventListener("dragenter", (event) => {
  event.preventDefault();
  elements.uploadArea.classList.add("drag-active");
});

elements.uploadArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.uploadArea.classList.add("drag-active");
});

elements.uploadArea.addEventListener("dragleave", (event) => {
  if (!elements.uploadArea.contains(event.relatedTarget)) {
    elements.uploadArea.classList.remove("drag-active");
  }
});

elements.uploadArea.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.uploadArea.classList.remove("drag-active");
  handleFile(event.dataTransfer.files[0]);
});

elements.retryButton.addEventListener("click", () => {
  if (selectedFile) {
    clearError();
    setLoading(true);
    generateCaption(selectedFile);
  }
});

elements.chooseButton.addEventListener("click", () => {
  resetPage();
  elements.fileInput.click();
});
