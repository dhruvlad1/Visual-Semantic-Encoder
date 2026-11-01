const MAX_FILE_SIZE = 5 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
const ALLOWED_EXTENSIONS = /\.(png|jpe?g|webp)$/i;

// Keep all DOM lookups together so the UI dependencies are easy to audit.
const elements = {
  uploadState: document.querySelector("#upload-state"),
  uploadArea: document.querySelector("#upload-area"),
  fileInput: document.querySelector("#file-input"),
  uploadError: document.querySelector("#upload-error"),
  resultState: document.querySelector("#result-state"),
  previewImage: document.querySelector("#preview-image"),
  loadingState: document.querySelector("#loading-state"),
  caption: document.querySelector("#caption"),
  resultError: document.querySelector("#result-error"),
  retryButton: document.querySelector("#retry-button"),
  chooseButton: document.querySelector("#choose-button"),
};

let lastResult = null;
let selectedFile = null;
let previewUrl = null;
let activeController = null;
let activeTimeoutId = null;

function showError(message, target) {
  target.textContent = message;
  target.hidden = false;
}

function clearError(target) {
  target.textContent = "";
  target.hidden = true;
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
  elements.retryButton.hidden = loading;
  if (loading) {
    elements.caption.textContent = "";
    clearError(elements.resultError);
  }
}

function resetPage() {
  if (activeController) {
    activeController.abort();
    activeController = null;
  }
  if (activeTimeoutId !== null) {
    window.clearTimeout(activeTimeoutId);
    activeTimeoutId = null;
  }
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  selectedFile = null;
  lastResult = null;
  elements.fileInput.value = "";
  elements.previewImage.removeAttribute("src");
  elements.uploadState.hidden = false;
  elements.resultState.hidden = true;
  elements.loadingState.hidden = true;
  elements.caption.textContent = "";
  elements.retryButton.hidden = true;
  clearError(elements.uploadError);
  clearError(elements.resultError);
  elements.uploadArea.focus();
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
  if (activeTimeoutId !== null) {
    window.clearTimeout(activeTimeoutId);
  }
  const controller = new AbortController();
  activeController = controller;
  const timeoutId = window.setTimeout(() => controller.abort(), 60000);
  activeTimeoutId = timeoutId;
  const formData = new FormData();
  formData.append("file", file);

  try {
    const apiBase = window.APP_CONFIG?.API_BASE || "";
    const response = await fetch(`${apiBase}/api/caption`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    lastResult = await response.json();
    elements.caption.textContent = formatCaption(lastResult.caption);
    elements.retryButton.hidden = true;
  } catch (error) {
    if (activeController !== controller) {
      return;
    }
    if (error.name === "AbortError") {
      showError("The request took too long. Please try again.", elements.resultError);
    } else {
      showError(
        error.message || "We could not generate a caption. Please try again.",
        elements.resultError,
      );
    }
    elements.retryButton.hidden = false;
  } finally {
    window.clearTimeout(timeoutId);
    if (activeController === controller) {
      activeController = null;
      activeTimeoutId = null;
    }
    elements.loadingState.hidden = true;
  }
}

function handleFile(file) {
  clearError(elements.uploadError);
  if (!file) {
    return;
  }
  if (!isAllowedFile(file)) {
    showError("Please choose a PNG, JPEG, or WEBP image.", elements.uploadError);
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    showError(
      "That image is too large. Please choose a file no bigger than 5 MB.",
      elements.uploadError,
    );
    return;
  }

  selectedFile = file;
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  elements.previewImage.src = previewUrl;
  elements.uploadState.hidden = true;
  elements.resultState.hidden = false;
  elements.retryButton.hidden = true;
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

elements.uploadArea.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.fileInput.click();
  }
});

elements.retryButton.addEventListener("click", () => {
  if (selectedFile) {
    clearError(elements.resultError);
    setLoading(true);
    generateCaption(selectedFile);
  }
});

elements.chooseButton.addEventListener("click", () => {
  resetPage();
});

window.addEventListener("dragover", (event) => {
  event.preventDefault();
});

window.addEventListener("drop", (event) => {
  event.preventDefault();
});
