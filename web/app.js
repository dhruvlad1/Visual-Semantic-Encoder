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
  heatmapStack: document.querySelector("#heatmap-stack"),
  heatmapBase: document.querySelector("#heatmap-base"),
  heatmapOverlay: document.querySelector("#heatmap-overlay"),
  loadingState: document.querySelector("#loading-state"),
  caption: document.querySelector("#caption"),
  resultError: document.querySelector("#result-error"),
  heatmapControls: document.querySelector("#heatmap-controls"),
  heatmapToggle: document.querySelector("#heatmap-toggle"),
  wordChips: document.querySelector("#word-chips"),
  selectedWord: document.querySelector("#selected-word"),
  strengthValue: document.querySelector("#strength-value"),
  heatmapStrength: document.querySelector("#heatmap-strength"),
  downloadButton: document.querySelector("#download-button"),
  heatmapGrid: document.querySelector("#heatmap-grid"),
  retryButton: document.querySelector("#retry-button"),
  chooseButton: document.querySelector("#choose-button"),
};

let lastResult = null;
let selectedFile = null;
let previewUrl = null;
let activeController = null;
let activeTimeoutId = null;
let heatmapMode = false;
let selectedIndex = 0;

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

function hasHeatmaps() {
  return Boolean(
    lastResult
    && Array.isArray(lastResult.words)
    && lastResult.words.length > 0
    && Array.isArray(lastResult.heatmaps)
    && lastResult.heatmaps.length === lastResult.words.length,
  );
}

function renderChips() {
  elements.wordChips.replaceChildren();
  if (!hasHeatmaps()) {
    return;
  }
  lastResult.words.forEach((word, index) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = word;
    chip.setAttribute("aria-pressed", String(index === selectedIndex));
    chip.addEventListener("click", () => selectWord(index));
    elements.wordChips.append(chip);
  });
}

function renderGrid() {
  elements.heatmapGrid.replaceChildren();
  if (!hasHeatmaps()) {
    elements.heatmapGrid.hidden = true;
    return;
  }
  lastResult.words.forEach((word, index) => {
    const item = document.createElement("button");
    const image = document.createElement("img");
    const label = document.createElement("span");
    item.type = "button";
    item.className = "grid-item";
    item.dataset.index = String(index);
    item.addEventListener("click", () => selectWord(index));
    image.src = lastResult.heatmaps[index];
    image.alt = `Attention heatmap for the word "${word}"`;
    label.textContent = word;
    item.append(image, label);
    elements.heatmapGrid.append(item);
  });
  elements.heatmapGrid.hidden = !heatmapMode;
}

function selectWord(index) {
  if (!hasHeatmaps()) {
    return;
  }
  selectedIndex = Math.max(0, Math.min(index, lastResult.words.length - 1));
  const word = lastResult.words[selectedIndex];
  elements.wordChips.querySelectorAll(".chip").forEach((chip, chipIndex) => {
    chip.setAttribute("aria-pressed", String(chipIndex === selectedIndex));
  });
  elements.heatmapGrid.querySelectorAll(".grid-item").forEach((item, itemIndex) => {
    item.classList.toggle("selected", itemIndex === selectedIndex);
  });
  elements.heatmapBase.src = lastResult.image;
  elements.heatmapOverlay.src = lastResult.heatmaps[selectedIndex];
  elements.heatmapOverlay.alt = `Attention heatmap for the word "${word}"`;
  elements.selectedWord.textContent = word;
  elements.downloadButton.disabled = false;
}

function setHeatmapMode(on) {
  heatmapMode = on && hasHeatmaps();
  elements.heatmapToggle.textContent = heatmapMode ? "Hide heatmap" : "Show heatmap";
  elements.heatmapToggle.setAttribute("aria-pressed", String(heatmapMode));
  elements.heatmapStack.hidden = !heatmapMode;
  elements.previewImage.hidden = heatmapMode;
  elements.wordChips.hidden = !heatmapMode;
  elements.strengthValue.textContent = `${elements.heatmapStrength.value}%`;
  elements.selectedWord.textContent = heatmapMode ? lastResult.words[selectedIndex] : "-";
  elements.downloadButton.hidden = !heatmapMode;
  elements.heatmapGrid.hidden = !heatmapMode;
  if (heatmapMode) {
    selectWord(selectedIndex);
  }
}

function resetHeatmapState() {
  heatmapMode = false;
  selectedIndex = 0;
  elements.heatmapStrength.value = "100";
  elements.heatmapOverlay.style.opacity = "1";
  elements.heatmapControls.hidden = true;
  elements.heatmapStack.hidden = true;
  elements.previewImage.hidden = false;
  elements.wordChips.replaceChildren();
  elements.heatmapGrid.replaceChildren();
  elements.heatmapGrid.hidden = true;
  elements.heatmapToggle.textContent = "Show heatmap";
  elements.heatmapToggle.setAttribute("aria-pressed", "false");
  elements.selectedWord.textContent = "-";
  elements.strengthValue.textContent = "100%";
  elements.downloadButton.hidden = true;
}

function downloadSelected() {
  if (!hasHeatmaps()) {
    return;
  }
  const word = String(lastResult.words[selectedIndex]).replace(/[\\/:*?"<>|]+/g, "_") || "word";
  const link = document.createElement("a");
  link.href = lastResult.heatmaps[selectedIndex];
  link.download = `heatmap-${word}.png`;
  link.click();
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
  resetHeatmapState();
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
    const result = await response.json();
    if (activeController !== controller) {
      return;
    }
    lastResult = result;
    elements.caption.textContent = formatCaption(lastResult.caption);
    elements.retryButton.hidden = true;
    resetHeatmapState();
    if (hasHeatmaps()) {
      renderChips();
      renderGrid();
      elements.heatmapControls.hidden = false;
    }
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
      elements.loadingState.hidden = true;
    }
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
  resetHeatmapState();
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

elements.heatmapToggle.addEventListener("click", () => {
  setHeatmapMode(!heatmapMode);
});

elements.heatmapStrength.addEventListener("input", () => {
  const value = Number(elements.heatmapStrength.value);
  elements.heatmapOverlay.style.opacity = String(value / 100);
  elements.strengthValue.textContent = `${value}%`;
});

elements.wordChips.addEventListener("keydown", (event) => {
  if (!hasHeatmaps() || !["ArrowLeft", "ArrowRight"].includes(event.key)) {
    return;
  }
  event.preventDefault();
  const nextIndex = event.key === "ArrowRight"
    ? selectedIndex + 1
    : selectedIndex - 1;
  const wrappedIndex = (nextIndex + lastResult.words.length) % lastResult.words.length;
  selectWord(wrappedIndex);
  elements.wordChips.children[wrappedIndex].focus();
});

elements.downloadButton.addEventListener("click", downloadSelected);

elements.chooseButton.addEventListener("click", () => {
  resetPage();
});

window.addEventListener("dragover", (event) => {
  event.preventDefault();
});

window.addEventListener("drop", (event) => {
  event.preventDefault();
});
