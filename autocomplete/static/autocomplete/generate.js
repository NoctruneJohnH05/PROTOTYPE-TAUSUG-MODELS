const docForm = document.getElementById("doc-form")
const textArea = document.getElementById("document")
const selectPred = document.getElementById("select-pred")
const predictionOverlay = document.getElementById("prediction-overlay")
const autocompleteInline = document.getElementById("autocomplete-inline")
const wordCountElem = document.getElementById("word-count")
const predictionStatusElem = document.getElementById("prediction-status")
const modelSelector = document.getElementById("model-selector")
const modelNameElem = document.getElementById("model-name")

// New elements for generation mode
const modeSelector = document.getElementById("mode-selector")
const generationControls = document.getElementById("generation-controls")
const lengthSlider = document.getElementById("length-slider")
const lengthValue = document.getElementById("length-value")
const tempSlider = document.getElementById("temp-slider")
const tempValue = document.getElementById("temp-value")
const generateBtn = document.getElementById("generate-btn")
const generatedOverlay = document.getElementById("generated-overlay")
const generatedText = document.getElementById("generated-text")
const insertGeneratedBtn = document.getElementById("insert-generated")

// Feature visibility elements
const predictFeature = document.getElementById("predict-feature")
const predictFeature2 = document.getElementById("predict-feature-2")
const generateFeature = document.getElementById("generate-feature")
const predictExample = document.getElementById("predict-example")
const generateExample = document.getElementById("generate-example")

// Debug: Verify model selector is found
console.log("%c[INIT] Model Selector Element:", "color: orange; font-weight: bold", modelSelector);
console.log("%c[INIT] Initial Model Value:", "color: orange; font-weight: bold", modelSelector ? modelSelector.value : "NOT FOUND");

// ==================== API CONFIGURATION ====================
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8000'
    : window.location.origin; // Uses the same domain as frontend

console.log(`%c[API] Base URL: ${API_BASE_URL}`, 'background: #10b981; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold');

// ==================== OPTIMIZATION: Request Management ====================
let typingTimer;
const debounceDelay = 500; // Increased from 300ms to 500ms for better performance
let abortController = null; // For canceling previous requests

// ==================== OPTIMIZATION: Local Prediction Cache ====================
const predictionCache = new Map();
const MAX_CACHE_SIZE = 100; // Limit cache size to prevent memory issues

function getCachedPrediction(text, model) {
    const cacheKey = `${model}:${text}`;
    return predictionCache.get(cacheKey);
}

function setCachedPrediction(text, model, prediction) {
    const cacheKey = `${model}:${text}`;
    
    // Implement LRU cache: remove oldest entry if cache is full
    if (predictionCache.size >= MAX_CACHE_SIZE) {
        const firstKey = predictionCache.keys().next().value;
        predictionCache.delete(firstKey);
    }
    
    predictionCache.set(cacheKey, prediction);
    console.log(`%c[CACHE] Stored prediction for: "${text.substring(0, 30)}..." (Total cached: ${predictionCache.size})`, 'color: #8b5cf6; font-size: 11px');
}

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
const csrftoken = getCookie('csrftoken');

// Update word count
function updateWordCount() {
    const text = textArea.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    wordCountElem.textContent = `Words: ${words}`;
}

// Update autocomplete inline display
function updateAutocompleteDisplay(suggestion) {
    if (!suggestion) {
        autocompleteInline.textContent = "";
        autocompleteInline.style.display = "none";
        return;
    }
    
    const cursorPos = textArea.selectionStart;
    const textBeforeCursor = textArea.value.substring(0, cursorPos);
    const words = textBeforeCursor.split(/\s+/);
    const currentWord = words[words.length - 1];
    
    // Show ONLY the completion part (what needs to be added)
    const completion = suggestion.startsWith(currentWord.toLowerCase()) 
        ? suggestion.substring(currentWord.length)
        : suggestion;
    
    // Only show if there's actual completion to display
    if (completion) {
        autocompleteInline.textContent = completion;
        autocompleteInline.style.display = "block";
        
        // Position the autocomplete text right after the cursor
        const textLines = textBeforeCursor.split('\n');
        const currentLine = textLines[textLines.length - 1];
        
        // Create a temporary span to measure text width
        const measureSpan = document.createElement('span');
        measureSpan.style.font = window.getComputedStyle(textArea).font;
        measureSpan.style.visibility = 'hidden';
        measureSpan.style.position = 'absolute';
        measureSpan.textContent = currentLine;
        document.body.appendChild(measureSpan);
        
        const textWidth = measureSpan.offsetWidth;
        document.body.removeChild(measureSpan);
        
        autocompleteInline.style.left = `${20 + textWidth}px`;
        autocompleteInline.style.top = '20px';
    } else {
        autocompleteInline.textContent = "";
        autocompleteInline.style.display = "none";
    }
}

// Text input event listener - BOTH features work simultaneously
textArea.addEventListener("input", async (e) => {
    updateWordCount();
    
    // Check if we're in generation mode - if so, skip prediction
    if (modeSelector.value === "generate") {
        return;
    }
    
    // Clear previous timer
    clearTimeout(typingTimer);
    
    // Cancel any pending request
    if (abortController) {
        abortController.abort();
        console.log('%c[REQUEST] Previous request cancelled', 'color: #f59e0b; font-size: 11px');
    }
    
    // Clear previous suggestions
    selectPred.innerHTML = "";
    autocompleteInline.textContent = "";
    
    const text = textArea.value;
    if (!text || !text.trim()) {
        predictionStatusElem.textContent = "Ready";
        predictionOverlay.style.display = "none";
        return;
    }
    
    // Get context for next-word prediction (last 8 complete words)
    const words = text.trim().split(/\s+/);
    const textWindow = words.slice(-8).join(" ");
    const cleanedText = textWindow.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()']/g, "");
    
    if (!cleanedText) {
        predictionStatusElem.textContent = "Ready";
        return;
    }
    
    // Set debounced prediction request
    typingTimer = setTimeout(() => fetchPrediction(cleanedText), debounceDelay);
});

// Separated fetch logic for cleaner code
async function fetchPrediction(cleanedText) {
    const selectedModel = modelSelector.value;
    
    // Check cache first
    const cachedResult = getCachedPrediction(cleanedText, selectedModel);
    if (cachedResult) {
        console.log(`%c[CACHE HIT] Using cached prediction for: "${cleanedText.substring(0, 30)}..."`, 'color: #10b981; font-weight: bold; font-size: 12px');
        displayPredictions(cachedResult);
        return;
    }
    
    predictionStatusElem.textContent = "🤖 AI processing...";
    predictionStatusElem.style.color = "#f59e0b";
    
    try {
        // Create new AbortController for this request
        abortController = new AbortController();
        
        console.log(`%c[MODEL] Using: ${selectedModel.toUpperCase()}`, 'color: #667eea; font-weight: bold; font-size: 14px');
        console.log(`[INPUT] Text: "${cleanedText}"`);
        
        const response = await fetch(`${API_BASE_URL}/document/`, {
            method: "POST",
            headers: {
                'X-CSRFToken': csrftoken,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: cleanedText,
                model: selectedModel
            }),
            mode: 'same-origin',
            signal: abortController.signal // Enable request cancellation
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        console.log(`%c[RESPONSE] Model used by backend: ${data.model_used ? data.model_used.toUpperCase() : 'UNKNOWN'}`, 'color: #10b981; font-weight: bold');
        console.log(`[PREDICTIONS] ${data.preds}`);
        
        // Cache the result
        if (data.preds && data.preds.trim()) {
            setCachedPrediction(cleanedText, selectedModel, data);
            displayPredictions(data);
        } else {
            predictionOverlay.style.display = "none";
            predictionStatusElem.textContent = "No predictions available";
            predictionStatusElem.style.color = "#718096";
        }
        
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('%c[REQUEST] Request aborted (newer request started)', 'color: #f59e0b; font-size: 11px');
            return; // Don't show error for aborted requests
        }
        
        console.error("Prediction error:", error);
        predictionStatusElem.textContent = "❌ Error getting predictions";
        predictionStatusElem.style.color = "#ef4444";
        
        setTimeout(() => {
            predictionStatusElem.textContent = "Ready";
            predictionStatusElem.style.color = "#667eea";
        }, 3000);
    } finally {
        abortController = null;
    }
}

// Separated display logic for reuse with cache
function displayPredictions(data) {
    predictionOverlay.style.display = "block";
    const predictions = data.preds.split(" ").filter(word => word.trim());
    
    selectPred.innerHTML = "";
    
    // Add model indicator
    const modelIndicator = document.createElement("div");
    modelIndicator.style.fontSize = "11px";
    modelIndicator.style.color = "#667eea";
    modelIndicator.style.fontWeight = "600";
    modelIndicator.style.marginBottom = "8px";
    modelIndicator.style.textAlign = "center";
    modelIndicator.textContent = `Generated by: ${data.model_used ? data.model_used.toUpperCase() : 'LSTM'}`;
    selectPred.appendChild(modelIndicator);
    
    // Show as NEXT-WORD PREDICTION BUTTONS (numbered 1, 2, 3)
    predictions.forEach((word, index) => {
        const predictedWordDiv = document.createElement("div");
        predictedWordDiv.textContent = `${index + 1}. ${word}`;
        
        predictedWordDiv.onclick = () => {
            const currentText = textArea.value;
            const needsSpace = currentText.length > 0 && !currentText.endsWith(" ");
            textArea.value = currentText + (needsSpace ? " " : "") + word + " ";
            
            selectPred.innerHTML = "";
            autocompleteInline.textContent = "";
            updateWordCount();
            predictionStatusElem.textContent = "✅ Word inserted";
            predictionStatusElem.style.color = "#10b981";
            textArea.focus();
            
            setTimeout(() => {
                predictionStatusElem.textContent = "Ready";
                predictionStatusElem.style.color = "#667eea";
            }, 2000);
        };
        
        selectPred.appendChild(predictedWordDiv);
    });
    
    // Also show first prediction as AUTOCOMPLETE INLINE (gray text overlay)
    if (predictions.length > 0) {
        updateAutocompleteDisplay(predictions[0]);
    }
    
    predictionStatusElem.textContent = `✨ ${predictions.length} next-word predictions | ✍️ Autocomplete (Tab)`;
    predictionStatusElem.style.color = "#667eea";
}

// Accept autocomplete suggestion with Tab key
textArea.addEventListener("keydown", (e) => {
    if (e.key === "Tab" && autocompleteInline.textContent) {
        e.preventDefault();
        
        // Extract the suggestion
        const cursorPos = textArea.selectionStart;
        const textBeforeCursor = textArea.value.substring(0, cursorPos);
        const words = textBeforeCursor.split(/\s+/);
        const currentWord = words[words.length - 1];
        
        // Get first prediction from the buttons
        const firstPredButton = selectPred.querySelector("div");
        if (firstPredButton) {
            const prediction = firstPredButton.textContent.substring(3); // Remove "1. "
            
            // Replace current word with prediction
            const textBefore = textArea.value.substring(0, cursorPos - currentWord.length);
            const textAfter = textArea.value.substring(cursorPos);
            textArea.value = textBefore + prediction + " " + textAfter;
            
            // Move cursor
            textArea.selectionStart = textArea.selectionEnd = textBefore.length + prediction.length + 1;
            
            autocompleteInline.textContent = "";
            selectPred.innerHTML = "";
            
            predictionStatusElem.textContent = "✅ Autocomplete accepted (Tab)";
            predictionStatusElem.style.color = "#10b981";
            
            setTimeout(() => {
                predictionStatusElem.textContent = "Ready";
                predictionStatusElem.style.color = "#667eea";
            }, 2000);
        }
    }
});

// Initialize word count on load
updateWordCount();

// Model selector change handler
modelSelector.addEventListener("change", (e) => {
    const selectedModel = e.target.value;
    console.log(`%c[MODEL SWITCH] Changed to: ${selectedModel.toUpperCase()}`, 'background: #667eea; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold');
    
    if (selectedModel === "bidirectional") {
        modelNameElem.textContent = "Bidirectional LSTM Neural Network";
    } else if (selectedModel === "gru") {
        modelNameElem.textContent = "Gated Recurrent Unit (GRU) - PyTorch";
    } else {
        modelNameElem.textContent = "Long Short-Term Memory (LSTM) Neural Network";
    }
    
    // Clear predictions when model changes
    selectPred.innerHTML = "";
    autocompleteInline.textContent = "";
    predictionOverlay.style.display = "none";
    
    // Cancel any pending request
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
    clearTimeout(typingTimer);
    
    predictionStatusElem.textContent = `✅ Model switched to ${selectedModel.toUpperCase()} - Ready to predict`;
    predictionStatusElem.style.color = "#667eea";
    
    setTimeout(() => {
        predictionStatusElem.textContent = "Ready";
    }, 3000);
});

// Prevent form submission
docForm.addEventListener("submit", (e) => {
    e.preventDefault();
});

// ==================== GENERATION MODE HANDLERS ====================

// Mode selector change handler
modeSelector.addEventListener("change", (e) => {
    const mode = e.target.value;
    console.log(`%c[MODE SWITCH] Changed to: ${mode.toUpperCase()}`, 'background: #764ba2; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold');
    
    if (mode === "generate") {
        // Show generation controls
        generationControls.style.display = "block";
        generateFeature.style.display = "flex";
        generateExample.style.display = "block";
        
        // Hide prediction features
        predictFeature.style.display = "none";
        predictFeature2.style.display = "none";
        predictExample.style.display = "none";
        predictionOverlay.style.display = "none";
        autocompleteInline.style.display = "none";
        
        predictionStatusElem.textContent = "Generation Mode Active";
        predictionStatusElem.style.color = "#764ba2";
    } else {
        // Show prediction features
        generationControls.style.display = "none";
        generateFeature.style.display = "none";
        generateExample.style.display = "none";
        generatedOverlay.style.display = "none";
        
        // Show prediction features
        predictFeature.style.display = "flex";
        predictFeature2.style.display = "flex";
        predictExample.style.display = "block";
        
        predictionStatusElem.textContent = "Prediction Mode Active";
        predictionStatusElem.style.color = "#667eea";
    }
});

// Slider updates
lengthSlider.addEventListener("input", (e) => {
    lengthValue.textContent = e.target.value;
});

tempSlider.addEventListener("input", (e) => {
    tempValue.textContent = parseFloat(e.target.value).toFixed(1);
});

// Generate button handler
generateBtn.addEventListener("click", async () => {
    const text = textArea.value.trim();
    
    if (!text) {
        alert("Please enter some text as a prompt for generation!");
        return;
    }
    
    const selectedModel = modelSelector.value;
    const numWords = parseInt(lengthSlider.value);
    const temperature = parseFloat(tempSlider.value);
    
    // Show loading state
    generateBtn.disabled = true;
    generateBtn.textContent = "⏳ Generating...";
    predictionStatusElem.textContent = "🤖 AI generating text...";
    predictionStatusElem.style.color = "#f59e0b";
    
    try {
        console.log(`%c[GENERATION] Model: ${selectedModel.toUpperCase()}, Words: ${numWords}, Temp: ${temperature}`, 'color: #764ba2; font-weight: bold');
        
        const response = await fetch(`${API_BASE_URL}/generate/`, {
            method: "POST",
            headers: {
                'X-CSRFToken': csrftoken,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: text,
                model: selectedModel,
                num_words: numWords,
                temperature: temperature
            }),
            mode: 'same-origin'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        console.log(`%c[GENERATION SUCCESS]`, 'color: #10b981; font-weight: bold');
        console.log(`Generated: ${data.generated_text}`);
        
        // Display generated text
        generatedText.textContent = data.generated_text;
        generatedOverlay.style.display = "block";
        
        predictionStatusElem.textContent = "✨ Generation Complete!";
        predictionStatusElem.style.color = "#10b981";
        
    } catch (error) {
        console.error("Generation error:", error);
        predictionStatusElem.textContent = "❌ Generation failed";
        predictionStatusElem.style.color = "#ef4444";
        alert("Failed to generate text. Please try again.");
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = "✨ Generate Text";
    }
});

// Insert generated text into editor
insertGeneratedBtn.addEventListener("click", () => {
    const generated = generatedText.textContent;
    if (generated) {
        // Append to textarea with a space
        textArea.value += " " + generated;
        updateWordCount();
        generatedOverlay.style.display = "none";
        predictionStatusElem.textContent = "✅ Text inserted!";
        setTimeout(() => {
            predictionStatusElem.textContent = "Ready";
        }, 2000);
    }
});

