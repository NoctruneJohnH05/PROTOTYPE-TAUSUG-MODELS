const docForm = document.getElementById("doc-form")
const textArea = document.getElementById("document")
const selectPred = document.getElementById("select-pred")
const predictionOverlay = document.getElementById("prediction-overlay")
const autocompleteInline = document.getElementById("autocomplete-inline")
const wordCountElem = document.getElementById("word-count")
const predictionStatusElem = document.getElementById("prediction-status")
const modelSelector = document.getElementById("model-selector")
const modelNameElem = document.getElementById("model-name")

// Debug: Verify model selector is found
console.log("%c[INIT] Model Selector Element:", "color: orange; font-weight: bold", modelSelector);
console.log("%c[INIT] Initial Model Value:", "color: orange; font-weight: bold", modelSelector ? modelSelector.value : "NOT FOUND");

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
    
    predictionStatusElem.textContent = "🤖 AI processing...";
    predictionStatusElem.style.color = "#f59e0b";
    
    try {
        const selectedModel = modelSelector.value;
        console.log(`%c[MODEL] Using: ${selectedModel.toUpperCase()}`, 'color: #667eea; font-weight: bold; font-size: 14px');
        console.log(`[INPUT] Text: "${cleanedText}"`);
        
        const response = await fetch("http://127.0.0.1:8000/document/", {
            method: "POST",
            headers: {
                'X-CSRFToken': csrftoken,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: cleanedText,
                model: selectedModel
            }),
            mode: 'same-origin'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        console.log(`%c[RESPONSE] Model used by backend: ${data.model_used ? data.model_used.toUpperCase() : 'UNKNOWN'}`, 'color: #10b981; font-weight: bold');
        console.log(`[PREDICTIONS] ${data.preds}`);
        
        // Display predictions as TOP-3 NEXT-WORD BUTTONS
        if (data.preds && data.preds.trim()) {
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
        } else {
            predictionOverlay.style.display = "none";
            predictionStatusElem.textContent = "No predictions available";
            predictionStatusElem.style.color = "#718096";
        }
        
    } catch (error) {
        console.error("Prediction error:", error);
        predictionStatusElem.textContent = "❌ Error getting predictions";
        predictionStatusElem.style.color = "#ef4444";
        
        setTimeout(() => {
            predictionStatusElem.textContent = "Ready";
            predictionStatusElem.style.color = "#667eea";
        }, 3000);
    }
});

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
    } else {
        modelNameElem.textContent = "Long Short-Term Memory (LSTM) Neural Network";
    }
    
    // Clear predictions when model changes
    selectPred.innerHTML = "";
    autocompleteInline.textContent = "";
    predictionOverlay.style.display = "none";
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
