// Store results to avoid unnecessary API calls
let cachedResults = null;

async function generateAll() {
    showLoader("Processing video content...");
    
    const result = await fetchSummaryData();
    if (!result) {
        hideLoader();
        return;
    }
    
    // Cache the result
    cachedResults = result;
    
    // Display summary
    document.getElementById("summary").innerText = result.summary;
    
    // Display flowchart
    document.getElementById("flowchart").innerText = result.flowchart;
    
    // Display diagrams if available
    displayDiagrams(result.diagrams || []);
    
    // Show video info
    updateVideoInfo(result);
    
    // Switch to summary tab by default
    switchTab('summary');
    
    // Show results with animation
    document.getElementById("resultContainer").classList.add("visible");
    hideLoader();
}

async function fetchSummaryData() {
    const url = document.getElementById("youtubeUrl").value;
    const points = document.getElementById("points").value;
    const detectDiagrams = document.getElementById("extractDiagrams").checked;
    
    if (!url) {
        showError("Missing URL", "Please enter a YouTube URL to generate a summary.");
        return null;
    }
    
    try {
        const response = await fetch("http://localhost:5000/summarize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                url: url,
                points: parseInt(points),
                detect_diagrams: detectDiagrams
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            showError(result.error, result.details || "");
            return null;
        }
        
        return result;
    } catch (error) {
        showError("Connection Error", "Failed to connect to server. Make sure your Flask backend is running!");
        return null;
    }
}

function displayDiagrams(diagrams) {
    const diagramsGrid = document.getElementById("diagramsGrid");
    const noDiagrams = document.getElementById("noDiagrams");
    
    diagramsGrid.innerHTML = '';
    
    if (diagrams.length === 0) {
        noDiagrams.style.display = 'block';
        return;
    }
    
    noDiagrams.style.display = 'none';
    
    diagrams.forEach(diagram => {
        const confidenceLevel = diagram.confidence || 0;
        
        const diagramCard = document.createElement('div');
        diagramCard.className = 'diagram-card';
        diagramCard.innerHTML = `
            <img class="diagram-image" src="data:image/jpeg;base64,${diagram.image}" alt="Diagram at ${diagram.timestamp}">
            <span class="diagram-timestamp">${diagram.timestamp}</span>
            <div class="diagram-content">
                <p>${diagram.description || 'Diagram extracted from video'}</p>
                <div class="diagram-confidence">
                    <div class="confidence-bar">
                        <div class="confidence-level" style="width: ${confidenceLevel}%"></div>
                    </div>
                    <span class="confidence-text">${confidenceLevel}% confidence</span>
                </div>
            </div>
        `;
        
        diagramsGrid.appendChild(diagramCard);
    });
}

function updateVideoInfo(result) {
    let methodText = result.method === "subtitles" ? "Subtitles" : "Audio Transcription";
    let diagramsText = result.diagrams && result.diagrams.length > 0 ? 
        `${result.diagrams.length} diagrams found` : "No diagrams found";
    
    document.getElementById("videoInfo").innerHTML = `
        <span><i class="fab fa-youtube"></i> <strong>Video:</strong> ${result.video_title || "Unknown Title"}</span>
        <span><i class="fas fa-robot"></i> <strong>AI Model:</strong> ${result.model}</span>
        <span><i class="fas fa-microphone"></i> <strong>Method:</strong> ${methodText}</span>
        <span><i class="fas fa-chart-bar"></i> <strong>Diagrams:</strong> ${diagramsText}</span>
    `;
}

function showLoader(message) {
    document.getElementById("loader").style.display = "block";
    const loadingText = document.getElementById("loadingText");
    loadingText.textContent = message;
    loadingText.style.display = "block";
}

function hideLoader() {
    document.getElementById("loader").style.display = "none";
    document.getElementById("loadingText").style.display = "none";
}

function switchTab(tabName) {
    // Deactivate all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Deactivate all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Activate selected tab
    document.getElementById(tabName + 'Tab').classList.add('active');
    document.getElementById(tabName + 'Content').classList.add('active');
}

function showError(title, message) {
    const resultContainer = document.getElementById("resultContainer");
    const summaryDiv = document.getElementById("summary");
    
    summaryDiv.innerHTML = `<div class="error-message">
        <strong>${title}</strong><br>
        ${message}
    </div>`;
    
    document.getElementById("flowchart").innerText = "";
    document.getElementById("diagramsGrid").innerHTML = "";
    document.getElementById("noDiagrams").style.display = "block";
    
    // Show results container with error
    resultContainer.classList.add("visible");
    
    // Ensure we're on the summary tab to show the error
    switchTab('summary');
}

// Copy functionality
document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("copySummary").addEventListener("click", function() {
        const summaryText = document.getElementById("summary").innerText;
        navigator.clipboard.writeText(summaryText).then(() => {
            this.innerHTML = '<i class="fas fa-check"></i> Copied!';
            setTimeout(() => {
                this.innerHTML = '<i class="far fa-copy"></i> Copy';
            }, 2000);
        });
    });
});