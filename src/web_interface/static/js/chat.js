// CiteSeekerAI Web Interface JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const questionForm = document.getElementById('question-form');
    const questionInput = document.getElementById('question-input');
    const submitBtn = document.getElementById('submit-btn');
    const subquestionsCount = document.getElementById('subquestions-count');
    const chatContainer = document.getElementById('chat-container');
    const statusContainer = document.getElementById('status-container');
    const statusText = document.getElementById('status-text');
    const progressBar = document.getElementById('progress-bar');
    const newChatBtn = document.getElementById('new-chat-btn'); // Added
    // const historyItems = document.querySelectorAll('.history-item'); // Will be handled by updateHistoryList
    
    // Variables
    let currentJobId = null;
    let statusCheckInterval = null;
    let hideTooltipTimeout = null; // Variable to manage tooltip hide delay

    // Function to format answer content (Markdown and Citations)
    function formatAnswer(text) {
        let html = text;
        // Apply Markdown processing first
        if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
            try {
                html = marked.parse(text);
            } catch (e) {
                console.error("Error parsing markdown:", e);
                // Fallback to plain text if markdown parsing fails
                html = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
                html = html.replace(/\n/g, '<br>');
            }
        } else {
            // Basic fallback: escape HTML and convert newlines to <br> if marked.js is not available
            html = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
            html = html.replace(/\n/g, '<br>');
        }

        // Then, wrap citations in the generated HTML
        // Regex to find patterns like (Author et al., YYYY, Chunk #N) or (Author et al., YYYY, Chunk #N, citing Other, YYYY)
        // It captures:
        // 1. The main citation key (e.g., "Author et al., YYYY, Chunk #N")
        // 2. The optional "citing" part (e.g., ", citing Other, YYYY")
        const citationRegex = /\(([A-Z][^;()]*?Chunk #\d+)((?:, citing [^;()]+(?:; [^;()]+)*)?)\)/g;
        
        html = html.replace(citationRegex, function(match, keyPart, citingPart) {
            // match: The entire matched string, e.g., "(Author et al., YYYY, Chunk #N, citing Other, YYYY)"
            // keyPart: The main citation key, e.g., "Author et al., YYYY, Chunk #N"
            // citingPart: The citing information, e.g., ", citing Other, YYYY" (or empty if not present)
            
            if (!currentJobId) {
                console.warn("currentJobId is not set while formatting citations. Citation tooltips may not work for:", match);
            }
            const citationKeyForLookup = keyPart.trim();
            
            return `<span class="citation" data-job-id="${currentJobId}" data-citation-key="${citationKeyForLookup}">${match}</span>`;
        });
        return html;
    }

    // Add event listeners to history items (delegated or re-attached after update)
    function attachHistoryEventListeners() {
        const historyList = document.getElementById('history-list');
        if (historyList) {
            historyList.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', function() {
                    const jobId = this.getAttribute('data-job-id');
                    showHistoryResult(jobId);
                });
            });
        }
    }
    // Initial attachment
    attachHistoryEventListeners();

    // New Chat button functionality
    if (newChatBtn) {
        newChatBtn.addEventListener('click', function() {
            if (chatContainer) {
                chatContainer.innerHTML = ''; // Clear chat area
                // Add default greeting message
                addMessage('assistant', "Hello! I'm CiteSeekerAI, your research assistant. Ask me a research question, and I'll analyze academic literature to provide you with a comprehensive answer.\n\nFor example: \"What is the difference between water scarcity and water security?\"");
            }
            currentJobId = null; // Reset current job ID
            if (statusContainer) statusContainer.classList.add('d-none'); // Hide status
            if (progressBar) {
                progressBar.style.width = '0%'; // Reset progress bar
                progressBar.classList.remove('bg-danger');
            }
            resetForm(); // Reset the input form
            questionInput.value = ''; // Clear the question input field
            // Optionally, deselect any active history item visually if you implement such styling
        });
    }


    // Submit question form
    if (questionForm) {
        questionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const question = questionInput.value.trim();
            if (!question) return;
            
            addMessage('user', question);
            
            questionInput.disabled = true;
            submitBtn.disabled = true;
            if (subquestionsCount) subquestionsCount.disabled = true;
            
            const formData = new FormData();
            formData.append('question', question);
            if (subquestionsCount) formData.append('subquestions_count', subquestionsCount.value);
            
            fetch('/ask', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    currentJobId = data.job_id; // Set currentJobId for this new conversation
                    statusContainer.classList.remove('d-none');
                    statusText.textContent = data.message;
                    progressBar.style.width = '0%';
                    
                    checkStatus();
                    if (statusCheckInterval) clearInterval(statusCheckInterval);
                    statusCheckInterval = setInterval(checkStatus, 3000); // Check more frequently initially
                } else {
                    addMessage('assistant', `Error: ${data.message}`);
                    resetForm();
                }
            })
            .catch(error => {
                console.error('Error:', error);
                addMessage('assistant', 'Sorry, there was an error processing your request.');
                resetForm();
            });
        });
    }
    
    // Show a result from history
    function showHistoryResult(jobId) {
        currentJobId = jobId; // Set currentJobId for this historical conversation
        fetch(`/result/${jobId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                if (chatContainer) chatContainer.innerHTML = '';
                
                addMessage('user', data.question);
                addMessage('assistant', data.answer, data.timestamp); // formatAnswer will be called inside addMessage
                
                if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
                
                resetForm(); // Reset form but currentJobId remains for this view
                // If user types a new question, currentJobId will be updated by /ask response
            } else {
                addMessage('assistant', `Error: Could not load result for ${jobId}.`);
            }
        })
        .catch(error => {
            console.error('Error fetching history result:', error);
            addMessage('assistant', 'Sorry, there was an error loading the history.');
        });
    }
    
    // Check job status
    function checkStatus() {
        if (!currentJobId) return;
        
        fetch(`/status/${currentJobId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'not_found') {
                clearInterval(statusCheckInterval);
                statusText.textContent = "Job not found or expired.";
                progressBar.style.width = '0%';
                // Potentially hide status container or show error more permanently
                return;
            }

            statusText.textContent = data.progress || data.status;
            // Example: Update progress bar based on fine-grained steps if available
            // For now, simple 50% for processing, 100% for completed
            if (data.status === 'Processing') {
                progressBar.style.width = '50%'; // Or more granular if progress provides percentage
            } else if (data.status === 'Completed') {
                clearInterval(statusCheckInterval);
                progressBar.style.width = '100%';
                statusContainer.classList.add('d-none'); // Hide after a short delay?
                
                fetch(`/result/${currentJobId}`)
                .then(response => response.json())
                .then(resultData => {
                    if (resultData.status === 'success') {
                        // Assistant message already added? No, this is where it's first shown for a new query.
                        // If chatContainer already has messages (e.g. user query), this adds to it.
                        // If it's a fresh page load and job completes, this might be the first message.
                        addMessage('assistant', resultData.answer, resultData.timestamp);
                        resetForm();
                        updateHistoryList(); // Refresh history list
                    } else {
                        addMessage('assistant', `Error retrieving result: ${resultData.message || 'Unknown error'}`);
                        resetForm();
                    }
                });
            } else if (data.status === 'Error') {
                clearInterval(statusCheckInterval);
                progressBar.style.width = '100%'; // Show completion of attempt
                progressBar.classList.add('bg-danger'); // Indicate error
                statusText.textContent = `Error: ${data.error || 'Unknown error'}`;
                // Do not hide statusContainer immediately on error, let user see it.
                resetForm();
            } else if (data.status === 'Starting') {
                 progressBar.style.width = '10%';
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
            clearInterval(statusCheckInterval);
            statusText.textContent = "Error checking status. Please try again.";
            // resetForm(); // Optional: reset form on status check error
        });
    }
    
    // Add a message to the chat
    function addMessage(sender, message, timestamp = null) {
        if (!chatContainer) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.classList.add(sender === 'user' ? 'user-message' : 'assistant-message', 'p-3', 'mb-3', 'rounded');
        
        if (sender === 'assistant') {
            messageDiv.innerHTML = formatAnswer(message); // formatAnswer handles markdown and citations
            
            if (timestamp) {
                const timeDiv = document.createElement('div');
                timeDiv.className = 'message-time text-muted small mt-2';
                timeDiv.textContent = timestamp;
                messageDiv.appendChild(timeDiv);
            }
        } else {
            // For user messages, just display plain text (or escape it)
            const p = document.createElement('p');
            p.textContent = message;
            messageDiv.appendChild(p);
        }
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Create tooltip element for citation chunks
    const citationTooltip = document.createElement('div');
    citationTooltip.id = 'citation-tooltip';
    citationTooltip.className = 'citation-tooltip'; // Ensure CSS class matches
    document.body.appendChild(citationTooltip);

    // Event delegation for citation hover
    document.body.addEventListener('mouseover', function(e) {
        const el = e.target.closest('.citation'); // Use closest to handle nested elements if any
        if (el) {
            clearTimeout(hideTooltipTimeout); // Cancel any pending hide operations

            const jobId = el.getAttribute('data-job-id');
            const citationKey = el.getAttribute('data-citation-key');
            
            if (!jobId || !citationKey) {
                console.warn("Missing job-id or citation-key on element:", el);
                citationTooltip.innerText = 'Error: Missing citation data.';
                citationTooltip.style.display = 'block'; // Show error in tooltip
                positionTooltip(el);
                return;
            }

            citationTooltip.innerText = 'Loading chunk...';
            citationTooltip.style.display = 'block';
            positionTooltip(el);

            fetch(`/get_prompt_chunk?job_id=${encodeURIComponent(jobId)}&citation_key=${encodeURIComponent(citationKey)}`)
                .then(res => res.json())
                .then(data => {
                    if (citationTooltip.style.display === 'block') { // Check if tooltip is still meant to be shown
                        if (data.status === 'success') {
                            citationTooltip.innerText = data.chunk;
                        } else {
                            citationTooltip.innerText = data.message || 'Chunk not found.';
                        }
                        // Re-position after content is loaded as size might change
                        positionTooltip(el);
                    }
                })
                .catch(err => {
                    console.error("Error fetching chunk:", err);
                    if (citationTooltip.style.display === 'block') {
                        citationTooltip.innerText = 'Error loading chunk.';
                        positionTooltip(el);
                    }
                });
        }
    });

    document.body.addEventListener('mouseout', function(e) {
        const el = e.target.closest('.citation');
        if (el) {
            // Delay hiding the tooltip, allowing mouse to move into the tooltip
            hideTooltipTimeout = setTimeout(() => {
                citationTooltip.style.display = 'none';
            }, 300); // 300ms delay
        }
    });

    // Keep tooltip open if mouse enters the tooltip itself
    citationTooltip.addEventListener('mouseenter', function() {
        clearTimeout(hideTooltipTimeout);
    });

    // Hide tooltip if mouse leaves the tooltip
    citationTooltip.addEventListener('mouseleave', function() {
        hideTooltipTimeout = setTimeout(() => {
            citationTooltip.style.display = 'none';
        }, 300); // 300ms delay
    });

    function positionTooltip(targetElement) {
        const rect = targetElement.getBoundingClientRect();
        citationTooltip.style.top = `${window.scrollY + rect.bottom + 5}px`; // 5px below the element
        citationTooltip.style.left = `${window.scrollX + rect.left}px`;
        // Basic boundary detection (viewport right edge)
        if (rect.left + citationTooltip.offsetWidth > window.innerWidth) {
            citationTooltip.style.left = `${window.scrollX + window.innerWidth - citationTooltip.offsetWidth - 5}px`;
        }
        // Basic boundary detection (viewport bottom edge)
        if (rect.bottom + 5 + citationTooltip.offsetHeight > window.innerHeight) {
            citationTooltip.style.top = `${window.scrollY + rect.top - citationTooltip.offsetHeight - 5}px`;
        }

    }
    
    // Fetch and update the chat history list
    function updateHistoryList() {
        fetch('/history_list') // Assuming you have or will create this endpoint
        .then(response => response.text())
        .then(html => {
            const historyList = document.getElementById('history-list');
            if (historyList) {
                historyList.innerHTML = html;
                attachHistoryEventListeners(); // Re-attach click handlers
            }
        })
        .catch(error => console.error("Error updating history list:", error));
    }

    // Reset the form
    function resetForm() {
        if (questionInput) {
            questionInput.disabled = false;
            // questionInput.value = ''; // Keep question for context if needed, or clear
        }
        if (submitBtn) {
            submitBtn.disabled = false;
        }
        if (subquestionsCount) {
            subquestionsCount.disabled = false;
        }
        // currentJobId is not reset here, as it might be needed for viewing results
        // It will be overwritten when a new question is asked.
        if (progressBar) {
            progressBar.classList.remove('bg-danger'); // Reset error state
        }
    }

    // Initial load of history if applicable, or if you want to ensure it's fresh
    // updateHistoryList(); // Call if you want to load/refresh history on page load
});
