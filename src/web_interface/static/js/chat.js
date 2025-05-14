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
    const outlineContainer = document.getElementById('outline-container'); // Added for outline
    
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
                // Add IDs to H4 tags that look like "Subquery X" for the outline
                html = html.replace(/<h4(.*?)>(Subquery\s+\d+)<\/h4>/gi, function(match, p1, title) {
                    const id = title.toLowerCase().replace(/\s+/g, '-');
                    return `<h4 id="${id}"${p1}>${title}</h4>`;
                });
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

        // Regex to find all parenthesized groups: (content)
        const parenthesizedGroupRegex = /\(([^)]+)\)/g;

        html = html.replace(parenthesizedGroupRegex, (fullMatch, innerContent) => {
            // Only process as citation if it contains "Chunk #"
            if (!innerContent.includes("Chunk #")) {
                return fullMatch; // Return original text for non-citation parentheses
            }
            
            const currentJobIdForSpans = currentJobId || '';
            if (!currentJobIdForSpans) { // Warn if job ID is missing
                console.warn("currentJobId is not set. Citation tooltips may not work for group:", fullMatch);
            }

            const citationSegments = innerContent.split(';');
            let processedSegmentsOutput = [];

            // Regex to parse individual citation segments.
            // Captures: 1. baseRef (e.g., "Author et al., YYYY")
            //           2. chunksGroup (e.g., ", Chunk #1, Chunk #2") - requires at least one chunk.
            //           3. citingInfo (e.g., ", citing Some Ref, YYYY") - optional.
            //           4. additionalInfo (e.g., ", Equation (15)") - optional.
            const segmentRegex = /^\s*([A-Z][^,]+(?:,\s*(?!Chunk #\d|citing)[^,]+)*?,\s*\d{4})((?:,\s*Chunk #\d+)+)(,\s*citing\s*[^;)]+)?(.*)\s*$/;

            for (const segment of citationSegments) {
                const segmentTrimmed = segment.trim();
                const segmentMatch = segmentTrimmed.match(segmentRegex);

                if (segmentMatch) {
                    let [, baseRef, chunksGroup, citingInfo, additionalInfo] = segmentMatch;
                    // Ensure citingInfo is a string (empty if not present or only whitespace)
                    citingInfo = citingInfo ? citingInfo.trim() : "";
                    additionalInfo = additionalInfo ? additionalInfo.trim() : ""; // Trim and ensure string

                    let currentSegmentHtml = "";
                    // Split chunksGroup by (Chunk #N) and filter out empty strings from the split.
                    // Example: chunksGroup = ", Chunk #0, Chunk #5" -> chunkParts = [", ", "Chunk #0", ", ", "Chunk #5"]
                    const chunkParts = chunksGroup.split(/(Chunk #\d+)/g).filter(part => part); 

                    let isFirstChunkInSegment = true;
                    // accumulatedTextForFirstChunkSpan will hold `baseRef` + any leading separators from chunksGroup before the first chunk.
                    let accumulatedTextForFirstChunkSpan = baseRef; 
                    
                    let elementsToRender = []; // Stores objects representing parts of the citation (chunk or separator)

                    for (const part of chunkParts) {
                        if (part.startsWith("Chunk #")) { // This is a chunk name, e.g., "Chunk #0"
                            const chunkName = part;
                            const citationKeyForLookup = `${baseRef}, ${chunkName}`.trim();
                            let displayText;

                            if (isFirstChunkInSegment) {
                                // Display text for the first chunk includes accumulated baseRef and leading separator(s) from chunksGroup.
                                displayText = accumulatedTextForFirstChunkSpan + chunkName;
                                accumulatedTextForFirstChunkSpan = ""; // Clear accumulator as it's now part of displayText
                                isFirstChunkInSegment = false;
                            } else {
                                // Display text for subsequent chunks is just their name.
                                displayText = chunkName;
                            }
                            elementsToRender.push({
                                type: 'chunk',
                                citationKey: citationKeyForLookup,
                                text: displayText
                            });
                        } else { // This part is a separator from chunksGroup, e.g., ", "
                            if (isFirstChunkInSegment) {
                                // If before the first chunk, append to its accumulator.
                                accumulatedTextForFirstChunkSpan += part;
                            } else {
                                // If it's a separator between chunks, store it as plain text.
                                elementsToRender.push({ type: 'separator', text: part });
                            }
                        }
                    }

                    // Construct the HTML for the segment, attaching citingInfo and additionalInfo to the last chunk span.
                    for (let i = 0; i < elementsToRender.length; i++) {
                        const el = elementsToRender[i];
                        if (el.type === 'chunk') {
                            let spanText = el.text;
                            // Check if this is the last 'chunk' type element in elementsToRender
                            let isLastChunkElement = true;
                            for(let j = i + 1; j < elementsToRender.length; j++) {
                                if (elementsToRender[j].type === 'chunk') {
                                    isLastChunkElement = false;
                                    break;
                                }
                            }
                            // If this is the last chunk element, append citingInfo and additionalInfo.
                            if (isLastChunkElement) {
                                if (citingInfo) {
                                    spanText += citingInfo;
                                }
                                if (additionalInfo) {
                                    // HTML-escape additionalInfo as it's free text
                                    spanText += additionalInfo.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                                }
                            }
                            currentSegmentHtml += `<span class="citation" data-job-id="${currentJobIdForSpans}" data-citation-key="${el.citationKey}">${spanText}</span>`;
                        } else if (el.type === 'separator') {
                            currentSegmentHtml += el.text; // Add separator as plain text
                        }
                    }
                    processedSegmentsOutput.push(currentSegmentHtml);

                } else {
                    // Segment didn't match the citation structure, add as plain text (HTML-escaped)
                    processedSegmentsOutput.push(segment.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"));
                }
            }
            
            // Join processed segments with ';' and wrap in original parentheses.
            // Then, wrap the entire reconstructed citation group in a span for hover styling,
            // but without a data-citation-key so it doesn't trigger a tooltip itself.
            return `<span class="citation" data-job-id="${currentJobIdForSpans}">(${processedSegmentsOutput.join(';')})</span>`;
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
            if (outlineContainer) outlineContainer.innerHTML = ''; // Clear outline
            updateOutline(null, []); // Explicitly clear outline with empty subqueries
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
                if (outlineContainer) outlineContainer.innerHTML = ''; // Clear outline before loading history
                
                addMessage('user', data.question);
                // Pass subqueries to addMessage, which will then pass to updateOutline
                addMessage('assistant', data.answer, data.timestamp, data.subqueries || []); 
                
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
                        addMessage('assistant', resultData.answer, resultData.timestamp, resultData.subqueries || []);
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
    function addMessage(sender, message, timestamp = null, subqueries = null) { // Added subqueries parameter
        if (!chatContainer) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.classList.add(sender === 'user' ? 'user-message' : 'assistant-message', 'p-3', 'mb-3', 'rounded');
        
        let formattedMessageContent = message;
        if (sender === 'assistant') {
            formattedMessageContent = formatAnswer(message); // formatAnswer handles markdown and citations
            messageDiv.innerHTML = formattedMessageContent;
            
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

        if (sender === 'assistant') {
            // Pass both HTML content (for IDs) and subqueries list (for text)
            updateOutline(formattedMessageContent, subqueries); 
        }
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

            const citationKey = el.getAttribute('data-citation-key');
            const jobId = el.getAttribute('data-job-id');
            
            // If there's no citationKey, this span is likely a wrapper for styling (the whole citation group).
            // Do not attempt to show a tooltip for it. It will still get hover styling from CSS.
            if (!citationKey) {
                return; 
            }

            // If citationKey is present, but jobId is missing, it's an issue for fetching the chunk.
            if (!jobId) {
                console.warn("Missing job-id on citation element with key:", citationKey, el);
                citationTooltip.innerText = 'Error: Missing job ID for citation.';
                citationTooltip.style.display = 'block'; 
                positionTooltip(el);
                return;
            }

            citationTooltip.innerText = 'Loading chunk...';
            citationTooltip.style.display = 'block';
            positionTooltip(el);

            // Using full absolute path to ensure URL is correctly formed
            const fetchUrl = `${window.location.origin}/get_prompt_chunk?job_id=${encodeURIComponent(jobId)}&citation_key=${encodeURIComponent(citationKey)}`;
            
            fetch(fetchUrl)
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
                    console.error("Error fetching chunk:", err, "URL was:", fetchUrl);
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
    
    // Function to update the outline panel
    function updateOutline(answerHtml, subqueries = null) { // Added subqueries parameter
        if (!outlineContainer) return;

        outlineContainer.innerHTML = ''; // Clear previous outline

        // If subqueries are provided and is an array with items, use them for the outline
        if (subqueries && Array.isArray(subqueries) && subqueries.length > 0) {
            const cardHeader = document.createElement('div');
            cardHeader.className = 'card-header';
            cardHeader.textContent = 'Outline';
            outlineContainer.appendChild(cardHeader);

            const list = document.createElement('ul');
            list.className = 'list-group list-group-flush flex-grow-1'; // Bootstrap styling + flex grow
            // Remove hardcoded maxHeight, controlled by CSS now
            list.style.overflowY = 'auto';  // Make outline scrollable if too long


            subqueries.forEach((subqueryText, index) => {
                const listItem = document.createElement('li');
                listItem.className = 'list-group-item';
                
                const link = document.createElement('a');
                // Generate ID based on index (1-based for subquery numbers)
                const targetId = `subquery-${index + 1}`; 
                link.href = `#${targetId}`;
                // Display format: "1. {subquery text}"
                link.textContent = `${index + 1}. ${subqueryText}`; 
                link.className = 'outline-link d-block'; // Removed text-truncate to allow wrapping
                link.style.cursor = 'pointer';
                link.style.whiteSpace = 'normal'; // Allow text to wrap
                link.style.overflowWrap = 'break-word'; // Break long words if necessary


                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const targetElement = document.getElementById(this.getAttribute('href').substring(1));
                    if (targetElement) {
                        targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                });

                listItem.appendChild(link);
                list.appendChild(listItem);
            });
            outlineContainer.appendChild(list);
        } else if (answerHtml) { // Fallback or for messages without explicit subqueries (like initial greeting)
            // This part tries to find H4s if no subqueries array is given.
            // For the initial greeting, this will likely find nothing, which is fine.
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = answerHtml;
            const headers = tempDiv.querySelectorAll('h4[id^="subquery-"]');
            
            if (headers.length > 0) {
                const cardHeader = document.createElement('div');
                cardHeader.className = 'card-header';
                cardHeader.textContent = 'Outline';
                outlineContainer.appendChild(cardHeader);

                const list = document.createElement('ul');
                list.className = 'list-group list-group-flush flex-grow-1';
                // Remove hardcoded maxHeight, controlled by CSS now
                list.style.overflowY = 'auto';

                headers.forEach(header => {
                    const listItem = document.createElement('li');
                    listItem.className = 'list-group-item';
                    
                    const link = document.createElement('a');
                    link.href = `#${header.id}`;
                    link.textContent = header.textContent; // This would be "Subquery X"
                    link.className = 'outline-link d-block text-truncate';
                    link.style.cursor = 'pointer';
                    link.addEventListener('click', function(e) {
                        e.preventDefault();
                        const targetElement = document.getElementById(this.getAttribute('href').substring(1));
                        if (targetElement) {
                            targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }
                    });

                    listItem.appendChild(link);
                    list.appendChild(listItem);
                });
                outlineContainer.appendChild(list);
            }
        }
        // If neither subqueries nor headers are found, the outline remains empty.
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
