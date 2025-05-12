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
    const historyItems = document.querySelectorAll('.history-item');
    
    // Variables
    let currentJobId = null;
    let statusCheckInterval = null;

    // Check if there's a stored job ID to open
    const openJobId = localStorage.getItem('openJobId');
    if (openJobId) {
        // Clear the stored job ID
        localStorage.removeItem('openJobId');
        
        // Show the job result
        showHistoryResult(openJobId);
    }

    // Add event listeners to history items
    historyItems.forEach(item => {
        item.addEventListener('click', function() {
            const jobId = this.getAttribute('data-job-id');
            showHistoryResult(jobId);
        });
    });

    // Submit question form
    if (questionForm) {
        questionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const question = questionInput.value.trim();
            if (!question) return;
            
            // Add user message to chat
            addMessage('user', question);
            
            // Disable input and button
            questionInput.disabled = true;
            submitBtn.disabled = true;
            subquestionsCount.disabled = true;
            
            // Send question to server
            const formData = new FormData();
            formData.append('question', question);
            formData.append('subquestions_count', subquestionsCount.value);
            
            fetch('/ask', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    currentJobId = data.job_id;
                    statusContainer.classList.remove('d-none');
                    statusText.textContent = data.message;
                    
                    // Start checking status
                    checkStatus();
                    statusCheckInterval = setInterval(checkStatus, 5000);
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
        fetch(`/result/${jobId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Clear chat
                chatContainer.innerHTML = '';
                
                // Add messages
                addMessage('user', data.question);
                addMessage('assistant', formatAnswer(data.answer), data.timestamp);
                
                // Scroll to bottom
                chatContainer.scrollTop = chatContainer.scrollHeight;
                
                // Reset form
                if (questionInput) {
                    questionInput.value = '';
                    questionInput.disabled = false;
                }
                if (submitBtn) {
                    submitBtn.disabled = false;
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    
    // Check job status
    function checkStatus() {
        if (!currentJobId) return;
        
        fetch(`/status/${currentJobId}`)
        .then(response => response.json())
        .then(data => {
            // Update status text and progress bar
            if (data.status === 'Processing') {
                statusText.textContent = data.progress;
                progressBar.style.width = '50%';
            } else if (data.status === 'Completed') {
                clearInterval(statusCheckInterval);
                statusContainer.classList.add('d-none');
                
                // Get result
                fetch(`/result/${currentJobId}`)
                .then(response => response.json())
                .then(resultData => {
                    if (resultData.status === 'success') {
                        addMessage('assistant', formatAnswer(resultData.answer), resultData.timestamp);
                        resetForm();
                        
                        // Add to history without page reload
                        addToHistory(currentJobId, resultData.question, resultData.timestamp);
                    }
                });
            } else if (data.status === 'Error') {
                clearInterval(statusCheckInterval);
                statusContainer.classList.add('d-none');
                addMessage('assistant', `Error: ${data.error}`);
                resetForm();
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    
    // Add a message to the chat
    function addMessage(sender, message, timestamp = null) {
        if (!chatContainer) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = sender === 'user' ? 'user-message' : 'assistant-message';
        
        if (sender === 'assistant') {
            messageDiv.innerHTML = formatAnswer(message);
            
            // Add timestamp if provided
            if (timestamp) {
                const timeDiv = document.createElement('div');
                timeDiv.className = 'message-time';
                timeDiv.textContent = timestamp;
                messageDiv.appendChild(timeDiv);
            }
        } else {
            messageDiv.textContent = message;
        }
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Format the answer
    function formatAnswer(text) {
        // Convert markdown to HTML if marked is available
        if (window.marked) {
            // Replace single line breaks with two spaces and a newline for markdown to treat as <br>
            return marked.parse(text.replace(/\n/g, '  \n'));
        }
        // Basic formatting if marked is not available
        return text.replace(/\n/g, '<br>');
    }
    
    // Add to history without page reload
    function addToHistory(jobId, question, timestamp) {
        // Instead of just adding one, fetch and update the full history
        updateHistoryList();
    }

    // Fetch and update the chat history list
    function updateHistoryList() {
        fetch('/history_list')
        .then(response => response.text())
        .then(html => {
            const historyList = document.getElementById('history-list');
            if (historyList) {
                historyList.innerHTML = html;

                // Re-attach click handlers
                const historyItems = historyList.querySelectorAll('.history-item');
                historyItems.forEach(item => {
                    item.addEventListener('click', function() {
                        const jobId = this.getAttribute('data-job-id');
                        showHistoryResult(jobId);
                    });
                });
            }
        });
    }

    // Reset the form
    function resetForm() {
        if (questionInput) {
            questionInput.disabled = false;
            questionInput.value = '';
        }
        if (submitBtn) {
            submitBtn.disabled = false;
        }
        if (subquestionsCount) {
            subquestionsCount.disabled = false;
        }
        currentJobId = null;
    }
});
