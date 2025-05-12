document.addEventListener('DOMContentLoaded', function() {
    const abstractSearchForm = document.getElementById('abstract-search-form');
    const submitBtn = document.getElementById('submit-btn');
    const statusContainer = document.getElementById('status-container');
    const statusText = document.getElementById('status-text');
    const progressBar = document.getElementById('progress-bar');
    const resultContainer = document.getElementById('result-container');
    const resultContent = document.getElementById('result-content');    abstractSearchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const searchQuery = document.getElementById('search-query').value.trim();
        
        if (!searchQuery) {
            alert('Please enter a search query');
            return;
        }
        
        // Disable form and show status
        submitBtn.disabled = true;
        statusContainer.classList.remove('d-none');
        resultContainer.classList.add('d-none');
        statusText.textContent = 'Initializing search...';
        progressBar.style.width = '10%';
          // Send the request to start the abstract collection
        fetch('/abstracts/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'query': searchQuery
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Poll for job status
                pollJobStatus(data.job_id);
            } else {
                showError(data.message || 'An error occurred starting the search');
            }
        })
        .catch(err => {
            showError('Network error: ' + err.message);
        });
    });

    function pollJobStatus(jobId) {
        const checkStatus = () => {
            fetch(`/abstracts/status/${jobId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'Completed') {
                        // Job is complete, show the results
                        statusText.textContent = 'Search completed!';
                        progressBar.style.width = '100%';
                        setTimeout(() => {
                            showResults(data);
                        }, 1000);
                    } else if (data.status === 'Error') {
                        // Job error
                        showError(data.error || 'An error occurred during the search');
                    } else if (data.status === 'Processing') {
                        // Update progress
                        statusText.textContent = data.progress || 'Processing...';
                        progressBar.style.width = '50%';
                        // Continue polling
                        setTimeout(checkStatus, 3000);
                    } else if (data.status === 'Starting') {
                        statusText.textContent = data.progress || 'Starting search...';
                        progressBar.style.width = '20%';
                        // Continue polling
                        setTimeout(checkStatus, 2000);
                    } else if (data.status === 'not_found') {
                        showError('Job not found or expired');
                    }
                })
                .catch(err => {
                    showError('Error checking status: ' + err.message);
                });
        };
        
        // Start polling immediately
        checkStatus();
    }

    function showResults(data) {
        statusContainer.classList.add('alert-success');
        statusContainer.classList.remove('alert-info');
        
        // Show result container
        resultContainer.classList.remove('d-none');
          // Populate results
        let resultHtml = '';
        
        if (data.file_path) {
            resultHtml += `<p>Search completed successfully! The abstracts have been stored in the database.</p>`;
            resultHtml += `<p>File saved: ${data.file_path}</p>`;
        }
        
        if (data.count) {
            resultHtml += `<p><strong>${data.count}</strong> abstracts were found and indexed.</p>`;
        } else {
            resultHtml += `<p>Abstracts have been successfully stored in the database.</p>`;
        }
        
        resultHtml += `
            <div class="alert alert-success mt-3">
                <h5>Next Steps:</h5>
                <ol>
                    <li>Return to the <a href="/" class="alert-link">Home page</a></li>
                    <li>Enter a research question related to the abstracts you just collected</li>
                    <li>CiteSeekerAI will analyze these abstracts to provide a comprehensive answer</li>
                </ol>
                <p class="mb-0"><small>The abstracts are now available in the database for all future research questions as well.</small></p>
            </div>
        `;
        
        resultContent.innerHTML = resultHtml;
        
        // Re-enable form
        submitBtn.disabled = false;
    }

    function showError(message) {
        statusContainer.classList.add('alert-danger');
        statusContainer.classList.remove('alert-info');
        statusText.textContent = 'Error: ' + message;
        progressBar.style.width = '100%';
        submitBtn.disabled = false;
    }
});
