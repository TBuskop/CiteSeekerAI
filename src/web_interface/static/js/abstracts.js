document.addEventListener('DOMContentLoaded', function() {
    const abstractSearchForm = document.getElementById('abstract-search-form');
    const submitBtn = document.getElementById('submit-btn');
    const statusContainer = document.getElementById('status-container');
    const statusText = document.getElementById('status-text');
    const progressBar = document.getElementById('progress-bar');
    const resultContainer = document.getElementById('result-container');
    const resultContent = document.getElementById('result-content');
    const searchQuery = document.getElementById('search-query');
    const scopusSearchScope = document.getElementById('scopus-search-scope');
    
    // New elements for database abstracts panel
    const abstractsList = document.getElementById('abstracts-list');
    const abstractCount = document.getElementById('abstract-count');
    const abstractSearch = document.getElementById('abstract-search');
    const abstractSearchBtn = document.getElementById('abstract-search-btn');
    
    // Load abstracts when page loads
    loadDatabaseAbstracts();
    
    // Search abstracts when search button is clicked
    abstractSearchBtn.addEventListener('click', function() {
        const searchTerm = abstractSearch.value.trim();
        loadDatabaseAbstracts(searchTerm);
    });
    
    // Search abstracts when Enter is pressed
    abstractSearch.addEventListener('keyup', function(e) {
        if (e.key === 'Enter') {
            const searchTerm = abstractSearch.value.trim();
            loadDatabaseAbstracts(searchTerm);
        }
    });
    
    abstractSearchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const query = searchQuery.value.trim();
        const selectedScope = scopusSearchScope.value;
        
        if (!query) {
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
                'query': query, // Ensure query is passed, not the element
                'scopus_search_scope': selectedScope // Added
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
        
        // Always scroll to results, not just on mobile
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Populate results
        let resultHtml = '';
        
        if (data.file_path) {
            resultHtml += `<p class="mb-1"><strong>Success!</strong> Abstracts stored in database.</p>`;
            resultHtml += `<p class="mb-1 text-muted small">File: ${data.file_path}</p>`;
        }
        
        if (data.count) {
            resultHtml += `<p class="mb-1"><strong>${data.count}</strong> abstracts indexed.</p>`;
        } else {
            resultHtml += `<p class="mb-1">Abstracts successfully stored.</p>`;
        }
        
        resultHtml += `
            <div class="alert alert-success mt-2 py-1 px-2">
                <h6 class="mb-1 fw-bold">Next Steps:</h6>
                <ol class="mb-1 ps-3 small">
                    <li>Return to <a href="/" class="alert-link">Home</a></li>
                    <li>Enter a research question</li>
                    <li>CiteSeekerAI will analyze these abstracts</li>
                </ol>
            </div>
        `;
        
        resultContent.innerHTML = resultHtml;
        
        // Reload abstracts after successful search
        loadDatabaseAbstracts();
        
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
    
    // Function to load abstracts from the database
    function loadDatabaseAbstracts(searchTerm = '', page = 1, append = false) {
        if (page === 1 && !append) {
            // Show loading state (only on first page or non-append loads)
            abstractsList.innerHTML = `
                <div class="text-center py-4">
                    <div class="spinner-border spinner-border-sm text-secondary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="text-muted small mt-2">Loading abstracts...</p>
                </div>
            `;
        } else if (append) {
            // When appending, add a loading indicator at the bottom
            const loadingIndicator = document.createElement('div');
            loadingIndicator.id = 'load-more-indicator';
            loadingIndicator.className = 'text-center py-2';
            loadingIndicator.innerHTML = `
                <div class="spinner-border spinner-border-sm text-secondary" role="status">
                    <span class="visually-hidden">Loading more...</span>
                </div>
            `;
            abstractsList.appendChild(loadingIndicator);
        }
        
        // API endpoint to fetch abstracts with pagination
        const url = new URL('/abstracts/list', window.location.origin);
        const params = { 
            page: page,
            search_fields: 'title,authors' // Specify that we only want to search titles and authors
        };
        if (searchTerm) params.search = searchTerm;
        url.search = new URLSearchParams(params).toString();
        
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to fetch abstracts');
                }
                return response.json();
            })
            .then(data => {
                // Remove loading indicator if appending
                const loadingIndicator = document.getElementById('load-more-indicator');
                if (loadingIndicator) loadingIndicator.remove();
                
                // Update count badge with total count
                abstractCount.textContent = data.total || 0;
                
                if (data.abstracts.length === 0 && page === 1) {
                    abstractsList.innerHTML = `
                        <div class="text-center py-4">
                            <p class="text-muted small">No abstracts found${searchTerm ? ' matching "' + searchTerm + '"' : ''}.</p>
                        </div>
                    `;
                    return;
                }
                
                // Clear container if not appending
                if (!append) {
                    abstractsList.innerHTML = '';
                }
                
                // No need to sort abstracts here - they're already sorted by the backend
                
                // Add each abstract
                data.abstracts.forEach(abstract => {
                    const abstractElement = document.createElement('div');
                    abstractElement.className = 'abstract-item';
                    
                    // Format authors for APA style
                    let authorsText = '';
                    if (abstract.authors) {
                        const authors = abstract.authors.split(', ');
                        if (authors.length === 1) {
                            authorsText = authors[0];
                        } else if (authors.length === 2) {
                            authorsText = `${authors[0]} & ${authors[1]}`;
                        } else {
                            authorsText = `${authors[0]} et al.`;
                        }
                    } else {
                        authorsText = 'Unknown';
                    }
                    
                    // Format APA citation with reference-style formatting
                    abstractElement.innerHTML = `
                        <div class="reference-style-citation">
                            <span class="authors">${authorsText}</span>
                            <span class="year">${abstract.year ? ` (${abstract.year}).` : ''}</span>
                            <span class="title"> ${abstract.title || 'Untitled'}.</span>
                            <span class="source"> <em>${abstract.source_title || ''}</em></span>
                            ${abstract.cited_by ? `<span class="cited-by">. Cited by: ${abstract.cited_by}</span>` : ''}
                            ${abstract.doi ? `<span class="doi-link">. <a href="https://doi.org/${abstract.doi}" target="_blank" class="doi_url">https://doi.org/${abstract.doi}</a></span>` : ''}
                        </div>
                        <div class="abstract-text">${abstract.document || 'No abstract available'}</div>
                        <span class="abstract-toggle" data-action="show">
                            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="currentColor" class="bi bi-chevron-down me-1" viewBox="0 0 16 16">
                                <path fill-rule="evenodd" d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/>
                            </svg>
                            Show abstract
                        </span>
                    `;
                    
                    // Toggle abstract visibility with improved UI indicators
                    const toggle = abstractElement.querySelector('.abstract-toggle');
                    const abstractText = abstractElement.querySelector('.abstract-text');
                    
                    toggle.addEventListener('click', function() {
                        if (toggle.dataset.action === 'show') {
                            abstractText.style.display = 'block';
                            toggle.innerHTML = `
                                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="currentColor" class="bi bi-chevron-up me-1" viewBox="0 0 16 16">
                                    <path fill-rule="evenodd" d="M7.646 4.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1-.708.708L8 5.707l-5.646 5.647a.5.5 0 0 1-.708-.708l6-6z"/>
                                </svg>
                                Hide abstract
                            `;
                            toggle.dataset.action = 'hide';
                        } else {
                            abstractText.style.display = 'none';
                            toggle.innerHTML = `
                                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="currentColor" class="bi bi-chevron-down me-1" viewBox="0 0 16 16">
                                    <path fill-rule="evenodd" d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/>
                                </svg>
                                Show abstract
                            `;
                            toggle.dataset.action = 'show';
                        }
                    });
                    
                    abstractsList.appendChild(abstractElement);
                });
                
                // If there are more abstracts to load, set up infinite scroll
                if (data.has_more) {
                    // Store the next page number as a data attribute on the abstracts container
                    abstractsList.dataset.nextPage = data.page + 1;
                    
                    // If not already set up, add scroll event listener for infinite scroll
                    if (!abstractsList.dataset.scrollListenerActive) {
                        abstractsList.dataset.scrollListenerActive = true;
                        
                        // Add scroll event listener to load more abstracts when scrolled to bottom
                        abstractsList.addEventListener('scroll', function() {
                            const scrollHeight = abstractsList.scrollHeight;
                            const scrollTop = abstractsList.scrollTop;
                            const clientHeight = abstractsList.clientHeight;
                            
                            // If scrolled near the bottom (within 100px) and not currently loading
                            if (scrollHeight - scrollTop - clientHeight < 100 && !document.getElementById('load-more-indicator')) {
                                const nextPage = parseInt(abstractsList.dataset.nextPage, 10);
                                if (nextPage) {
                                    loadDatabaseAbstracts(searchTerm, nextPage, true);
                                }
                            }
                        });
                    }
                } else {
                    // No more abstracts to load, remove next page data
                    delete abstractsList.dataset.nextPage;
                }
            })
            .catch(error => {
                // Remove loading indicator if appending
                const loadingIndicator = document.getElementById('load-more-indicator');
                if (loadingIndicator) loadingIndicator.remove();
                
                if (!append) {
                    abstractsList.innerHTML = `
                        <div class="text-center py-4">
                            <p class="text-danger small">Error loading abstracts: ${error.message}</p>
                        </div>
                    `;
                } else {
                    // If appending, show error at bottom
                    const errorElement = document.createElement('div');
                    errorElement.className = 'text-center py-2';
                    errorElement.innerHTML = `<p class="text-danger small">Error loading more abstracts: ${error.message}</p>`;
                    abstractsList.appendChild(errorElement);
                }
                console.error('Error loading abstracts:', error);
            });
    }
});
