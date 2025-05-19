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
    const selectAllCheckbox = document.getElementById('select-all-abstracts-checkbox');
    const downloadSelectedBtn = document.getElementById('download-selected-btn');
    const selectedCountSpan = document.getElementById('selected-count');
    const successNotificationDiv = document.getElementById('download-success-notification');
    
    let selectedDOIs = new Set();

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
        let currentScrollTop = 0;
        if (append && abstractsList.scrollTop > 0) {
            currentScrollTop = abstractsList.scrollTop; // Preserve scroll position for append
        } else if (!append) {
            abstractsList.innerHTML = ''; // Clear only if not appending (page 1 or search)
        }


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
                    // Ensure abstract.id is filesystem-safe if used directly in IDs.
                    // Using a sanitized DOI or a simple counter might be safer if abstract.id can have special chars.
                    const uniqueIdSuffix = abstract.doi ? abstract.doi.replace(/[^a-zA-Z0-9-_]/g, '_') : abstract.id.replace(/[^a-zA-Z0-9-_]/g, '_');
                    abstractElement.id = `abstract-item-${uniqueIdSuffix}`; 
                    
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
                    
                    // Downloaded indicator / Download button
                    let downloadDisplay = '';
                    let checkboxDisplay = '';

                    if (abstract.doi) {
                        checkboxDisplay = `
                            <input class="form-check-input abstract-checkbox" type="checkbox" value="${abstract.doi}" id="checkbox-${uniqueIdSuffix}" data-doi="${abstract.doi}" ${selectedDOIs.has(abstract.doi) ? 'checked' : ''}>
                        `;
                        if (abstract.is_downloaded) {
                            downloadDisplay = `
                                <span class="ms-2" title="TXT File Downloaded">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 16 16">
                                        <circle cx="8" cy="8" r="7" fill="white" stroke="black" stroke-width="1"/>
                                        <path fill-rule="evenodd" d="M10.97 4.97a.75.75 0 0 1 1.07 1.05l-3.99 4.99a.75.75 0 0 1-1.08.02L4.324 8.384a.75.75 0 1 1 1.06-1.06l2.094 2.093 3.473-4.425a.267.267 0 0 1 .02-.022z" fill="black"/>
                                    </svg>
                                </span>`;
                        } else {
                            downloadDisplay = `
                                <button class="btn btn-outline-secondary btn-sm py-0 px-1 ms-2 download-paper-btn" data-doi="${abstract.doi}" title="Download TXT">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="currentColor" class="bi bi-download" viewBox="0 0 16 16">
                                        <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                                        <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
                                    </svg>
                                </button>`;
                        }
                    }


                    // Format APA citation with reference-style formatting
                    abstractElement.innerHTML = `
                        <div class="d-flex align-items-start reference-style-citation">
                            ${checkboxDisplay}
                            <div class="flex-grow-1">
                                <span class="authors">${authorsText}</span>
                                <span class="year">${abstract.year ? ` (${abstract.year}).` : ''}</span>
                                <span class="title"> ${abstract.title || 'Untitled'}.</span>
                                <span class="source"> <em>${abstract.source_title || ''}</em></span>
                                ${abstract.cited_by ? `<span class="cited-by">. Cited by: ${abstract.cited_by}</span>` : ''}
                                ${abstract.doi ? `<span class="doi-link">. <a href="https://doi.org/${abstract.doi}" target="_blank" class="doi_url">https://doi.org/${abstract.doi}</a></span>` : ''}
                                ${downloadDisplay}
                            </div>
                        </div>
                        <div class="abstract-text" style="margin-left: 25px;">${abstract.document || 'No abstract available'}</div>
                        <span class="abstract-toggle" data-action="show" style="margin-left: 25px;">
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
                                    <path fill-rule="evenodd" d="M7.646 4.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1-.708.708L8 5.707l-5.646 5.647a.5.5 0 1 0-.708.708l3 3z"/>
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
                
                if (currentScrollTop > 0) { // Restore scroll position if appending
                    abstractsList.scrollTop = currentScrollTop;
                }
                updateSelectedCount(); // Update count after loading
                updateSelectAllCheckboxState(); // Update select-all checkbox based on current items

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
                                    // Fetch the CURRENT search term from the input field
                                    const currentSearchTerm = abstractSearch.value.trim();
                                    loadDatabaseAbstracts(currentSearchTerm, nextPage, true);
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

    // Event listener for download buttons
    abstractsList.addEventListener('click', function(event) {
        const downloadButton = event.target.closest('.download-paper-btn');
        if (downloadButton) {
            event.preventDefault();
            const doi = downloadButton.dataset.doi;
            if (doi) {
                handleDownloadClick(doi, downloadButton);
            }
        }
    });

    // Event listener for individual checkboxes
    abstractsList.addEventListener('change', function(event) {
        if (event.target.classList.contains('abstract-checkbox')) {
            const doi = event.target.dataset.doi;
            if (event.target.checked) {
                selectedDOIs.add(doi);
            } else {
                selectedDOIs.delete(doi);
            }
            updateSelectedCount();
            updateSelectAllCheckboxState();
        }
    });

    // Event listener for "Select All" checkbox
    selectAllCheckbox.addEventListener('change', function() {
        const checkboxes = abstractsList.querySelectorAll('.abstract-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.checked = selectAllCheckbox.checked;
            const doi = checkbox.dataset.doi;
            if (selectAllCheckbox.checked) {
                selectedDOIs.add(doi);
            } else {
                selectedDOIs.delete(doi);
            }
        });
        updateSelectedCount();
    });
    
    // Function to update the "Select All" checkbox based on visible items
    function updateSelectAllCheckboxState() {
        const checkboxes = abstractsList.querySelectorAll('.abstract-checkbox');
        if (checkboxes.length === 0) {
            selectAllCheckbox.checked = false;
            selectAllCheckbox.indeterminate = false;
            return;
        }
        const checkedCount = Array.from(checkboxes).filter(cb => cb.checked).length;
        if (checkedCount === 0) {
            selectAllCheckbox.checked = false;
            selectAllCheckbox.indeterminate = false;
        } else if (checkedCount === checkboxes.length) {
            selectAllCheckbox.checked = true;
            selectAllCheckbox.indeterminate = false;
        } else {
            selectAllCheckbox.checked = false;
            selectAllCheckbox.indeterminate = true;
        }
    }


    function updateSelectedCount() {
        const count = selectedDOIs.size;
        selectedCountSpan.textContent = count;
        downloadSelectedBtn.disabled = count === 0;
    }

    function hideSuccessNotification() {
        if (successNotificationDiv) {
            successNotificationDiv.classList.add('d-none');
            successNotificationDiv.textContent = '';
        }
    }

    function showSuccessNotification(message, duration = 5000) {
        if (successNotificationDiv) {
            successNotificationDiv.textContent = message;
            successNotificationDiv.classList.remove('d-none');
            setTimeout(() => {
                hideSuccessNotification();
            }, duration);
        }
    }

    downloadSelectedBtn.addEventListener('click', function() {
        if (selectedDOIs.size === 0) {
            alert("No papers selected for download."); // This alert can remain as it's a user action validation
            return;
        }
        hideSuccessNotification(); // Hide any previous success message
        handleBatchDownload([...selectedDOIs]);
    });

    function handleBatchDownload(doisToDownload) {
        downloadSelectedBtn.disabled = true;
        const originalButtonText = downloadSelectedBtn.innerHTML;
        downloadSelectedBtn.innerHTML = `
            <div class="spinner-border spinner-border-sm me-1" role="status" style="width: 0.8rem; height: 0.8rem;">
                <span class="visually-hidden">Loading...</span>
            </div> Downloading ${doisToDownload.length}...`;

        fetch('/abstracts/download_multiple_papers', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json', // Sending as JSON
            },
            body: JSON.stringify({ dois: doisToDownload }) // Send DOIs as a JSON array
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success' && data.job_id) {
                // Use a generic poller, or adapt pollDownloadJobStatus if more specific feedback is needed
                pollBatchDownloadJobStatus(data.job_id, doisToDownload.length, downloadSelectedBtn, originalButtonText);
            } else {
                showAbstractError(`Failed to start batch download: ${data.message || 'Unknown error'}`, downloadSelectedBtn, originalButtonText);
            }
        })
        .catch(err => {
            showAbstractError(`Network error starting batch download: ${err.message}`, downloadSelectedBtn, originalButtonText);
        });
    }

    function pollBatchDownloadJobStatus(jobId, numPapers, buttonElement, originalButtonText) {
        const checkStatus = () => {
            fetch(`/abstracts/status/${jobId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'Completed') {
                        const currentSearchTerm = abstractSearch.value.trim();
                        loadDatabaseAbstracts(currentSearchTerm, 1, false); // Reload list
                        selectedDOIs.clear(); // Clear selection
                        updateSelectedCount();
                        updateSelectAllCheckboxState();
                        buttonElement.innerHTML = originalButtonText; // Restore button text
                        buttonElement.disabled = selectedDOIs.size === 0; // Re-evaluate disabled state
                        showSuccessNotification(`${numPapers} paper(s) download process completed. List refreshed.`);
                        console.log(`Batch download job ${jobId} completed.`);
                    } else if (data.status === 'Error') {
                        showAbstractError(`Batch download failed: ${data.error || 'Unknown error'}`, buttonElement, originalButtonText);
                        console.error(`Batch download job ${jobId} failed.`);
                    } else if (data.status === 'Processing' || data.status === 'Starting') {
                        buttonElement.innerHTML = `
                            <div class="spinner-border spinner-border-sm me-1" role="status" style="width: 0.8rem; height: 0.8rem;"></div>
                            ${data.progress || `Downloading ${numPapers}...`}`;
                        setTimeout(checkStatus, 3000); // Continue polling
                    } else if (data.status === 'not_found') {
                        showAbstractError(`Batch download job ${jobId} not found.`, buttonElement, originalButtonText);
                        console.error(`Batch download job ${jobId} not found.`);
                    } else {
                        setTimeout(checkStatus, 3000);
                    }
                })
                .catch(err => {
                    showAbstractError(`Error checking batch download status: ${err.message}`, buttonElement, originalButtonText);
                    console.error(`Error polling batch download job ${jobId}:`, err);
                });
        };
        checkStatus();
    }


    function handleDownloadClick(doi, buttonElement) {
        buttonElement.disabled = true;
        buttonElement.innerHTML = `
            <div class="spinner-border spinner-border-sm" role="status" style="width: 0.8rem; height: 0.8rem;">
                <span class="visually-hidden">Loading...</span>
            </div>`;
        hideSuccessNotification(); // Hide any previous success message

        fetch('/abstracts/download_paper', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({ 'doi': doi })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success' && data.job_id) {
                pollDownloadJobStatus(data.job_id, doi, buttonElement);
            } else {
                showAbstractError(`Failed to start download for ${doi}: ${data.message || 'Unknown error'}`, buttonElement);
            }
        })
        .catch(err => {
            showAbstractError(`Network error starting download for ${doi}: ${err.message}`, buttonElement);
        });
    }

    function pollDownloadJobStatus(jobId, doi, buttonElement) {
        const checkStatus = () => {
            fetch(`/abstracts/status/${jobId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'Completed') {
                        const currentSearchTerm = abstractSearch.value.trim();
                        loadDatabaseAbstracts(currentSearchTerm, 1, false); 
                        if (selectedDOIs.has(doi)) {
                            selectedDOIs.delete(doi);
                            updateSelectedCount();
                            updateSelectAllCheckboxState();
                        }
                        showSuccessNotification(`Paper (DOI: ${doi}) downloaded successfully. List refreshed.`);
                        console.log(`Individual download job ${jobId} for DOI ${doi} completed.`);
                    } else if (data.status === 'Error') {
                        showAbstractError(`Download failed for ${doi}: ${data.error || 'Unknown error'}`, buttonElement);
                        console.error(`Individual download job ${jobId} for DOI ${doi} failed.`);
                    } else if (data.status === 'Processing' || data.status === 'Starting') {
                        setTimeout(checkStatus, 3000); // Continue polling
                    } else if (data.status === 'not_found') {
                        showAbstractError(`Download job for ${doi} not found.`, buttonElement);
                        console.error(`Individual download job ${jobId} for DOI ${doi} not found.`);
                    } else {
                        setTimeout(checkStatus, 3000);
                    }
                })
                .catch(err => {
                    showAbstractError(`Error checking download status for ${doi}: ${err.message}`, buttonElement);
                    console.error(`Error polling individual download job ${jobId} for DOI ${doi}:`, err);
                });
        };
        checkStatus(); // Start polling
    }

    function showAbstractError(message, buttonElement, originalButtonText = null) {
        hideSuccessNotification(); // Hide success message if an error occurs
        if (buttonElement) {
            if (buttonElement.id === 'download-selected-btn') {
                // For the batch download button
                buttonElement.innerHTML = originalButtonText || `Download Selected (<span id="selected-count">${selectedDOIs.size}</span>)`;
                buttonElement.disabled = selectedDOIs.size === 0; 
                // Ensure the count in the button text is also updated if not part of originalButtonText
                if (originalButtonText && !originalButtonText.includes('selected-count')) {
                    selectedCountSpan.textContent = selectedDOIs.size; // selectedCountSpan is global
                } else if (!originalButtonText) {
                     document.getElementById('selected-count').textContent = selectedDOIs.size; // If no original text, update count
                }
            } else if (buttonElement.classList.contains('download-paper-btn')) {
                // For an individual download button
                buttonElement.disabled = false;
                buttonElement.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="currentColor" class="bi bi-download" viewBox="0 0 16 16">
                        <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                        <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
                    </svg>`;
            }
        }
        // Display error message
        // The alert() is synchronous and blocks UI. If "freezing" persists after this change,
        // consider replacing alert() with a non-blocking notification (e.g., a toast).
        alert(`Error: ${message}`); 
        console.error("Abstracts pane error:", message);
    }

});
