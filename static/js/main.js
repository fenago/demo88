// Main JavaScript for Unsupervised Learning Analyzer

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Handle range input display
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(function(input) {
        input.addEventListener('input', function() {
            this.nextElementSibling.value = this.value;
        });
    });
    
    // Synchronize data URL between tabs
    const dataUrlInputs = [document.getElementById('data-url'), document.getElementById('elbow-data-url')];
    dataUrlInputs.forEach(function(input) {
        if (input) {
            input.addEventListener('change', function() {
                const value = this.value;
                dataUrlInputs.forEach(function(otherInput) {
                    if (otherInput && otherInput !== input) {
                        otherInput.value = value;
                    }
                });
            });
        }
    });
    
    // Synchronize analysis type between tabs
    const analysisTypeInputs = [document.getElementById('analysis-type'), document.getElementById('elbow-analysis-type')];
    analysisTypeInputs.forEach(function(input) {
        if (input) {
            input.addEventListener('change', function() {
                const value = this.value;
                analysisTypeInputs.forEach(function(otherInput) {
                    if (otherInput && otherInput !== input) {
                        otherInput.value = value;
                    }
                });
            });
        }
    });
    
    // Add loading state to forms
    const forms = document.querySelectorAll('form');
    forms.forEach(function(form) {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                submitBtn.disabled = true;
                
                // Re-enable after 30 seconds in case of error
                setTimeout(function() {
                    submitBtn.innerHTML = originalText;
                    submitBtn.disabled = false;
                }, 30000);
            }
        });
    });
});
