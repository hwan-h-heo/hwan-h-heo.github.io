// Code block copy button functionality
(function() {
    'use strict';

    // Add copy button to all code blocks
    function addCopyButtons() {
        const codeBlocks = document.querySelectorAll('pre code');

        codeBlocks.forEach((codeBlock) => {
            const pre = codeBlock.parentElement;

            // Skip if button already exists
            if (pre.querySelector('.copy-code-button')) {
                return;
            }

            // Create copy button
            const button = document.createElement('button');
            button.className = 'copy-code-button';
            button.innerHTML = '<i class="bi bi-clipboard"></i>';
            button.setAttribute('aria-label', 'Copy code to clipboard');
            button.setAttribute('title', 'Copy code');

            // Add click event
            button.addEventListener('click', async () => {
                const code = codeBlock.textContent;

                try {
                    await navigator.clipboard.writeText(code);

                    // Show success feedback
                    button.innerHTML = '<i class="bi bi-check2"></i>';
                    button.classList.add('copied');

                    // Reset after 2 seconds
                    setTimeout(() => {
                        button.innerHTML = '<i class="bi bi-clipboard"></i>';
                        button.classList.remove('copied');
                    }, 2000);
                } catch (err) {
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = code;
                    textArea.style.position = 'fixed';
                    textArea.style.left = '-999999px';
                    document.body.appendChild(textArea);
                    textArea.select();

                    try {
                        document.execCommand('copy');
                        button.innerHTML = '<i class="bi bi-check2"></i>';
                        button.classList.add('copied');

                        setTimeout(() => {
                            button.innerHTML = '<i class="bi bi-clipboard"></i>';
                            button.classList.remove('copied');
                        }, 2000);
                    } catch (err) {
                        console.error('Failed to copy code:', err);
                        button.innerHTML = '<i class="bi bi-x"></i>';

                        setTimeout(() => {
                            button.innerHTML = '<i class="bi bi-clipboard"></i>';
                        }, 2000);
                    } finally {
                        document.body.removeChild(textArea);
                    }
                }
            });

            // Make pre relative for absolute positioning
            pre.style.position = 'relative';

            // Add button to pre element
            pre.appendChild(button);
        });
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', addCopyButtons);
    } else {
        addCopyButtons();
    }

    // Re-add buttons after dynamic content loads
    // Useful for blog posts loaded via AJAX
    window.addCopyButtonsToCode = addCopyButtons;

    // Observer for dynamically added code blocks
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                addCopyButtons();
            }
        });
    });

    // Start observing the document body for changes
    if (document.body) {
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    } else {
        document.addEventListener('DOMContentLoaded', () => {
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        });
    }
})();
