// Scroll Progress Bar
(function() {
    'use strict';

    function initScrollProgress() {
        // Create progress bar element
        const progressBar = document.createElement('div');
        progressBar.className = 'scroll-progress-bar';
        progressBar.innerHTML = '<div class="scroll-progress-fill"></div>';

        // Insert at the beginning of body
        document.body.insertBefore(progressBar, document.body.firstChild);

        const progressFill = progressBar.querySelector('.scroll-progress-fill');

        // Update progress on scroll
        function updateProgress() {
            const windowHeight = window.innerHeight;
            const documentHeight = document.documentElement.scrollHeight;
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

            // Calculate progress percentage
            const scrollableHeight = documentHeight - windowHeight;
            const progress = scrollableHeight > 0 ? (scrollTop / scrollableHeight) * 100 : 0;

            // Update progress bar width
            progressFill.style.width = Math.min(progress, 100) + '%';
        }

        // Listen to scroll events with throttling for performance
        let ticking = false;
        window.addEventListener('scroll', function() {
            if (!ticking) {
                window.requestAnimationFrame(function() {
                    updateProgress();
                    ticking = false;
                });
                ticking = true;
            }
        });

        // Initial update
        updateProgress();

        // Update on window resize
        window.addEventListener('resize', updateProgress);
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initScrollProgress);
    } else {
        initScrollProgress();
    }
})();
