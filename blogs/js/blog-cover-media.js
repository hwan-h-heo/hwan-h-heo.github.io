(function(root, factory) {
    const api = factory();

    if (typeof module === 'object' && module.exports) {
        module.exports = api;
    }

    if (root) {
        root.blogCoverMedia = api;
    }
})(typeof window !== 'undefined' ? window : null, function() {
    const initializedImages = new WeakSet();

    function sanitizeFilePart(value) {
        return String(value || '')
            .trim()
            .replace(/[^A-Za-z0-9_-]+/g, '-')
            .replace(/^-+|-+$/g, '');
    }

    function getBlogCoverPreviewUrl(postId, variant = 'cover') {
        const suffix = variant === 'portfolio' ? '-portfolio' : '';
        return `/assets/generated/blog-covers/${sanitizeFilePart(postId)}${suffix}.webp`;
    }

    function isAnimatedCover(source) {
        return /\.gif(?:[?#].*)?$/i.test(String(source || ''));
    }

    function initializeImage(image) {
        if (!image || initializedImages.has(image)) {
            return;
        }

        initializedImages.add(image);
        const autoplaySource = image.dataset.autoplaySrc;
        const animatedSource = autoplaySource || image.dataset.animatedSrc;
        if (!animatedSource || !isAnimatedCover(animatedSource)) {
            return;
        }

        const previewSource = image.dataset.previewSrc || image.getAttribute('src');
        const interactionTarget = image.closest('.post-card-cover')
            || image.closest('.blog-feature-cover')
            || image.closest('.portfolio-blog-preview-cover-link')
            || image.closest('a')
            || image;
        const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
        const hoverCapable = window.matchMedia('(hover: hover) and (pointer: fine)');
        let active = false;
        let animatedReady = false;
        let loader = null;

        const showAnimatedCover = () => {
            if (reducedMotion.matches) {
                return;
            }

            active = true;
            if (animatedReady) {
                image.src = animatedSource;
                return;
            }

            if (loader) {
                return;
            }

            interactionTarget.classList.add('is-cover-loading');
            loader = new Image();
            loader.decoding = 'async';
            loader.onload = () => {
                animatedReady = true;
                interactionTarget.classList.remove('is-cover-loading');
                if (active) {
                    image.src = animatedSource;
                }
            };
            loader.onerror = () => {
                interactionTarget.classList.remove('is-cover-loading');
            };
            loader.src = animatedSource;
        };

        const showPreviewCover = () => {
            active = false;
            image.src = previewSource;
        };

        if (autoplaySource) {
            let isNearViewport = false;
            const syncAutoplayState = () => {
                if (isNearViewport && !reducedMotion.matches) {
                    showAnimatedCover();
                } else {
                    showPreviewCover();
                }
            };

            if ('IntersectionObserver' in window) {
                const observer = new IntersectionObserver((entries) => {
                    isNearViewport = entries.some((entry) => entry.isIntersecting);
                    syncAutoplayState();
                }, {
                    rootMargin: '300px 0px',
                    threshold: 0.01
                });
                observer.observe(image);
            } else {
                isNearViewport = true;
                syncAutoplayState();
            }

            if (typeof reducedMotion.addEventListener === 'function') {
                reducedMotion.addEventListener('change', syncAutoplayState);
            }
            return;
        }

        interactionTarget.addEventListener('pointerenter', () => {
            if (hoverCapable.matches) {
                showAnimatedCover();
            }
        });
        interactionTarget.addEventListener('pointerleave', showPreviewCover);
        interactionTarget.addEventListener('focusin', (event) => {
            if (event.target.matches(':focus-visible')) {
                showAnimatedCover();
            }
        });
        interactionTarget.addEventListener('focusout', showPreviewCover);
    }

    function initializeBlogCoverMedia(container) {
        const rootElement = container && typeof container.querySelectorAll === 'function'
            ? container
            : document;
        rootElement.querySelectorAll('img[data-blog-cover]').forEach(initializeImage);
    }

    if (typeof document !== 'undefined') {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                initializeBlogCoverMedia(document);
            }, { once: true });
        } else {
            initializeBlogCoverMedia(document);
        }
    }

    return {
        getBlogCoverPreviewUrl,
        initializeBlogCoverMedia,
        isAnimatedCover
    };
});
