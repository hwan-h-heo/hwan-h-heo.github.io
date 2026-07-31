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
    const initializedCardRoots = new WeakSet();

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
        const interactionTarget = image.closest('.post-preview')
            || image.closest('.blog-feature-card')
            || image.closest('.portfolio-blog-preview-link')
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

    function initializeBlogCardInteractions(container) {
        const rootElement = container && typeof container.addEventListener === 'function'
            ? container
            : document;
        if (initializedCardRoots.has(rootElement)) {
            return;
        }

        initializedCardRoots.add(rootElement);
        rootElement.addEventListener('click', (event) => {
            if (event.defaultPrevented || event.button !== 0) {
                return;
            }

            const card = event.target.closest('.post-preview, .blog-feature-card');
            if (!card || !rootElement.contains(card)) {
                return;
            }

            if (event.target.closest('a, button, input, textarea, select, label, [role="button"]')) {
                return;
            }

            const selection = window.getSelection?.();
            if (selection && !selection.isCollapsed) {
                return;
            }

            const primaryLink = card.querySelector('.post-title a, .blog-feature-copy h2 a, .post-card-cover');
            if (!primaryLink) {
                return;
            }

            if (event.metaKey || event.ctrlKey || event.shiftKey) {
                window.open(primaryLink.href, '_blank', 'noopener');
                return;
            }

            window.location.assign(primaryLink.href);
        });
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
                initializeBlogCardInteractions(document);
            }, { once: true });
        } else {
            initializeBlogCoverMedia(document);
            initializeBlogCardInteractions(document);
        }
    }

    return {
        getBlogCoverPreviewUrl,
        initializeBlogCardInteractions,
        initializeBlogCoverMedia,
        isAnimatedCover
    };
});
