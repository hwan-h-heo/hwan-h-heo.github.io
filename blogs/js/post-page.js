(function() {
    const icons = window.SiteIcons;

    function initializeMathRendering() {
        if (typeof renderMathInElement === 'undefined') {
            return;
        }

        renderMathInElement(document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false },
                { left: '\\[', right: '\\]', display: true }
            ],
            throwOnError: false
        });
    }

    function setupShareButton() {
        const copyButton = document.getElementById('copyButton');
        const shareModal = document.getElementById('myshare_modal');

        if (!copyButton || !shareModal) {
            return;
        }

        copyButton.setAttribute('type', 'button');
        copyButton.setAttribute('aria-label', 'Copy post link');
        copyButton.setAttribute('title', 'Copy link');
        copyButton.innerHTML = icons.render('link-45deg');

        shareModal.setAttribute('role', 'status');
        shareModal.setAttribute('aria-live', 'polite');
        shareModal.setAttribute('aria-hidden', 'true');

        const modalContent = shareModal.querySelector('.share_modal-content');
        const modalLabels = window.blogPostPageConfig?.lang === 'kor'
            ? {
                copied: '링크 복사됨',
                ready: '글 주소를 바로 공유할 수 있습니다.'
            }
            : {
                copied: 'Link copied',
                ready: 'Post URL is ready to share.'
            };

        if (modalContent) {
            modalContent.innerHTML = `
                <button class="share_modal_close" type="button" aria-label="Dismiss">
                    ${icons.render('x-lg')}
                </button>
                <span class="share_modal-icon" aria-hidden="true">
                    ${icons.render('check2')}
                </span>
                <span class="share_modal-message">
                    <strong>${modalLabels.copied}</strong>
                    <small>${modalLabels.ready}</small>
                </span>
                <div class="copy_indicator-container">
                    <div class="copy_indicator" id="share_modalIndicator"></div>
                </div>
            `;
        }

        const closeModal = shareModal.querySelector('.share_modal_close');
        const indicator = document.getElementById('share_modalIndicator');

        if (!indicator) {
            return;
        }

        let animationId;

        function updateShareButtonVisibility() {
            const headerHeight = document.querySelector('.masthead')?.offsetHeight || 300;
            copyButton.style.display = window.innerWidth > 1280 && window.scrollY > headerHeight ? 'inline-flex' : 'none';
        }

        function animateIndicator() {
            let startTime = null;
            const duration = 1500;

            function step(timestamp) {
                if (!startTime) {
                    startTime = timestamp;
                }

                const progress = Math.min((timestamp - startTime) / duration, 1);
                indicator.style.width = `${(1 - progress) * 100}%`;

                if (progress < 1) {
                    animationId = requestAnimationFrame(step);
                } else {
                    shareModal.classList.remove('is-visible');
                    shareModal.setAttribute('aria-hidden', 'true');
                    shareModal.style.display = 'none';
                }
            }

            cancelAnimationFrame(animationId);
            animationId = requestAnimationFrame(step);
        }

        function closeShareModal() {
            cancelAnimationFrame(animationId);
            shareModal.classList.remove('is-visible');
            shareModal.setAttribute('aria-hidden', 'true');
            shareModal.style.display = 'none';
            indicator.style.width = '0%';
        }

        document.addEventListener('scroll', updateShareButtonVisibility);
        window.addEventListener('resize', updateShareButtonVisibility);

        copyButton.addEventListener('click', function() {
            const url = new URL(window.location.href);
            url.hash = '';

            navigator.clipboard.writeText(url.href).then(function() {
                shareModal.style.display = 'block';
                shareModal.setAttribute('aria-hidden', 'false');
                window.requestAnimationFrame(() => {
                    shareModal.classList.add('is-visible');
                });
                indicator.style.width = '100%';
                animateIndicator();
            }).catch(function(error) {
                console.error('Link copy failed:', error);
            });
        });

        if (closeModal) {
            closeModal.addEventListener('click', closeShareModal);
        }

        window.addEventListener('click', function(event) {
            if (event.target === shareModal) {
                closeShareModal();
            }
        });

        updateShareButtonVisibility();
    }

    function setupImageLightboxes() {
        if (typeof HTMLDialogElement === 'undefined' || !HTMLDialogElement.prototype.showModal) {
            return;
        }

        const labels = window.blogPostPageConfig?.lang === 'kor'
            ? { enlarge: '이미지 크게 보기', close: '확대 이미지 닫기', original: '원본 크기', fit: '화면에 맞추기' }
            : { enlarge: 'Enlarge image', close: 'Close enlarged image', original: 'Actual size', fit: 'Fit to screen' };
        const controllers = new Map();

        function controllerFor(dialog) {
            if (controllers.has(dialog)) {
                return controllers.get(dialog);
            }
            const closeButton = dialog.querySelector('[data-image-lightbox-close]');
            let returnFocus = null;

            function closeLightbox() {
                if (dialog.open) {
                    dialog.close();
                }
            }

            closeButton?.addEventListener('click', closeLightbox);
            dialog.addEventListener('click', (event) => {
                if (event.target === dialog || event.target.hasAttribute('data-image-lightbox-stage')) {
                    closeLightbox();
                }
            });
            dialog.addEventListener('close', () => {
                document.documentElement.classList.remove('post-image-lightbox-open');
                if (returnFocus?.isConnected) {
                    returnFocus.focus({ preventScroll: true });
                }
            });

            const controller = {
                open(trigger) {
                    if (dialog.open) {
                        return;
                    }
                    returnFocus = trigger;
                    document.documentElement.classList.add('post-image-lightbox-open');
                    dialog.showModal();
                    closeButton?.focus({ preventScroll: true });
                }
            };
            controllers.set(dialog, controller);
            return controller;
        }

        document.querySelectorAll('[data-image-lightbox-target]').forEach((trigger) => {
            const targetSelector = trigger.getAttribute('data-image-lightbox-target');
            const dialog = targetSelector ? document.querySelector(targetSelector) : null;
            if (dialog instanceof HTMLDialogElement) {
                trigger.addEventListener('click', () => controllerFor(dialog).open(trigger));
            }
        });

        const images = Array.from(document.querySelectorAll('.main-content img')).filter((image) => {
            return !image.closest('a, button, dialog, [role="button"], [aria-hidden="true"], [data-image-lightbox="off"]');
        });
        if (!images.length) {
            return;
        }

        const dialog = document.createElement('dialog');
        dialog.id = 'post-image-lightbox';
        dialog.className = 'post-image-lightbox post-image-lightbox--auto';
        dialog.setAttribute('aria-label', labels.enlarge);
        dialog.innerHTML = `
            <div class="post-image-lightbox-frame">
                <div class="post-image-lightbox-toolbar">
                    <button class="post-image-lightbox-zoom" type="button" aria-pressed="false">
                        ${icons.render('fullscreen')}
                    </button>
                    <button class="post-image-lightbox-close" type="button" data-image-lightbox-close>
                        ${icons.render('x-lg')}
                    </button>
                </div>
                <div class="post-image-lightbox-stage" data-image-lightbox-stage></div>
                <div class="post-image-lightbox-caption" id="post-image-lightbox-caption">
                    <span class="post-image-lightbox-index"></span>
                    <span class="post-image-lightbox-title"></span>
                </div>
            </div>`;
        document.body.appendChild(dialog);

        const controller = controllerFor(dialog);
        const imageElement = document.createElement('img');
        imageElement.className = 'post-image-lightbox-image';
        imageElement.decoding = 'async';
        const captionElement = dialog.querySelector('.post-image-lightbox-caption');
        const indexElement = dialog.querySelector('.post-image-lightbox-index');
        const titleElement = dialog.querySelector('.post-image-lightbox-title');
        const stage = dialog.querySelector('.post-image-lightbox-stage');
        const zoomButton = dialog.querySelector('.post-image-lightbox-zoom');
        const zoomIcon = zoomButton.querySelector('.site-icon');
        dialog.querySelector('[data-image-lightbox-close]').setAttribute('aria-label', labels.close);

        function updateZoomAvailability() {
            if (!dialog.open) {
                return;
            }
            const bounds = imageElement.getBoundingClientRect();
            const canZoom = dialog.classList.contains('is-zoomed')
                || imageElement.naturalWidth > bounds.width + 1
                || imageElement.naturalHeight > bounds.height + 1;
            zoomButton.disabled = !canZoom;
            dialog.classList.toggle('can-zoom', canZoom);
        }

        function setZoom(zoomed) {
            dialog.classList.toggle('is-zoomed', zoomed);
            zoomButton.setAttribute('aria-pressed', String(zoomed));
            const zoomLabel = zoomed ? labels.fit : labels.original;
            zoomButton.setAttribute('aria-label', zoomLabel);
            zoomButton.setAttribute('title', zoomLabel);
            icons.set(zoomIcon, zoomed ? 'fullscreen-exit' : 'fullscreen');
            stage.scrollTo(0, 0);
            updateZoomAvailability();
        }

        function toggleZoom() {
            if (!zoomButton.disabled) {
                setZoom(!dialog.classList.contains('is-zoomed'));
            }
        }

        zoomButton.addEventListener('click', toggleZoom);
        imageElement.addEventListener('click', toggleZoom);
        imageElement.addEventListener('load', updateZoomAvailability);
        window.addEventListener('resize', updateZoomAvailability);

        images.forEach((image) => {
            const description = image.alt.trim();
            image.setAttribute('role', 'button');
            image.setAttribute('tabindex', '0');
            image.setAttribute('aria-haspopup', 'dialog');
            image.setAttribute('aria-controls', dialog.id);
            image.setAttribute('aria-label', description ? `${labels.enlarge}: ${description}` : labels.enlarge);
            image.setAttribute('data-image-lightbox-target', `#${dialog.id}`);
            image.setAttribute('data-image-lightbox-auto-trigger', '');

            function openImage() {
                imageElement.src = image.currentSrc || image.src;
                imageElement.alt = image.alt;
                if (!imageElement.isConnected) {
                    stage.appendChild(imageElement);
                }
                const caption = image.closest('figure')?.querySelector('figcaption');
                const index = caption?.querySelector('.post-figure-index')?.innerText || '';
                const captionText = caption?.querySelector('.post-figure-caption')?.innerText || caption?.innerText;
                indexElement.textContent = index;
                indexElement.hidden = !index;
                titleElement.textContent = (captionText || description).replace(/\s+/g, ' ').trim();
                captionElement.hidden = !titleElement.textContent;
                if (titleElement.textContent) {
                    dialog.setAttribute('aria-describedby', captionElement.id);
                } else {
                    dialog.removeAttribute('aria-describedby');
                }
                dialog.setAttribute('aria-label', description || labels.enlarge);
                controller.open(image);
                setZoom(false);
            }

            image.addEventListener('click', openImage);
            image.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    openImage();
                }
            });
        });
    }

    function initializeTOC() {
        const toc = document.querySelector('.toc');
        if (!toc) {
            return;
        }

        const persistentTocMedia = window.matchMedia('(min-width: 1600px)');
        let tocItems = [];

        function updateTocVisibility() {
            const headerHeight = document.querySelector('.masthead')?.offsetHeight || 300;
            const isVisible = persistentTocMedia.matches && window.scrollY > headerHeight;
            toc.classList.toggle('is-visible', isVisible);
            toc.setAttribute('aria-hidden', isVisible ? 'false' : 'true');
        }

        function getDirectAnchor(item) {
            return Array.from(item.children).find((child) => child.tagName === 'A') || item.querySelector('a');
        }

        function normalizeLegacyTocLists() {
            toc.querySelectorAll('ul > ul, ul > ol, ol > ul, ol > ol').forEach((list) => {
                const parentItem = list.previousElementSibling;
                if (!parentItem?.matches('li')) {
                    return;
                }

                parentItem.appendChild(list);
                list.classList.add('toc-sublist');
            });
        }

        function initTocItems() {
            tocItems = Array.from(toc.querySelectorAll('li')).map((item) => {
                const anchor = getDirectAnchor(item);
                const href = anchor?.getAttribute('href');
                if (!href || href === '#') {
                    return null;
                }

                const parentItem = item.parentElement?.closest('li') || null;
                const level = item.dataset.level || (parentItem ? '3' : '2');
                item.dataset.level = level;
                item.classList.add('toc-item', `toc-item-level-${level}`);
                anchor.classList.add('toc-link');

                return {
                    listItem: item,
                    parentItem,
                    target: document.getElementById(href.slice(1))
                };
            }).filter((item) => item && item.target);
        }

        function updateSublistHeights() {
            toc.querySelectorAll('.toc-sublist, ul ul').forEach((list) => {
                list.style.setProperty('--toc-sublist-height', `${list.scrollHeight}px`);
            });
        }

        function syncToc() {
            const windowHeight = window.innerHeight;
            const activationOffset = Math.min(220, Math.max(120, windowHeight * 0.32));
            let currentSection = null;

            tocItems.forEach((item) => {
                const targetBounds = item.target.getBoundingClientRect();
                if (targetBounds.top <= activationOffset) {
                    currentSection = item;
                }
            });

            tocItems.forEach((item) => {
                item.listItem.classList.remove('active', 'is-current', 'is-parent', 'is-expanded');
            });

            if (!currentSection) {
                return;
            }

            currentSection.listItem.classList.add('active', 'is-current', 'is-expanded');

            let parentItem = currentSection.parentItem;
            while (parentItem) {
                parentItem.classList.add('is-parent', 'is-expanded');
                parentItem = parentItem.parentElement?.closest('li') || null;
            }
        }

        document.addEventListener('scroll', updateTocVisibility, { passive: true });
        window.addEventListener('scroll', syncToc, false);
        window.addEventListener('resize', () => {
            updateTocVisibility();
            updateSublistHeights();
        });
        persistentTocMedia.addEventListener('change', updateTocVisibility);
        updateTocVisibility();
        normalizeLegacyTocLists();
        initTocItems();
        updateSublistHeights();
        if (document.fonts?.ready) {
            document.fonts.ready.then(updateSublistHeights).catch(() => {});
        }
        syncToc();
    }

    function initializeSpecialViewers() {
        const postId = window.blogPostPageConfig?.postId;
        const runtimeFeatures = window.blogPostPageConfig?.runtimeFeatures || {};

        if (runtimeFeatures.gaussianSplats || postId === '240917_3djs') {
            import('/blogs/3DViewer/js/gaussian_viewer.js').then((module) => {
                module.initGaussianViewer();
            }).catch((error) => {
                console.error('Failed to load Gaussian viewer:', error);
            });
        }

        if (runtimeFeatures.simpleModelViewer || postId === '250310_model_viewer') {
            import('/js/simple-model-viewer.js').then((module) => {
                module.initGaussianViewer();
            }).catch((error) => {
                console.error('Failed to load model viewer:', error);
            });
        }
    }

    function setupCompactContents() {
        const contents = document.querySelector('.post-contents');
        const item = document.querySelector('.post-nav-contents-item');
        const summary = contents?.querySelector('summary');
        if (!contents || !item || !summary) {
            return;
        }

        const compactMedia = window.matchMedia('(max-width: 767.98px)');
        const originalPosition = document.createComment('Inline article contents');
        contents.before(originalPosition);
        const label = document.createElement('span');
        label.className = 'post-contents-label';
        label.textContent = summary.textContent;
        summary.setAttribute('aria-label', label.textContent);
        summary.replaceChildren(label);
        summary.insertAdjacentHTML('afterbegin', icons.render('list-ul'));

        function syncPosition() {
            const hadFocus = contents.contains(document.activeElement);
            contents.open = false;
            contents.classList.toggle('post-contents--nav', compactMedia.matches);
            item.hidden = !compactMedia.matches;
            if (compactMedia.matches) {
                item.appendChild(contents);
            } else {
                originalPosition.after(contents);
            }
            if (hadFocus) {
                summary.focus({ preventScroll: true });
            }
        }

        contents.addEventListener('toggle', () => {
            if (compactMedia.matches && contents.open) {
                const labs = document.querySelector('.blog-labs');
                if (labs) {
                    labs.open = false;
                }
            }
        });
        contents.addEventListener('click', (event) => {
            const link = event.target.closest('a[href^="#"]');
            if (!compactMedia.matches || !link) {
                return;
            }
            contents.open = false;
            const target = document.getElementById(link.hash.slice(1));
            if (target) {
                target.setAttribute('tabindex', '-1');
                requestAnimationFrame(() => target.focus({ preventScroll: true }));
            }
        });
        document.addEventListener('pointerdown', (event) => {
            if (compactMedia.matches && !contents.contains(event.target)) {
                contents.open = false;
            }
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && compactMedia.matches && contents.open) {
                event.preventDefault();
                contents.open = false;
                summary.focus({ preventScroll: true });
            }
        });
        contents.addEventListener('focusout', () => {
            requestAnimationFrame(() => {
                if (compactMedia.matches && !contents.contains(document.activeElement)) {
                    contents.open = false;
                }
            });
        });
        compactMedia.addEventListener('change', syncPosition);
        syncPosition();
    }

    function syncLanguagePreference() {
        const config = window.blogPostPageConfig;
        if (!config?.lang) {
            return;
        }

        document.documentElement.lang = config.lang === 'kor' ? 'ko' : 'en';

        try {
            // Opening an English-only fallback preserves the preferred listing language.
            const preferredLanguage = localStorage.getItem('language');
            if (config.alternateHref || !['eng', 'kor'].includes(preferredLanguage)) {
                localStorage.setItem('language', config.lang);
            }
        } catch (error) {
            console.warn('Failed to persist language preference:', error);
        }

        document.querySelectorAll('[data-language-target]').forEach((link) => {
            link.addEventListener('click', () => {
                const targetLang = link.dataset.languageTarget;
                if (!targetLang) {
                    return;
                }

                try {
                    localStorage.setItem('language', targetLang);
                } catch (error) {
                    console.warn('Failed to persist language preference:', error);
                }
            });
        });
    }

    function setupPostNavPanel() {
        const nav = document.getElementById('mainNav');
        const toggle = nav?.querySelector('[data-nav-toggle]');
        const panel = nav?.querySelector('.post-nav-panel');

        if (!toggle || !panel) {
            return;
        }

        function setExpanded(expanded) {
            panel.classList.toggle('is-open', expanded);
            toggle.setAttribute('aria-expanded', String(expanded));
            toggle.setAttribute(
                'aria-label',
                expanded ? toggle.dataset.closeLabel : toggle.dataset.openLabel
            );
        }

        toggle.addEventListener('click', () => {
            setExpanded(!panel.classList.contains('is-open'));
        });

        panel.querySelectorAll('a, button').forEach((item) => {
            item.addEventListener('click', () => {
                if (window.matchMedia('(max-width: 991.98px)').matches) {
                    setExpanded(false);
                }
            });
        });

        window.addEventListener('resize', () => {
            if (!window.matchMedia('(max-width: 991.98px)').matches) {
                setExpanded(false);
            }
        });

        document.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape' || !panel.classList.contains('is-open')) {
                return;
            }
            setExpanded(false);
            toggle.focus();
        });
    }

    function setupAutoRevealNav() {
        const nav = document.getElementById('mainNav');
        if (!nav) {
            return;
        }

        let lastScrollY = window.scrollY;
        let ticking = false;

        function isMenuOpen() {
            return Boolean(nav.querySelector('.post-nav-panel.is-open, .post-contents[open], .blog-labs[open]'));
        }

        function updateNav() {
            const currentY = Math.max(window.scrollY, 0);
            const headerHeight = document.querySelector('.masthead')?.offsetHeight || 260;
            const pinStart = Math.max(96, Math.min(headerHeight * 0.5, headerHeight - 72));
            const delta = currentY - lastScrollY;

            if (currentY <= pinStart) {
                nav.classList.remove('is-fixed', 'is-visible');
            } else {
                nav.classList.add('is-fixed');

                if (isMenuOpen() || delta < -6) {
                    nav.classList.add('is-visible');
                } else if (delta > 6) {
                    nav.classList.remove('is-visible');
                }
            }

            lastScrollY = currentY;
            ticking = false;
        }

        function requestUpdate() {
            if (ticking) {
                return;
            }
            ticking = true;
            window.requestAnimationFrame(updateNav);
        }

        window.addEventListener('scroll', requestUpdate, { passive: true });
        window.addEventListener('resize', requestUpdate);
        nav.querySelector('.post-nav-toggle')?.addEventListener('click', () => {
            window.setTimeout(requestUpdate, 0);
        });
        nav.addEventListener('toggle', requestUpdate, true);
        updateNav();
    }

    function initializePage() {
        syncLanguagePreference();
        initializeMathRendering();
        setupShareButton();
        setupImageLightboxes();
        setupPostNavPanel();
        setupCompactContents();
        setupAutoRevealNav();
        initializeTOC();
        initializeSpecialViewers();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializePage, { once: true });
    } else {
        initializePage();
    }
})();
