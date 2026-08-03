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
        document.querySelectorAll('[data-image-lightbox-target]').forEach((trigger) => {
            const targetSelector = trigger.getAttribute('data-image-lightbox-target');
            const dialog = targetSelector ? document.querySelector(targetSelector) : null;

            if (!(dialog instanceof HTMLDialogElement)) {
                return;
            }

            const closeButton = dialog.querySelector('[data-image-lightbox-close]');
            let returnFocus = trigger;

            function openLightbox() {
                if (dialog.open) {
                    return;
                }

                returnFocus = document.activeElement instanceof HTMLElement
                    ? document.activeElement
                    : trigger;
                document.documentElement.classList.add('post-image-lightbox-open');
                dialog.showModal();
                window.requestAnimationFrame(() => closeButton?.focus());
            }

            function closeLightbox() {
                if (dialog.open) {
                    dialog.close();
                }
            }

            trigger.addEventListener('click', openLightbox);
            closeButton?.addEventListener('click', closeLightbox);
            dialog.addEventListener('click', (event) => {
                if (event.target === dialog) {
                    closeLightbox();
                }
            });
            dialog.addEventListener('close', () => {
                document.documentElement.classList.remove('post-image-lightbox-open');
                if (returnFocus?.isConnected) {
                    returnFocus.focus({ preventScroll: true });
                }
            });
        });
    }

    function initializeTOC() {
        const toc = document.querySelector('.toc');
        if (!toc) {
            return;
        }

        const contentsToggle = document.querySelector('.sidebar-contents-toggle');
        const labsMenu = document.querySelector('.sidebar-labs-menu');
        const railFlyoutMedia = window.matchMedia('(min-width: 1200px) and (max-width: 1599px)');
        const persistentTocMedia = window.matchMedia('(min-width: 1600px)');
        let tocItems = [];
        let railFlyoutOpen = false;

        if (!toc.id) {
            toc.id = 'post-toc';
        }
        contentsToggle?.setAttribute('aria-controls', toc.id);

        function updateRailFlyoutPosition() {
            if (!contentsToggle || !railFlyoutMedia.matches) {
                return;
            }

            const triggerRect = contentsToggle.getBoundingClientRect();
            const viewportPadding = 10;
            const top = Math.max(viewportPadding, Math.round(triggerRect.top));
            toc.style.setProperty('--post-rail-flyout-top', `${top}px`);
        }

        function updateTocVisibility() {
            const headerHeight = document.querySelector('.masthead')?.offsetHeight || 300;
            if (!railFlyoutMedia.matches && railFlyoutOpen) {
                railFlyoutOpen = false;
                contentsToggle?.setAttribute('aria-expanded', 'false');
            }

            const isPersistent = persistentTocMedia.matches && window.scrollY > headerHeight;
            const isVisible = railFlyoutMedia.matches ? railFlyoutOpen : isPersistent;
            toc.classList.toggle('is-rail-flyout', railFlyoutMedia.matches);
            toc.classList.toggle('is-visible', isVisible);
            toc.setAttribute('aria-hidden', isVisible ? 'false' : 'true');

            if (railFlyoutMedia.matches) {
                updateRailFlyoutPosition();
            }
        }

        function setRailFlyoutOpen(open, restoreFocus = false) {
            railFlyoutOpen = railFlyoutMedia.matches && open;
            contentsToggle?.setAttribute('aria-expanded', String(railFlyoutOpen));

            if (railFlyoutOpen && labsMenu?.open) {
                labsMenu.open = false;
            }

            updateTocVisibility();
            if (restoreFocus && contentsToggle) {
                contentsToggle.focus();
            }
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
        railFlyoutMedia.addEventListener('change', updateTocVisibility);
        persistentTocMedia.addEventListener('change', updateTocVisibility);
        contentsToggle?.addEventListener('click', () => {
            setRailFlyoutOpen(!railFlyoutOpen);
        });
        labsMenu?.addEventListener('toggle', () => {
            if (labsMenu.open && railFlyoutOpen) {
                setRailFlyoutOpen(false);
            }
        });
        document.addEventListener('pointerdown', (event) => {
            if (
                railFlyoutOpen
                && !toc.contains(event.target)
                && !contentsToggle?.contains(event.target)
            ) {
                setRailFlyoutOpen(false);
            }
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && railFlyoutOpen) {
                setRailFlyoutOpen(false, true);
            }
        });
        toc.querySelectorAll('a').forEach((link) => {
            link.addEventListener('click', () => {
                if (railFlyoutOpen) {
                    setRailFlyoutOpen(false);
                }
            });
        });
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

    function syncLanguagePreference() {
        const config = window.blogPostPageConfig;
        if (!config?.lang) {
            return;
        }

        document.documentElement.lang = config.lang === 'kor' ? 'ko' : 'en';

        try {
            localStorage.setItem('language', config.lang);
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
            return Boolean(nav.querySelector('.post-nav-panel.is-open'));
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
        updateNav();
    }

    function initializePage() {
        syncLanguagePreference();
        initializeMathRendering();
        setupShareButton();
        setupImageLightboxes();
        setupPostNavPanel();
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
