(function() {
    const DESKTOP_MEDIA = '(min-width: 1200px)';
    const STORAGE_KEY = 'site-sidebar-collapsed';
    const CONTENT_FADE_DURATION = 100;
    const SIDEBAR_TRANSITION_DURATION = 480;

    function createIconButton(className, iconClass, label, controls) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = className;
        button.setAttribute('aria-label', label);
        button.setAttribute('title', label);

        if (controls) {
            button.setAttribute('aria-controls', controls);
        }

        const icon = document.createElement('i');
        icon.className = `bi ${iconClass}`;
        icon.setAttribute('aria-hidden', 'true');
        button.appendChild(icon);
        return button;
    }

    function readCollapsedPreference() {
        try {
            return localStorage.getItem(STORAGE_KEY) === 'true';
        } catch (error) {
            return false;
        }
    }

    function storeCollapsedPreference(collapsed) {
        try {
            localStorage.setItem(STORAGE_KEY, String(collapsed));
        } catch (error) {}
    }

    document.documentElement.classList.toggle(
        'sidebar-collapsed',
        readCollapsedPreference()
    );

    function initializeSidebar() {
        const header = document.getElementById('header');
        if (!header || header.dataset.sidebarInitialized === 'true') {
            return;
        }

        header.dataset.sidebarInitialized = 'true';
        const desktopMedia = window.matchMedia(DESKTOP_MEDIA);
        const reducedMotionMedia = window.matchMedia('(prefers-reduced-motion: reduce)');
        const documentRoot = document.documentElement;
        let collapseChangeTimer;
        let collapseRevealTimer;

        const mobileToggle = createIconButton(
            'sidebar-mobile-toggle',
            'bi-list',
            'Open navigation',
            header.id
        );
        mobileToggle.setAttribute('aria-expanded', 'false');
        header.before(mobileToggle);

        const collapseToggle = createIconButton(
            'sidebar-collapse-toggle',
            'bi-layout-sidebar-inset',
            'Collapse sidebar',
            header.id
        );
        collapseToggle.setAttribute('aria-expanded', 'true');
        header.appendChild(collapseToggle);

        function isCollapsed() {
            return documentRoot.classList.contains('sidebar-collapsed');
        }

        function updateCollapseButton(collapsed) {
            const label = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
            collapseToggle.setAttribute('aria-label', label);
            collapseToggle.setAttribute('title', label);
            collapseToggle.setAttribute('aria-expanded', String(!collapsed));
        }

        function applyCollapsed(collapsed) {
            documentRoot.classList.toggle('sidebar-collapsed', collapsed);
            updateCollapseButton(collapsed);

            if (collapsed) {
                header.querySelectorAll('details[open]').forEach((details) => {
                    details.open = false;
                });
            }

            window.dispatchEvent(new CustomEvent('sidebar:resize', {
                detail: { collapsed }
            }));
        }

        function cancelCollapsedTransition() {
            window.clearTimeout(collapseChangeTimer);
            window.clearTimeout(collapseRevealTimer);
            header.classList.remove('sidebar-layout-changing');
            collapseToggle.disabled = false;
        }

        function setCollapsed(collapsed, persist = true, onComplete) {
            if (persist) {
                storeCollapsedPreference(collapsed);
            }

            const shouldAnimate = (
                persist
                && desktopMedia.matches
                && !reducedMotionMedia.matches
                && collapsed !== isCollapsed()
            );

            if (!shouldAnimate) {
                cancelCollapsedTransition();
                applyCollapsed(collapsed);
                onComplete?.();
                return;
            }

            cancelCollapsedTransition();
            header.classList.add('sidebar-layout-changing');
            collapseToggle.disabled = true;
            updateCollapseButton(collapsed);

            collapseChangeTimer = window.setTimeout(() => {
                applyCollapsed(collapsed);
                collapseRevealTimer = window.setTimeout(() => {
                    header.classList.remove('sidebar-layout-changing');
                    collapseToggle.disabled = false;
                    onComplete?.();
                }, SIDEBAR_TRANSITION_DURATION);
            }, CONTENT_FADE_DURATION);
        }

        function setMobileOpen(open) {
            header.classList.toggle('header-show', open);
            documentRoot.classList.toggle('sidebar-mobile-open', open);
            mobileToggle.setAttribute('aria-expanded', String(open));
            mobileToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
            mobileToggle.setAttribute('title', open ? 'Close navigation' : 'Open navigation');
            mobileToggle.querySelector('i')?.classList.toggle('bi-list', !open);
            mobileToggle.querySelector('i')?.classList.toggle('bi-x', open);
        }

        function syncViewportState() {
            if (desktopMedia.matches) {
                setMobileOpen(false);
                setCollapsed(readCollapsedPreference(), false);
            } else {
                setMobileOpen(false);
            }
        }

        collapseToggle.addEventListener('click', () => {
            setCollapsed(!isCollapsed());
        });

        mobileToggle.addEventListener('click', () => {
            setMobileOpen(!header.classList.contains('header-show'));
        });

        header.querySelectorAll('#navmenu a, .sidebar-labs-panel a').forEach((link) => {
            const label = link.textContent.replace(/\s+/g, ' ').trim();
            if (label && !link.hasAttribute('title')) {
                link.setAttribute('title', label);
            }

            link.addEventListener('click', () => {
                if (!desktopMedia.matches) {
                    setMobileOpen(false);
                }
            });
        });

        header.querySelectorAll('details > summary').forEach((summary) => {
            const label = summary.textContent.replace(/\s+/g, ' ').trim();
            if (label && !summary.hasAttribute('title')) {
                summary.setAttribute('title', label);
            }

            summary.addEventListener('click', (event) => {
                if (!desktopMedia.matches || !isCollapsed()) {
                    return;
                }

                event.preventDefault();
                setCollapsed(false, true, () => {
                    summary.parentElement.open = true;
                    summary.focus();
                });
            });
        });

        document.addEventListener('pointerdown', (event) => {
            if (
                !desktopMedia.matches
                && header.classList.contains('header-show')
                && !header.contains(event.target)
                && !mobileToggle.contains(event.target)
            ) {
                setMobileOpen(false);
            }
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && header.classList.contains('header-show')) {
                setMobileOpen(false);
                mobileToggle.focus();
            }
        });

        const autoHideTargetSelector = header.dataset.sidebarAutoHide;
        if (autoHideTargetSelector && 'IntersectionObserver' in window) {
            const target = document.querySelector(autoHideTargetSelector);
            if (target) {
                const observer = new IntersectionObserver((entries) => {
                    const targetVisible = entries.some((entry) => entry.isIntersecting);
                    mobileToggle.classList.toggle('is-auto-hidden', targetVisible);
                    header.classList.toggle(
                        'sidebar-auto-hidden',
                        desktopMedia.matches && targetVisible
                    );
                }, { threshold: 0.2 });
                observer.observe(target);

                desktopMedia.addEventListener('change', () => {
                    if (!desktopMedia.matches) {
                        header.classList.remove('sidebar-auto-hidden');
                    }
                });
            }
        } else {
            header.classList.remove('sidebar-auto-hidden');
        }

        desktopMedia.addEventListener('change', syncViewportState);
        syncViewportState();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeSidebar, { once: true });
    } else {
        initializeSidebar();
    }
})();
