(function() {
    const icons = window.SiteIcons;
    const DESKTOP_MEDIA = '(min-width: 1200px)';
    const STORAGE_KEY = 'site-sidebar-collapsed';
    const CONTENT_FADE_DURATION = 100;
    const SIDEBAR_TRANSITION_DURATION = 480;
    const MOBILE_SCROLL_THRESHOLD = 8;
    const MOBILE_SCROLL_TOP_ZONE = 64;

    function createIconButton(className, iconName, label, controls) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = className;
        button.setAttribute('aria-label', label);
        button.setAttribute('title', label);

        if (controls) {
            button.setAttribute('aria-controls', controls);
        }

        button.insertAdjacentHTML('beforeend', icons.render(iconName));
        return button;
    }

    function readCollapsedPreference() {
        try {
            const storedPreference = localStorage.getItem(STORAGE_KEY);
            return storedPreference === null
                ? true
                : storedPreference === 'true';
        } catch (error) {
            return true;
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
        let mobileScrollFrame;
        let mobileScrollReferenceY = Math.max(0, window.scrollY);
        const collapsedLabsMenu = header.querySelector('.sidebar-labs-menu');

        const mobileToggle = createIconButton(
            'sidebar-mobile-toggle',
            'list',
            'Open navigation',
            header.id
        );
        mobileToggle.setAttribute('aria-expanded', 'false');
        header.before(mobileToggle);

        const collapseToggle = createIconButton(
            'sidebar-collapse-toggle',
            'layout-sidebar-inset',
            'Collapse sidebar',
            header.id
        );
        collapseToggle.setAttribute('aria-expanded', 'true');
        header.appendChild(collapseToggle);

        const collapsedTooltip = document.createElement('div');
        collapsedTooltip.className = 'sidebar-collapsed-tooltip';
        collapsedTooltip.setAttribute('role', 'tooltip');
        collapsedTooltip.setAttribute('aria-hidden', 'true');
        document.body.appendChild(collapsedTooltip);
        let collapsedTooltipTarget = null;

        function isCollapsed() {
            return documentRoot.classList.contains('sidebar-collapsed');
        }

        function closeCollapsedLabsMenu(restoreFocus = false) {
            if (!collapsedLabsMenu?.open) {
                return;
            }
            collapsedLabsMenu.open = false;
            if (restoreFocus) {
                collapsedLabsMenu.querySelector('summary')?.focus();
            }
        }

        function hideCollapsedTooltip(target) {
            if (target && target !== collapsedTooltipTarget) {
                return;
            }

            collapsedTooltipTarget = null;
            collapsedTooltip.classList.remove('is-visible');
            collapsedTooltip.setAttribute('aria-hidden', 'true');
        }

        function showCollapsedTooltip(target) {
            const label = target.dataset.sidebarLabel;
            const canShow = (
                label
                && desktopMedia.matches
                && isCollapsed()
                && !header.classList.contains('sidebar-layout-changing')
                && !header.classList.contains('sidebar-auto-hidden')
            );
            if (!canShow) {
                hideCollapsedTooltip();
                return;
            }

            collapsedTooltipTarget = target;
            collapsedTooltip.textContent = label;
            collapsedTooltip.classList.add('is-visible');
            collapsedTooltip.setAttribute('aria-hidden', 'false');

            const targetRect = target.getBoundingClientRect();
            const tooltipHalfHeight = collapsedTooltip.offsetHeight / 2;
            const viewportPadding = 8;
            const targetCenter = targetRect.top + targetRect.height / 2;
            const top = Math.min(
                Math.max(targetCenter, tooltipHalfHeight + viewportPadding),
                window.innerHeight - tooltipHalfHeight - viewportPadding
            );
            collapsedTooltip.style.left = `${Math.round(targetRect.right + 12)}px`;
            collapsedTooltip.style.top = `${Math.round(top)}px`;
        }

        function bindCollapsedTooltip(target) {
            const projectCopy = target.querySelector('.project-selector-copy');
            const projectLabel = projectCopy
                ? Array.from(projectCopy.querySelectorAll('small, strong'))
                    .map(element => element.textContent.replace(/\s+/g, ' ').trim())
                    .filter(Boolean)
                    .join(' · ')
                : '';
            const label = (
                target.getAttribute('aria-label')
                || projectLabel
                || target.textContent
            ).replace(/\s+/g, ' ').trim();
            if (!label) {
                return;
            }

            target.dataset.sidebarLabel = label;
            target.addEventListener('mouseenter', () => {
                showCollapsedTooltip(target);
            });
            target.addEventListener('mouseleave', () => {
                hideCollapsedTooltip(target);
            });
            target.addEventListener('focus', () => {
                showCollapsedTooltip(target);
            });
            target.addEventListener('blur', () => {
                hideCollapsedTooltip(target);
            });
        }

        function updateCollapseButton(collapsed) {
            const label = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
            collapseToggle.setAttribute('aria-label', label);
            collapseToggle.setAttribute('title', label);
            collapseToggle.setAttribute('aria-expanded', String(!collapsed));
        }

        function applyCollapsed(collapsed) {
            hideCollapsedTooltip();
            closeCollapsedLabsMenu();
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
            hideCollapsedTooltip();
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
            hideCollapsedTooltip();
            header.classList.toggle('header-show', open);
            documentRoot.classList.toggle('sidebar-mobile-open', open);
            if (open) {
                mobileToggle.classList.remove('is-scroll-hidden');
            }
            mobileToggle.setAttribute('aria-expanded', String(open));
            mobileToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
            mobileToggle.setAttribute('title', open ? 'Close navigation' : 'Open navigation');
            icons.set(mobileToggle.querySelector('.site-icon'), open ? 'x' : 'list');
        }

        function syncViewportState() {
            if (desktopMedia.matches) {
                setMobileOpen(false);
                setCollapsed(readCollapsedPreference(), false);
                mobileToggle.classList.remove('is-scroll-hidden');
            } else {
                setMobileOpen(false);
            }
            mobileScrollReferenceY = Math.max(0, window.scrollY);
        }

        function updateMobileToggleForScroll() {
            mobileScrollFrame = null;
            const currentScrollY = Math.max(0, window.scrollY);

            if (desktopMedia.matches || header.classList.contains('header-show')) {
                mobileToggle.classList.remove('is-scroll-hidden');
                mobileScrollReferenceY = currentScrollY;
                return;
            }

            if (currentScrollY <= MOBILE_SCROLL_TOP_ZONE) {
                mobileToggle.classList.remove('is-scroll-hidden');
                mobileScrollReferenceY = currentScrollY;
                return;
            }

            const scrollDelta = currentScrollY - mobileScrollReferenceY;
            if (Math.abs(scrollDelta) < MOBILE_SCROLL_THRESHOLD) {
                return;
            }

            mobileToggle.classList.toggle('is-scroll-hidden', scrollDelta > 0);
            mobileScrollReferenceY = currentScrollY;
        }

        window.addEventListener('scroll', () => {
            if (mobileScrollFrame) {
                return;
            }

            mobileScrollFrame = window.requestAnimationFrame(updateMobileToggleForScroll);
        }, { passive: true });

        collapseToggle.addEventListener('click', () => {
            setCollapsed(!isCollapsed());
        });

        mobileToggle.addEventListener('click', () => {
            setMobileOpen(!header.classList.contains('header-show'));
        });

        header.querySelectorAll('#navmenu a, .sidebar-labs-panel a').forEach((link) => {
            bindCollapsedTooltip(link);

            link.addEventListener('click', () => {
                hideCollapsedTooltip(link);
                if (!desktopMedia.matches) {
                    setMobileOpen(false);
                }
            });
        });

        header.querySelectorAll('details > summary').forEach((summary) => {
            bindCollapsedTooltip(summary);

            summary.addEventListener('click', (event) => {
                hideCollapsedTooltip(summary);
                if (!desktopMedia.matches || !isCollapsed()) {
                    return;
                }

                event.preventDefault();
                if (summary.parentElement.matches('.sidebar-labs-menu')) {
                    summary.parentElement.open = !summary.parentElement.open;
                    return;
                }
                setCollapsed(false, true, () => {
                    summary.parentElement.open = true;
                    summary.focus();
                });
            });
        });

        document.addEventListener('pointerdown', (event) => {
            if (
                desktopMedia.matches
                && isCollapsed()
                && collapsedLabsMenu?.open
                && !collapsedLabsMenu.contains(event.target)
            ) {
                closeCollapsedLabsMenu();
            }
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
            if (event.key === 'Escape' && collapsedLabsMenu?.open) {
                closeCollapsedLabsMenu(true);
                return;
            }
            if (event.key === 'Escape' && header.classList.contains('header-show')) {
                setMobileOpen(false);
                mobileToggle.focus();
            }
        });

        window.addEventListener('resize', () => {
            hideCollapsedTooltip();
            closeCollapsedLabsMenu();
            mobileScrollReferenceY = Math.max(0, window.scrollY);
        });
        header.addEventListener('scroll', () => {
            hideCollapsedTooltip();
        }, { passive: true });

        const autoHideTargetSelector = header.dataset.sidebarAutoHide;
        if (autoHideTargetSelector && 'IntersectionObserver' in window) {
            const target = document.querySelector(autoHideTargetSelector);
            if (target) {
                const observer = new IntersectionObserver((entries) => {
                    const targetVisible = entries.some((entry) => entry.isIntersecting);
                    if (targetVisible) {
                        hideCollapsedTooltip();
                    }
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
