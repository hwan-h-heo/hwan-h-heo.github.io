(() => {
    'use strict';
    function initialize() {
        if (window.portfolioChapters || !document.querySelector('.pbr-actions')) return;
        const body = document.body;
        const cover = document.querySelector('.pbr-hero');
        const ids = ['home', 'portfolio', 'blog', 'about'];
        const titles = {home:'Cover', portfolio:'Projects', blog:'Blogs', about:'About'};
        const sections = new Map(ids.map(id => [id, document.getElementById(id)]));
        const reduced = matchMedia('(prefers-reduced-motion: reduce)');
        const status = document.querySelector('.edition-status');
        const scrollPositions = new Map();
        try {
            const saved = JSON.parse(sessionStorage.getItem('portfolio-chapter-scroll') || '{}');
            for (const id of ids) scrollPositions.set(id, Number(saved[id]) || 0);
        } catch {}
        let current = 'home', transition = 0, ignoreScrollUntil = 0;
        let lastCoverLink = document.querySelector('.pbr-actions a');
        function saveScroll() {
            scrollPositions.set(current, sections.get(current).scrollTop);
            try { sessionStorage.setItem('portfolio-chapter-scroll', JSON.stringify(Object.fromEntries(scrollPositions))); } catch {}
        }
        function saveReadingState() {
            saveScroll();
            const disclosures = [...document.querySelectorAll('.edition-chapter details')].map(node => node.open);
            try { sessionStorage.setItem('portfolio-disclosures', JSON.stringify(disclosures)); } catch {}
        }
        try {
            const saved = JSON.parse(sessionStorage.getItem('portfolio-disclosures') || '[]');
            document.querySelectorAll('.edition-chapter details').forEach((node,index) => {node.open = saved[index] === true;});
        } catch {}
        addEventListener('pagehide', saveReadingState);
        document.addEventListener('click', event => {
            const link = event.target.closest('.edition-chapter a[href]');
            if (link && !link.hash && link.origin === location.origin) {
                saveReadingState();
                try { sessionStorage.setItem('portfolio-return-focus', JSON.stringify({chapter:current, href:link.getAttribute('href')})); } catch {}
            }
        });
        function freezeCover() {
            window.portfolioHero?.setChapterActive(false);
            // Keep this exact canvas and camera in place. Only the reading scrim
            // changes; replacing it with a second crop caused a visible seam in v23.
        }
        function showChapter(id, focus = true) {
            if (!sections.has(id)) id = 'home';
            const previous = current;
            if (id === current && body.dataset.ready) return;
            const request = ++transition;
            saveScroll();
            if (previous === 'home' && id !== 'home') freezeCover();
            current = id;
            stopCoverIdleHint();
            clearNavigationHint();
            ignoreScrollUntil = performance.now() + 600;
            const previousActions = actionGroups.get(previous);
            previousActions.pointer = previousActions.focus = previousActions.active = null;
            previousActions.links.forEach(link => link.classList.remove('is-action-active'));
            body.dataset.surface = id === 'home' ? 'cover' : 'reading';
            body.classList.add('edition-changing');
            document.dispatchEvent(new CustomEvent('portfolio:chapter-change', {detail:{chapter:id}}));
            for (const [key, section] of sections) {
                section.inert = true;
                section.setAttribute('aria-hidden', 'true');
                section.classList.remove('is-active');
            }
            // The backdrop is already dimming while the outgoing copy fades.
            // Chapter-to-chapter changes leave the stage and reading scrim untouched.
            setTimeout(() => {
                if (request !== transition) return;
                body.dataset.chapter = id;
                const incoming = sections.get(id);
                incoming.inert = false;
                incoming.removeAttribute('aria-hidden');
                incoming.classList.add('is-active');
                incoming.scrollTop = scrollPositions.get(id) || 0;
                for (const link of document.querySelectorAll('.edition-nav a')) {
                    if (link.hash === '#' + id) link.setAttribute('aria-current', 'page');
                    else link.removeAttribute('aria-current');
                }
                document.querySelector('.edition-page-number').textContent = `${String(ids.indexOf(id)).padStart(2,'0')} / 03`;
                document.title = `Hwan Heo — ${titles[id]}`;
                status.textContent = `${titles[id]} opened`;
                if (id === 'home') window.portfolioHero?.setChapterActive(true);
                body.classList.remove('edition-changing');
                body.dataset.ready = 'true';
                if (!focus && document.referrer && new URL(document.referrer).pathname !== '/') {
                    const restoreReturnFocus = () => {
                        try {
                            const saved = JSON.parse(sessionStorage.getItem('portfolio-return-focus') || 'null');
                            sessionStorage.removeItem('portfolio-return-focus');
                            // Native fragment navigation can focus the section at
                            // load. Restore after it, without stealing newer input.
                            const active = document.activeElement;
                            if (current === id && saved?.chapter === id
                                && [body, document.documentElement, incoming].includes(active)) {
                                incoming.querySelector(`a[href="${CSS.escape(saved.href)}"]`)?.focus({preventScroll:true});
                            }
                        } catch {}
                    };
                    if (document.readyState === 'complete') restoreReturnFocus();
                    else addEventListener('pageshow', restoreReturnFocus, {once:true});
                }
                if (focus) {
                    const destination = id === 'home' ? lastCoverLink : incoming.querySelector('h2');
                    destination?.focus({preventScroll:true});
                }
                recordCoverActivity();
            }, reduced.matches || !body.dataset.ready ? 0 : 240);
        }
        document.addEventListener('click', event => {
            const link = event.target.closest('a[href^="#"]');
            if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
            const id = link.hash.slice(1);
            if (!ids.includes(id)) return;
            event.preventDefault();
            if (current === 'home') lastCoverLink = link;
            saveScroll();
            if (current !== id) history.pushState({chapter:id}, '', '#' + id);
            showChapter(id);
        });
        addEventListener('popstate', () => showChapter(location.hash.slice(1) || 'home'));
        addEventListener('hashchange', () => showChapter(location.hash.slice(1) || 'home'));
        document.addEventListener('keydown', event => {
            if (event.key === 'Escape' && current !== 'home') {
                event.preventDefault(); history.pushState({chapter:'home'}, '', '#home'); showChapter('home');
            }
        });
        // Gesture intent offers a link; native scrolling and focus remain intact.
        let hintTimer = 0, hintUntil = 0, hintedLink = null, wheelAmount = 0, wheelDirection = 0, lastWheel = 0, touch = null;
        const actionGroups = new Map(ids.map(id => {
            const section = sections.get(id), nav = section.querySelector(id === 'home' ? '.pbr-actions' : '.edition-next');
            const links = [...nav.querySelectorAll('a')];
            const profileLink = id === 'home' && section.querySelector('.pbr-profile-link');
            if (profileLink) links.push(profileLink);
            return [id, {nav, links, pointer:null, focus:null, active:null}];
        }));
        const idleProjects = actionGroups.get('home').links[0];
        let idleTimer = 0, idlePulseTimer = 0, lastCoverActivity = performance.now();
        function clearCoverIdlePulse() {
            clearTimeout(idlePulseTimer); idlePulseTimer = 0;
            idleProjects.classList.remove('is-idle-hint');
        }
        function stopCoverIdleHint() {
            clearTimeout(idleTimer); idleTimer = 0;
            clearCoverIdlePulse();
        }
        function canScheduleCoverHint() {
            return current === 'home' && body.dataset.ready && !document.hidden && !reduced.matches;
        }
        function scheduleCoverHint(delay = 6000) {
            if (canScheduleCoverHint() && !idleTimer && !idlePulseTimer) idleTimer = setTimeout(playCoverIdleHint, delay);
        }
        function recordCoverActivity() {
            if (current !== 'home') return;
            lastCoverActivity = performance.now();
            if (idlePulseTimer) clearCoverIdlePulse();
            // High-frequency pointer events only update a timestamp; one timeout
            // checks the remaining quiet period. No polling or renderer work.
            scheduleCoverHint();
        }
        function playCoverIdleHint() {
            idleTimer = 0;
            if (!canScheduleCoverHint()) return;
            const remaining = 6000 - (performance.now() - lastCoverActivity);
            if (remaining > 0) {scheduleCoverHint(remaining); return;}
            if (body.classList.contains('edition-changing') || hintedLink || actionGroups.get('home').active
                || cover.querySelector('a:hover,button:hover,:focus-visible')) return;
            idleProjects.classList.add('is-idle-hint');
            // This decorative cue never changes focus or announces to a live region.
            idlePulseTimer = setTimeout(() => {
                clearCoverIdlePulse();
                scheduleCoverHint(7000);
            }, 1400);
        }
        for (const name of ['pointermove','pointerdown','wheel','touchstart','touchmove','keydown','focusin','focusout']) {
            document.addEventListener(name, recordCoverActivity, {passive:true, capture:true});
        }
        addEventListener('pagehide', stopCoverIdleHint);
        addEventListener('pageshow', recordCoverActivity);
        reduced.addEventListener('change', () => {stopCoverIdleHint(); recordCoverActivity();});
        function selectAction(group, link) {
            recordCoverActivity();
            group.active = link;
            if (link && group === actionGroups.get(current)) clearNavigationHint();
            for (const action of group.links) action.classList.toggle('is-action-active', action === link);
        }
        for (const group of actionGroups.values()) for (const link of group.links) {
            link.addEventListener('pointerenter', event => {
                if (event.pointerType === 'touch') return;
                group.pointer = link; selectAction(group, link);
            });
            link.addEventListener('pointerleave', () => {
                group.pointer = null;
                selectAction(group, group.focus?.matches(':focus-visible') ? group.focus : null);
            });
            link.addEventListener('focus', () => {group.focus = link; selectAction(group, link);});
            link.addEventListener('blur', () => {group.focus = null; selectAction(group, group.pointer);});
        }
        function clearNavigationHint() {
            clearTimeout(hintTimer); hintTimer = 0;
            if (hintedLink) {
                hintedLink.classList.remove('is-navigation-hint');
                cover.classList.remove('edition-navigation-hint');
            }
            hintedLink = null; hintUntil = 0; wheelAmount = 0; wheelDirection = 0; touch = null;
        }
        function expireNavigationHint() {
            hintTimer = 0;
            const remaining = hintUntil - performance.now();
            if (remaining > 0) hintTimer = setTimeout(expireNavigationHint, remaining);
            else clearNavigationHint();
        }
        function hintNavigation(direction) {
            if (!body.dataset.ready || document.hidden || body.classList.contains('edition-changing')) return;
            recordCoverActivity();
            // Scrolling back through a long page is reading, not previous-chapter intent.
            if (direction < 0) {clearNavigationHint(); return;}
            const group = actionGroups.get(current);
            const link = current === 'home' ? group.links[0] : group.links[1];
            if (!link || (group.active && group.active !== link)) return;
            hintUntil = performance.now() + 2400;
            if (hintedLink !== link) {
                hintedLink?.classList.remove('is-navigation-hint');
                hintedLink = link; link.classList.add('is-navigation-hint');
                cover.classList.toggle('edition-navigation-hint', current === 'home');
                status.textContent = current === 'home' ? 'Open Projects to continue.' : `Next chapter: ${titles[link.hash.slice(1)]}.`;
            }
            // Continuous input extends one expiry timestamp; it does not restart the
            // underline transition, write classes or create a timer for every event.
            if (!hintTimer) hintTimer = setTimeout(expireNavigationHint, 2400);
        }
        addEventListener('wheel', event => {
            if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
            if (!event.deltaY || Math.abs(event.deltaX) >= Math.abs(event.deltaY)) {wheelAmount = 0; return;}
            const now = performance.now(), direction = Math.sign(event.deltaY);
            if (now - lastWheel > 500 || direction !== wheelDirection) wheelAmount = 0;
            lastWheel = now; wheelDirection = direction;
            wheelAmount += Math.abs(event.deltaY) * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1);
            if (wheelAmount >= 18) {hintNavigation(direction); wheelAmount = 0;}
        }, {passive:true});
        document.addEventListener('touchstart', event => {
            touch = event.touches.length === 1 ? {x:event.touches[0].clientX,y:event.touches[0].clientY} : null;
        }, {passive:true});
        document.addEventListener('touchmove', event => {
            if (!touch || event.touches.length !== 1) {touch = null; return;}
            const dy = touch.y - event.touches[0].clientY, dx = touch.x - event.touches[0].clientX;
            if (Math.abs(dy) > 24 && Math.abs(dy) > Math.abs(dx) * 1.4) {
                hintNavigation(Math.sign(dy));
                touch = {x:event.touches[0].clientX,y:event.touches[0].clientY};
            }
        }, {passive:true});
        for (const name of ['touchend','touchcancel']) document.addEventListener(name, () => {touch = null;}, {passive:true});
        // Also cover scrollbar dragging and scrolling that continues after a touch.
        for (const id of ids.filter(id => id !== 'home')) {
            const section = sections.get(id);
            let previousTop = 0;
            section.addEventListener('scroll', () => {
                const top = section.scrollTop, delta = top - previousTop; previousTop = top;
                if (current === id && performance.now() > ignoreScrollUntil && Math.abs(delta) >= 4) hintNavigation(Math.sign(delta));
            }, {passive:true});
        }
        document.addEventListener('keydown', event => {
            if (event.defaultPrevented || event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
            if (event.target.closest('input,textarea,select,[contenteditable]')) return;
            if (event.key === ' ' && event.target.closest('a,button,summary')) return;
            if (['ArrowUp','PageUp'].includes(event.key) || (event.key === ' ' && event.shiftKey)) hintNavigation(-1);
            else if (!event.shiftKey && ['ArrowDown','PageDown',' '].includes(event.key)) hintNavigation(1);
        });
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {clearNavigationHint(); stopCoverIdleHint();}
            else recordCoverActivity();
        });
        showChapter(ids.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'home', false);
        if (current !== 'home') freezeCover();
        window.portfolioChapters = window.editionPreview = {state: () => ({chapter:current, tone:window.siteTheme.get() === 'dark' ? 'ink' : 'paper',
            pausedByChapter:window.portfolioHero?.state().suspended || false,
            idleHint:idleProjects.classList.contains('is-idle-hint'),
            navigationHint:current === 'home' && !!hintedLink, chapterHint:current === 'home' ? null : hintedLink?.rel || null, windowScroll:scrollY})};
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, {once:true});
    else initialize();
    document.addEventListener('portfolio:core-ready', initialize, {once:true});
})();
