(() => {
    'use strict';
    function initialize() {
        if (window.portfolioChapters || !document.querySelector('.pbr-actions')) return;
        const body = document.body;
        const ids = ['home', 'portfolio', 'blog', 'about'];
        const titles = {home:'Cover', portfolio:'Projects', blog:'Blog', about:'About'};
        const chapters = ids.map(id => ({id, section:document.getElementById(id), story:document.querySelector('[data-chapter-story="' + id + '"]')}));
        const cover = chapters[0].section;
        const reduced = matchMedia('(prefers-reduced-motion: reduce)');
        const mobile = matchMedia('(max-width: 900px)');
        let continuous = mobile.matches;
        const status = document.querySelector('.edition-status');
        const positions = new Map();
        let current = '', ranges = [], frame = 0, layoutFrame = 0, historyTimer = 0;
        let navigation = null, navigationTimer = 0, viewTransition = null, heroActive;
        let fixedTimer = 0;
        let viewportHeight = innerHeight, activation = 0;
        let lastHash = location.hash, interacted = false, initialized = false;
        let lastCoverLink = cover.querySelector('.pbr-actions a');
        const initialId = ids.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'home';
        const navigationType = performance.getEntriesByType('navigation')[0]?.type;
        const detailReturn = document.referrer && new URL(document.referrer).origin === location.origin
            && new URL(document.referrer).pathname.startsWith('/projects/');
        try {
            const saved = JSON.parse(sessionStorage.getItem('portfolio-chapter-scroll') || '{}');
            for (const id of ids) positions.set(id, Math.max(0, Number(saved[id]) || 0));
            const disclosures = JSON.parse(sessionStorage.getItem('portfolio-disclosures') || '[]');
            document.querySelectorAll('.edition-chapter details').forEach((node,index) => {node.open = disclosures[index] === true;});
        } catch {}
        history.scrollRestoration = 'manual';
        body.classList.toggle('edition-scroll', continuous);
        body.classList.toggle('edition-fixed', !continuous);

        const clamp = value => Math.max(0, Math.min(1, value));
        const ease = value => {const t = clamp(value); return t * t * (3 - 2 * t);};
        function style(node, name, value) {
            const next = String(value);
            if (node.style.getPropertyValue(name) !== next) node.style.setProperty(name, next);
        }
        function rangeFor(id) {return ranges.find(range => range.id === id);}
        function rememberPosition() {
            const range = rangeFor(current);
            if (range) positions.set(current, continuous
                ? Math.max(0, Math.min(range.overflow, scrollY - range.start))
                : range.section.scrollTop);
        }
        function rememberHistory() {
            clearTimeout(historyTimer);
            if (!current) return;
            rememberPosition();
            history.replaceState({...history.state, chapter:current, portfolioMode:continuous ? 'scroll' : 'fixed',
                portfolioScroll:scrollY, portfolioOffset:positions.get(current) || 0}, '', location.href);
        }
        function saveReadingState() {
            rememberPosition(); rememberHistory();
            try {
                sessionStorage.setItem('portfolio-chapter-scroll', JSON.stringify(Object.fromEntries(positions)));
                sessionStorage.setItem('portfolio-disclosures', JSON.stringify([...document.querySelectorAll('.edition-chapter details')].map(node => node.open)));
            } catch {}
        }
        function setCurrent(id) {
            if (current === id) return;
            if (!navigation) rememberPosition();
            const outgoingActions = actionGroups.find(group => group.id === current);
            if (outgoingActions) {
                outgoingActions.pointer = outgoingActions.focus = outgoingActions.active = null;
                outgoingActions.links.forEach(link => link.classList.remove('is-action-active'));
            }
            current = id;
            stopCoverIdleHint();
            clearNavigationHint();
            body.dataset.chapter = id;
            body.dataset.surface = id === 'home' ? 'cover' : 'reading';
            syncChapterVisibility();
            document.querySelectorAll('.edition-nav a').forEach(link => {
                if (link.hash === '#' + id) link.setAttribute('aria-current', 'page');
                else link.removeAttribute('aria-current');
            });
            document.title = 'Hwan Heo — ' + titles[id];
            document.querySelector('.edition-page-number').textContent = String(ids.indexOf(id)).padStart(2,'0') + ' / 03';
            document.dispatchEvent(new CustomEvent('portfolio:chapter-change', {detail:{chapter:id}}));
            if (continuous && initialized && !navigation) {
                lastHash = '#' + id;
                history.replaceState({...history.state, chapter:id, portfolioScroll:scrollY}, '', lastHash);
            }
            recordCoverActivity();
        }
        function syncChapterVisibility() {
            chapters.forEach(chapter => {
                const active = chapter.id === current;
                chapter.section.classList.toggle('is-active', active);
                chapter.section.inert = !continuous && !active;
                if (!continuous && !active) chapter.section.setAttribute('aria-hidden', 'true');
                else chapter.section.removeAttribute('aria-hidden');
            });
        }
        function setHeroActive(value) {
            if (heroActive === value) return;
            heroActive = value;
            cover.dataset.chapterActive = String(value);
            window.portfolioHero?.setChapterActive(value);
        }
        function render() {
            frame = 0;
            if (!ranges.length) return;
            if (!continuous) {
                if (current && !navigation) setHeroActive(current === 'home');
                return;
            }
            const y = scrollY;
            let active = 'home';
            for (const range of ranges) if (y >= range.start - activation) active = range.id;
            const coverRange = ranges[0];
            const blend = reduced.matches ? (y > coverRange.overflow + 1 ? 1 : 0)
                : ease((y - coverRange.overflow) / (viewportHeight * .72));
            style(body, '--ed-reading-opacity', blend.toFixed(4));
            setCurrent(active);
            if (initialized && !navigation && location.hash !== '#' + active) {
                lastHash = '#' + active;
                history.replaceState({...history.state, chapter:active, portfolioScroll:scrollY}, '', lastHash);
            }
            const nextHeroActive = y <= coverRange.overflow + 1;
            if ((initialized || navigation) && heroActive !== nextHeroActive) {
                setHeroActive(nextHeroActive);
            }
        }
        function refreshLayout() {
            layoutFrame = 0;
            const changedMode = continuous !== mobile.matches;
            const selected = navigation?.id || current;
            if (changedMode) {
                rememberPosition();
                continuous = mobile.matches;
                clearTimeout(fixedTimer); clearTimeout(navigationTimer); clearTimeout(historyTimer);
                navigation = null; viewTransition?.skipTransition();
                body.classList.remove('edition-changing','edition-selecting');
                clearNavigationHint();
                body.classList.toggle('edition-scroll', continuous);
                body.classList.toggle('edition-fixed', !continuous);
                syncChapterVisibility();
            }
            body.classList.toggle('edition-motion', continuous && !reduced.matches);
            // The scene is display:none in the static fallback. The fixed poster
            // retains viewport geometry in reduced motion and data-saving modes.
            const viewport = cover.querySelector('.pbr-poster').clientHeight || innerHeight;
            viewportHeight = viewport;
            const css = getComputedStyle(body);
            const header = parseFloat(css.getPropertyValue('--ed-header-size'));
            const top = parseFloat(css.getPropertyValue('--ed-top'));
            const stage = Math.max(1, viewport - header);
            activation = Math.min(180, stage * .22);
            style(body, '--ed-stage-height', stage + 'px');
            const intro = cover.querySelector('.pbr-intro');
            const coverExtent = Math.max(viewport, intro.offsetTop + intro.offsetHeight + 76);
            style(body, '--ed-cover-height', coverExtent + 'px');
            ranges = chapters.map((chapter,index) => {
                if (!continuous) {
                    const overflow = Math.max(0, chapter.section.scrollHeight - chapter.section.clientHeight);
                    return {...chapter, extent:chapter.section.scrollHeight, start:0, end:overflow, overflow, handoff:0};
                }
                const extent = chapter.section.offsetHeight;
                const padding = index === 0 ? 0 : parseFloat(getComputedStyle(chapter.section).paddingTop);
                const start = index === 0 ? 0 : chapter.story.offsetTop + padding - top - header;
                const overflow = Math.max(0, extent - (index === 0 ? viewport : stage + padding - top));
                return {...chapter, extent, start, end:start + overflow, overflow, handoff:0};
            });
            if (changedMode && selected) {
                navigation = {id:selected, focus:false};
                setCurrent(selected); syncChapterVisibility();
                if (continuous) window.scrollTo({top:rangeFor(selected).start + (positions.get(selected) || 0), behavior:'instant'});
                else {
                    window.scrollTo({top:0, behavior:'instant'});
                    rangeFor(selected).section.scrollTop = positions.get(selected) || 0;
                }
                render(); navigation = null; render(); rememberHistory();
            }
            // Native scroll anchoring keeps the reading position as content grows.
            // Measuring anchors must not interrupt touch inertia or wheel input.
            render();
        }
        function queueLayout() {
            if (!layoutFrame) layoutFrame = requestAnimationFrame(() => refreshLayout());
        }
        function focusChapter(id) {
            const destination = id === 'home' ? lastCoverLink : rangeFor(id).section.querySelector('h2');
            destination?.focus({preventScroll:true});
            status.textContent = titles[id] + ' opened';
        }
        function finishNavigation() {
            clearTimeout(navigationTimer);
            if (!navigation) return;
            const request = navigation;
            navigation = null;
            render(); rememberPosition(); rememberHistory();
            if (request.focus && current === request.id) focusChapter(request.id);
        }
        function navigate(id, {push = false, focus = true, top} = {}) {
            if (!ids.includes(id)) id = 'home';
            saveReadingState();
            if (push) history.pushState({chapter:id}, '', '#' + id);
            lastHash = location.hash;
            navigation = {id, focus};
            const request = navigation;
            viewTransition?.skipTransition();
            clearTimeout(fixedTimer); clearTimeout(navigationTimer);
            if (!continuous) {
                const offset = top ?? positions.get(id) ?? 0;
                const move = () => {
                    if (navigation !== request) return;
                    setCurrent(id); syncChapterVisibility();
                    rangeFor(id).section.scrollTop = offset;
                    navigation = null;
                    body.classList.remove('edition-changing');
                    setHeroActive(id === 'home');
                    rememberHistory();
                    if (focus) focusChapter(id);
                    recordCoverActivity();
                };
                if (id === current) {move(); return;}
                stopCoverIdleHint(); clearNavigationHint();
                body.dataset.surface = id === 'home' ? 'cover' : 'reading';
                body.classList.add('edition-changing');
                chapters.forEach(chapter => {
                    chapter.section.inert = true;
                    chapter.section.setAttribute('aria-hidden','true');
                    chapter.section.classList.remove('is-active');
                });
                if (id !== 'home') setHeroActive(false);
                if (reduced.matches || !initialized) move();
                else fixedTimer = setTimeout(move, 240);
                return;
            }
            const destination = top ?? rangeFor(id).start;
            const distance = Math.abs(destination - scrollY);
            const move = () => {
                if (navigation !== request) return;
                window.scrollTo({top:destination, behavior:'instant'});
                render(); finishNavigation();
            };
            if (!reduced.matches && distance > cover.clientHeight && document.startViewTransition) {
                body.classList.add('edition-selecting');
                const transition = document.startViewTransition(move);
                viewTransition = transition;
                transition.finished.finally(() => {
                    if (viewTransition === transition) {viewTransition = null; body.classList.remove('edition-selecting');}
                });
            } else if (!reduced.matches && distance > 1) {
                window.scrollTo({top:destination, behavior:'smooth'});
                clearTimeout(navigationTimer);
                navigationTimer = setTimeout(finishNavigation, 180);
            } else move();
        }
        function restoreRoute() {
            const state = history.state;
            const id = location.hash.slice(1) || 'home';
            const offset = Number.isFinite(state?.portfolioOffset) ? state.portfolioOffset : positions.get(id) || 0;
            const top = continuous
                ? (state?.portfolioMode === 'scroll' && Number.isFinite(state.portfolioScroll) ? state.portfolioScroll : rangeFor(id)?.start + offset)
                : offset;
            navigate(id, {top});
        }
        addEventListener('popstate', restoreRoute);
        addEventListener('hashchange', () => {if (location.hash !== lastHash) restoreRoute();});
        document.addEventListener('click', event => {
            const link = event.target.closest('a[href]');
            if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
            if (link.getAttribute('href').startsWith('#') && ids.includes(link.hash.slice(1))) {
                event.preventDefault();
                if (current === 'home') lastCoverLink = link;
                navigate(link.hash.slice(1), {push:location.hash !== link.hash});
            } else if (link.closest('.edition-chapter') && link.origin === location.origin && link.pathname.startsWith('/projects/')) {
                saveReadingState();
                try {sessionStorage.setItem('portfolio-return-focus', JSON.stringify({chapter:current, href:link.getAttribute('href')}));} catch {}
            }
        });
        document.addEventListener('keydown', event => {
            if (event.key === 'Escape' && current !== 'home' && !event.defaultPrevented) {
                event.preventDefault(); navigate('home', {push:true});
            }
        });
        // Content and focus use native document flow. Only the heading receives
        // a small entrance once; fast or reverse scrolling never hides a chapter.
        const reveals = new IntersectionObserver(entries => {
            for (const entry of entries) if (entry.isIntersecting) {
                entry.target.closest('.edition-chapter').classList.add('is-revealed');
                reveals.unobserve(entry.target);
            }
        }, {rootMargin:'0px 0px -48px 0px', threshold:.05});
        chapters.slice(1).forEach(chapter => reveals.observe(chapter.section.querySelector('.section-title')));
        addEventListener('scroll', () => {
            if (!continuous) return;
            if (!frame) frame = requestAnimationFrame(render);
            clearTimeout(historyTimer);
            historyTimer = setTimeout(rememberHistory, 200);
            if (navigation) {
                clearTimeout(navigationTimer);
                navigationTimer = setTimeout(finishNavigation, 160);
            }
        }, {passive:true});
        addEventListener('scrollend', () => {if (continuous && navigation) finishNavigation();});
        for (const name of ['wheel','touchstart','pointerdown','keydown']) document.addEventListener(name, () => {
            interacted = true;
            if (continuous && navigation && (name === 'wheel' || name === 'touchstart')) {
                navigation = null; clearTimeout(navigationTimer); viewTransition?.skipTransition();
            }
        }, {passive:true, capture:true});

        const actionGroups = chapters.map(({id,section}) => ({id,
            links:id === 'home' ? [...section.querySelectorAll('.pbr-actions a'), section.querySelector('.pbr-profile-link')].filter(Boolean)
                : [...section.querySelectorAll('.edition-next a')],
            pointer:null, focus:null, active:null}));
        const idleProjects = actionGroups[0].links[0];
        let hintedLink = null, hintTimer = 0, hintUntil = 0, wheelAmount = 0, wheelDirection = 0, lastWheel = 0;
        function clearNavigationHint() {
            clearTimeout(hintTimer); hintTimer = 0;
            hintedLink?.classList.remove('is-navigation-hint');
            cover.classList.remove('edition-navigation-hint');
            hintedLink = null; hintUntil = 0; wheelAmount = 0;
        }
        function expireNavigationHint() {
            hintTimer = 0;
            const remaining = hintUntil - performance.now();
            if (remaining > 0) hintTimer = setTimeout(expireNavigationHint, remaining);
            else clearNavigationHint();
        }
        function hintNavigation(direction) {
            if (continuous || !initialized || navigation || document.hidden) return;
            if (direction < 0) {clearNavigationHint(); return;}
            const group = actionGroups.find(group => group.id === current);
            const link = group?.links[current === 'home' ? 0 : 1];
            if (!link || (group.active && group.active !== link)) return;
            hintUntil = performance.now() + 2400;
            if (hintedLink !== link) {
                hintedLink?.classList.remove('is-navigation-hint');
                hintedLink = link; link.classList.add('is-navigation-hint');
                cover.classList.toggle('edition-navigation-hint', current === 'home');
            }
            if (!hintTimer) hintTimer = setTimeout(expireNavigationHint, 2400);
        }
        addEventListener('wheel', event => {
            if (continuous || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey
                || !event.deltaY || Math.abs(event.deltaX) >= Math.abs(event.deltaY)) return;
            const now = performance.now(), direction = Math.sign(event.deltaY);
            if (now - lastWheel > 500 || direction !== wheelDirection) wheelAmount = 0;
            lastWheel = now; wheelDirection = direction;
            wheelAmount += Math.abs(event.deltaY) * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1);
            if (wheelAmount >= 18) {hintNavigation(direction); wheelAmount = 0;}
        }, {passive:true});
        chapters.forEach(({id,section}) => {
            let previousTop = 0;
            section.addEventListener('scroll', () => {
                const top = section.scrollTop, delta = top - previousTop; previousTop = top;
                if (continuous || current !== id || navigation) return;
                clearTimeout(historyTimer); historyTimer = setTimeout(rememberHistory, 200);
                if (Math.abs(delta) >= 4) hintNavigation(Math.sign(delta));
            }, {passive:true});
        });
        document.addEventListener('keydown', event => {
            if (event.defaultPrevented || event.repeat || event.ctrlKey || event.metaKey || event.altKey
                || event.target.closest('input,textarea,select,[contenteditable]')) return;
            if (event.key === ' ' && event.target.closest('a,button,summary')) return;
            if (['ArrowUp','PageUp'].includes(event.key) || (event.key === ' ' && event.shiftKey)) hintNavigation(-1);
            else if (!event.shiftKey && ['ArrowDown','PageDown',' '].includes(event.key)) hintNavigation(1);
        });
        let idleTimer = 0, idlePulseTimer = 0, lastCoverActivity = performance.now();
        function clearCoverIdlePulse() {
            clearTimeout(idlePulseTimer); idlePulseTimer = 0; idleProjects.classList.remove('is-idle-hint');
        }
        function stopCoverIdleHint() {
            clearTimeout(idleTimer); idleTimer = 0; clearCoverIdlePulse();
        }
        function canScheduleCoverHint() {return current === 'home' && body.dataset.ready && !document.hidden && !reduced.matches;}
        function scheduleCoverHint(delay = 6000) {
            if (canScheduleCoverHint() && !idleTimer && !idlePulseTimer) idleTimer = setTimeout(playCoverIdleHint, delay);
        }
        function recordCoverActivity() {
            if (current !== 'home') return;
            lastCoverActivity = performance.now();
            if (idlePulseTimer) clearCoverIdlePulse();
            scheduleCoverHint();
        }
        function playCoverIdleHint() {
            idleTimer = 0;
            if (!canScheduleCoverHint()) return;
            const remaining = 6000 - (performance.now() - lastCoverActivity);
            if (remaining > 0) {scheduleCoverHint(remaining); return;}
            if (navigation || hintedLink || actionGroups[0].active || cover.querySelector('a:hover,button:hover,:focus-visible')) return;
            idleProjects.classList.add('is-idle-hint');
            idlePulseTimer = setTimeout(() => {clearCoverIdlePulse(); scheduleCoverHint(7000);}, 1400);
        }
        for (const name of ['pointermove','pointerdown','wheel','touchstart','touchmove','keydown','focusin','focusout','scroll']) {
            document.addEventListener(name, recordCoverActivity, {passive:true, capture:true});
        }
        for (const group of actionGroups) {
            function select(link) {
                recordCoverActivity(); group.active = link;
                if (link) clearNavigationHint();
                group.links.forEach(action => action.classList.toggle('is-action-active', action === link));
            }
            for (const link of group.links) {
                link.addEventListener('pointerenter', event => {if (event.pointerType !== 'touch') {group.pointer = link; select(link);}});
                link.addEventListener('pointerleave', () => {group.pointer = null; select(group.focus?.matches(':focus-visible') ? group.focus : null);});
                link.addEventListener('focus', () => {group.focus = link; select(link);});
                link.addEventListener('blur', () => {group.focus = null; select(group.pointer);});
            }
        }
        document.addEventListener('visibilitychange', () => {if (document.hidden) {stopCoverIdleHint(); clearNavigationHint();} else recordCoverActivity();});
        addEventListener('pagehide', () => {saveReadingState(); stopCoverIdleHint(); clearNavigationHint();});
        addEventListener('resize', queueLayout, {passive:true});
        mobile.addEventListener('change', queueLayout);
        reduced.addEventListener('change', () => {stopCoverIdleHint(); refreshLayout(); recordCoverActivity();});
        const observer = new ResizeObserver(queueLayout);
        chapters.slice(1).forEach(chapter => observer.observe(chapter.section));
        chapters.slice(1).forEach(chapter => chapter.section.querySelectorAll('.portfolio-shell').forEach(shell => observer.observe(shell)));
        observer.observe(cover.querySelector('.pbr-intro'));
        document.fonts.ready.then(queueLayout);

        refreshLayout();
        const restorePosition = detailReturn || navigationType === 'reload' || navigationType === 'back_forward';
        const storedTop = restorePosition && history.state?.portfolioMode === 'scroll'
            && Number.isFinite(history.state.portfolioScroll) ? history.state.portfolioScroll : null;
        function restoreInitial() {
            if (interacted) return;
            navigation = {id:initialId, focus:false};
            if (continuous) {
                const initialTop = storedTop ?? rangeFor(initialId).start + (restorePosition ? positions.get(initialId) || 0 : 0);
                window.scrollTo({top:initialTop, behavior:'instant'}); render();
            } else {
                setCurrent(initialId); syncChapterVisibility();
                rangeFor(initialId).section.scrollTop = restorePosition ? positions.get(initialId) || 0 : 0;
                window.scrollTo({top:0, behavior:'instant'});
                setHeroActive(initialId === 'home');
            }
            navigation = null;
        }
        function restoreReturnFocus() {
            if (detailReturn && !interacted) {
                try {
                    const saved = JSON.parse(sessionStorage.getItem('portfolio-return-focus') || 'null');
                    sessionStorage.removeItem('portfolio-return-focus');
                    const incoming = rangeFor(current).section;
                    if (saved?.chapter === current && [body, document.documentElement, incoming].includes(document.activeElement)) {
                        incoming.querySelector('a[href="' + CSS.escape(saved.href) + '"]')?.focus({preventScroll:true});
                    }
                } catch {}
            }
        }
        restoreInitial();
        body.dataset.ready = 'true'; initialized = true; recordCoverActivity();
        // Native fragment focus happens after DOMContentLoaded. Restore departure
        // focus at pageshow so it does not get replaced by the fragment target.
        addEventListener('pageshow', event => {
            if (event.persisted) {refreshLayout(); render();} else restoreInitial();
            restoreReturnFocus();
        });
        if (document.readyState === 'complete') restoreReturnFocus();
        window.portfolioChapters = window.editionPreview = {state:() => ({chapter:current, tone:window.siteTheme.get() === 'dark' ? 'ink' : 'paper',
            pausedByChapter:window.portfolioHero?.state().suspended || false, idleHint:idleProjects.classList.contains('is-idle-hint'),
            navigationHint:current === 'home' && !!hintedLink, chapterHint:current === 'home' ? null : hintedLink?.rel || null,
            windowScroll:scrollY, continuousScroll:continuous, reducedMotion:reduced.matches,
            chapters:ranges.map(({id,start,end,overflow,handoff}) => ({id,start,end,overflow,handoff}))})};
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, {once:true});
    else initialize();
    document.addEventListener('portfolio:core-ready', initialize, {once:true});
})();
