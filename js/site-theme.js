/* Pre-paint theme controls with independent portfolio and Blog preferences. */
(() => {
    'use strict';
    const valid = value => value === 'dark' || value === 'light';
    const isBlog = /^\/blogs(?:\/|$)/.test(location.pathname);
    const storageKey = isBlog ? 'blog-reading-theme' : 'portfolio-theme';
    const defaultTheme = isBlog ? 'light' : 'dark';
    function savedTheme(fallback = defaultTheme) {
        try {
            const saved = localStorage.getItem(storageKey);
            if (valid(saved)) return saved;
            // The old Blog key was mirrored by portfolio controls. Only migrate
            // the old shared preference into the portfolio, never into writing.
            const legacy = !isBlog && localStorage.getItem('site-theme');
            if (valid(legacy)) return legacy;
        } catch { return fallback; }
        return defaultTheme;
    }
    const params = new URLSearchParams(location.search);
    const override = params.get('theme') || (params.get('tone') === 'paper' ? 'light' : params.get('tone') === 'ink' ? 'dark' : null);
    let theme = valid(override) ? override : savedTheme();
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let themeTransition = null, themeRevision = 0;
    try { localStorage.setItem(storageKey, theme); } catch {}
    function sync() {
        document.documentElement.dataset.theme = theme;
        if (document.body) document.body.dataset.tone = theme === 'dark' ? 'ink' : 'paper';
        document.querySelectorAll('[data-theme-toggle],.edition-tone').forEach(button => {
            const dark = theme === 'dark';
            button.setAttribute('aria-pressed', String(dark));
            button.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} theme`);
            button.setAttribute('title', dark ? 'Light theme' : 'Dark theme');
            const label = button.querySelector('.edition-tone-label');
            if (label) label.textContent = dark ? 'Light' : 'Dark';
            const icon = button.querySelector('.site-icon');
            if (icon) window.SiteIcons?.set(icon, dark ? 'sun' : 'moon-stars');
        });
    }
    function set(value, persist = true, animate = false) {
        if (!valid(value)) return;
        const revision = ++themeRevision;
        theme = value;
        if (persist) try {
            localStorage.setItem(storageKey, theme);
        } catch {}
        themeTransition?.skipTransition();
        themeTransition = null;
        const root = document.documentElement;
        const apply = () => {
            // A skipped transition can still run its update callback. Only the
            // latest requested theme may update the document after rapid clicks.
            if (revision !== themeRevision) return;
            sync();
            document.dispatchEvent(new CustomEvent('site:theme-change', {detail:{theme}}));
        };
        if (!animate || reduced.matches || document.hidden || !document.body || !document.startViewTransition) {
            root.classList.remove('is-theme-changing');
            apply();
            return;
        }
        root.classList.add('is-theme-changing');
        try {
            const transition = document.startViewTransition(apply);
            themeTransition = transition;
            transition.ready.catch(() => {}); // Skipping an interrupted fade is expected.
            const finish = () => {
                if (themeTransition !== transition) return;
                themeTransition = null;
                root.classList.remove('is-theme-changing');
            };
            transition.finished.then(finish, finish);
        } catch {
            root.classList.remove('is-theme-changing');
            apply();
        }
    }
    window.siteTheme = {set, get: () => theme, sync};
    sync();
    if (valid(override)) set(theme);
    document.addEventListener('DOMContentLoaded', sync, {once:true});
    addEventListener('pageshow', () => {
        set(savedTheme(theme), false);
    });
    addEventListener('storage', event => {
        if (event.key === storageKey || event.key === null) set(savedTheme(), false);
    });
    document.addEventListener('click', event => {
        let button = event.target.closest('[data-theme-toggle],.edition-tone');
        // Captured root content may hit-test as <html> during a native fade.
        // Resolve another click on the still-visible theme control only.
        if (!button && themeTransition && event.target === document.documentElement && event.detail > 0) {
            button = [...document.querySelectorAll('[data-theme-toggle],.edition-tone')].find(node => {
                const rect = node.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0 && event.clientX >= rect.left && event.clientX <= rect.right
                    && event.clientY >= rect.top && event.clientY <= rect.bottom;
            });
            button?.focus({preventScroll:true});
        }
        if (button) set(theme === 'dark' ? 'light' : 'dark', true, true);
    });
    reduced.addEventListener('change', () => {if (reduced.matches) set(theme, false);});
    document.addEventListener('visibilitychange', () => {if (document.hidden && themeTransition) set(theme, false);});
})();
