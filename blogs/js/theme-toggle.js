/* site-theme.js owns Blog theme controls and its independent preference. */
(() => {
    if (window.siteTheme) { window.siteTheme.sync(); return; }
    const script = document.createElement('script');
    script.src = '/js/site-theme.js';
    document.head.append(script);
})();
