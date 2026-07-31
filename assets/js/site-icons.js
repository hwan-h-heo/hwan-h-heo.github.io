(function(root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) {
        module.exports = api;
    }
    if (root) {
        root.SiteIcons = api;
    }
})(typeof globalThis !== 'undefined' ? globalThis : null, function() {
    const DEFAULT_SPRITE_PATH = '/assets/icons/site-icons.svg';
    const ICON_NAME_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

    function escapeAttribute(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function normalizeName(name) {
        const normalized = String(name || '').trim();
        if (!ICON_NAME_PATTERN.test(normalized)) {
            throw new Error(`Invalid site icon name: ${name}`);
        }
        return normalized;
    }

    function getHref(name, spritePath = DEFAULT_SPRITE_PATH) {
        return `${spritePath}#icon-${normalizeName(name)}`;
    }

    function render(name, options = {}) {
        const href = getHref(name, options.spritePath);
        const className = ['site-icon', options.className]
            .filter(Boolean)
            .join(' ');
        const accessibility = options.label
            ? `role="img" aria-label="${escapeAttribute(options.label)}"`
            : 'aria-hidden="true"';

        return `<svg class="${escapeAttribute(className)}" ${accessibility} focusable="false"><use href="${escapeAttribute(href)}"></use></svg>`;
    }

    function set(icon, name, options = {}) {
        if (!(icon instanceof Element)) {
            return false;
        }
        const use = icon.matches('use') ? icon : icon.querySelector('use');
        if (!use) {
            return false;
        }
        use.setAttribute('href', getHref(name, options.spritePath));
        icon.dataset.icon = normalizeName(name);
        return true;
    }

    return {
        getHref,
        render,
        set
    };
});
