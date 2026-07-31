(function() {
    const icons = window.SiteIcons;
    const STORAGE_KEY = 'blog-theme';
    function getStoredTheme() {
        try {
            return localStorage.getItem(STORAGE_KEY);
        } catch (error) {
            return null;
        }
    }

    function storeTheme(theme) {
        try {
            localStorage.setItem(STORAGE_KEY, theme);
        } catch (error) {
            console.warn('Failed to persist theme preference:', error);
        }
    }

    function getPreferredTheme() {
        const storedTheme = getStoredTheme();
        if (storedTheme === 'dark' || storedTheme === 'light') {
            return storedTheme;
        }
        return 'light';
    }

    function updateToggleButtons(theme) {
        document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
            const icon = button.querySelector('.site-icon');
            const isDark = theme === 'dark';

            button.setAttribute('aria-pressed', String(isDark));
            button.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
            button.setAttribute('title', isDark ? 'Light mode' : 'Dark mode');

            if (icon) {
                icons.set(icon, isDark ? 'sun' : 'moon-stars');
            }
        });
    }

    function applyTheme(theme) {
        document.documentElement.dataset.theme = theme;
        updateToggleButtons(theme);
    }

    function initializeThemeToggle() {
        applyTheme(getPreferredTheme());

        document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
            button.addEventListener('click', () => {
                const currentTheme = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
                const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
                storeTheme(nextTheme);
                applyTheme(nextTheme);
            });
        });

    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeThemeToggle, { once: true });
    } else {
        initializeThemeToggle();
    }
})();
