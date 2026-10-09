function renderSiteHeader({ active = 'portfolio', themeControl = true } = {}) {
    const chapters = [['home', '00', 'Cover'], ['portfolio', '01', 'Projects'], ['blog', '02', 'Blogs'], ['about', '03', 'About']];
    return `<header class="edition-header">
    <a class="edition-brand" href="/#home" aria-label="Hwan Heo — return to cover">Hwan Heo<span class="edition-brand-note"> / Selected work &amp; writing</span></a>
    <nav class="edition-nav" aria-label="Portfolio chapters">${chapters.map(([id, number, label]) => `<a href="/#${id}"${id === active ? ' aria-current="page"' : ''}><span>${number}</span> ${label}</a>`).join('')}</nav>
    ${themeControl ? '<button class="edition-tone" type="button" aria-label="Switch to light theme" aria-pressed="true"><span class="edition-tone-icon" aria-hidden="true">◐</span><span class="edition-tone-label">Light</span></button>' : ''}
</header>`;
}
module.exports = { renderSiteHeader };
