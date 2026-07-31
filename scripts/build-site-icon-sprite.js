const fs = require('fs');
const path = require('path');

const iconNames = [
    'arrow-down-right',
    'arrow-left',
    'arrow-repeat',
    'arrow-right',
    'arrow-up',
    'arrow-up-right',
    'bounding-box-circles',
    'box',
    'box-arrow-up-right',
    'briefcase',
    'card-image',
    'caret-down-square',
    'caret-left',
    'caret-left-fill',
    'caret-right',
    'caret-right-fill',
    'check2',
    'chevron-down',
    'clipboard',
    'clock',
    'cloud-arrow-down',
    'cloud-arrow-up',
    'collection',
    'cursor',
    'download',
    'envelope',
    'envelope-fill',
    'file-earmark-text',
    'folder2-open',
    'fullscreen',
    'fullscreen-exit',
    'gear',
    'github',
    'google',
    'grid',
    'house',
    'house-door',
    'image',
    'images',
    'keyboard',
    'layout-sidebar',
    'layout-sidebar-inset',
    'layout-text-window-reverse',
    'link',
    'link-45deg',
    'linkedin',
    'list',
    'list-ul',
    'markdown',
    'moon-stars',
    'mortarboard-fill',
    'pencil',
    'pencil-square',
    'person',
    'plus',
    'plus-circle',
    'plus-lg',
    'quote',
    'search',
    'sun',
    'superscript',
    'table',
    'tags',
    'tools',
    'trash3',
    'x',
    'x-lg',
    'youtube'
];

function normalizeIcon(source, name) {
    const viewBox = source.match(/viewBox="([^"]+)"/)?.[1];
    const body = source.match(/<svg[^>]*>([\s\S]*?)<\/svg>/)?.[1]?.trim();
    if (!viewBox || !body) {
        throw new Error(`Could not parse ${name}.svg.`);
    }
    return `  <symbol id="icon-${name}" viewBox="${viewBox}" fill="currentColor">${body}</symbol>`;
}

function main() {
    const sourceDirectory = process.argv[2];
    const outputFile = process.argv[3];
    if (!sourceDirectory || !outputFile) {
        throw new Error('Usage: node scripts/build-site-icon-sprite.js <upstream-svg-dir> <output-file>');
    }

    const symbols = iconNames.map((name) => {
        const sourcePath = path.join(sourceDirectory, `${name}.svg`);
        return normalizeIcon(fs.readFileSync(sourcePath, 'utf8'), name);
    });
    const sprite = [
        '<svg xmlns="http://www.w3.org/2000/svg">',
        '  <!-- Paths derived from Bootstrap Icons 1.11.3; see LICENSE.site-icons.txt. -->',
        ...symbols,
        '</svg>',
        ''
    ].join('\n');

    fs.mkdirSync(path.dirname(outputFile), { recursive: true });
    fs.writeFileSync(outputFile, sprite);
    console.log(`Wrote ${iconNames.length} symbols to ${outputFile}.`);
}

main();
