function stripHtml(value) {
    return String(value || '')
        .replace(/<[^>]*>/g, ' ')
        .replace(/&nbsp;/gi, ' ')
        .replace(/&amp;/gi, '&')
        .replace(/&quot;/gi, '"')
        .replace(/&#39;/gi, "'")
        .replace(/\s+/g, ' ')
        .trim();
}

function escapeAttribute(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function isGenericAlt(value) {
    return /^(?:alt description|figure|image|image\.(?:gif|jpe?g|png|webp)|img|img alt|teaser)?$/i.test(
        String(value || '').trim()
    );
}

function setAttribute(attributes, name, value) {
    const pattern = new RegExp(`\\s${name}=(["'])(.*?)\\1`, 'i');
    const serialized = `${name}="${escapeAttribute(value)}"`;

    if (pattern.test(attributes)) {
        return attributes.replace(pattern, ` ${serialized}`);
    }

    return `${attributes} ${serialized}`;
}

function ensureBooleanAttribute(attributes, name, value) {
    const pattern = new RegExp(`\\s${name}(?:\\s|=|$)`, 'i');
    return pattern.test(attributes) ? attributes : `${attributes} ${name}="${value}"`;
}

function normalizeContentImageAccessibility(html, options = {}) {
    const title = stripHtml(options.title) || 'Article';
    let currentSection = '';
    let figureIndex = 0;
    const sectionFigures = new Map();

    return String(html || '').replace(
        /<h([2-6])\b[^>]*>([\s\S]*?)<\/h\1>|<img\b([^>]*)>/gi,
        (match, headingLevel, headingHtml, imageAttributes) => {
            if (headingLevel) {
                currentSection = stripHtml(headingHtml);
                return match;
            }

            let attributes = String(imageAttributes || '').replace(/\s*\/\s*$/, '');
            const altMatch = attributes.match(/\salt=(["'])(.*?)\1/i);
            const existingAlt = altMatch ? stripHtml(altMatch[2]) : '';

            if (!altMatch || isGenericAlt(existingAlt)) {
                figureIndex += 1;
                const sectionKey = currentSection || title;
                const sectionIndex = (sectionFigures.get(sectionKey) || 0) + 1;
                sectionFigures.set(sectionKey, sectionIndex);

                const context = currentSection && currentSection.toLowerCase() !== title.toLowerCase()
                    ? `${title} — ${currentSection}`
                    : title;
                const suffix = sectionIndex > 1 ? `, figure ${sectionIndex}` : '';
                attributes = setAttribute(attributes, 'alt', `${context}${suffix}`);
            }

            attributes = ensureBooleanAttribute(attributes, 'loading', 'lazy');
            attributes = ensureBooleanAttribute(attributes, 'decoding', 'async');
            return `<img${attributes}>`;
        }
    );
}

module.exports = {
    normalizeContentImageAccessibility
};
