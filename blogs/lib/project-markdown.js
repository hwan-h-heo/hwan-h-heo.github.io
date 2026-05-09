const { parseMarkdownWithMath } = require('../js/markdown-with-math');

function escapeHtml(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function tokenizeAttributes(source) {
    return String(source || '').match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [];
}

function stripQuotes(value) {
    const text = String(value || '');
    if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
        return text.slice(1, -1);
    }
    return text;
}

function renderAttributes(source) {
    const classes = [];
    const attrs = [];
    let id = '';

    tokenizeAttributes(source).forEach((token) => {
        if (token.startsWith('.') && token.length > 1) {
            classes.push(token.slice(1));
            return;
        }

        if (token.startsWith('#') && token.length > 1) {
            id = token.slice(1);
            return;
        }

        const separatorIndex = token.indexOf('=');
        if (separatorIndex === -1) {
            attrs.push([token, '']);
            return;
        }

        attrs.push([
            token.slice(0, separatorIndex),
            stripQuotes(token.slice(separatorIndex + 1))
        ]);
    });

    const rendered = [];
    if (id) {
        rendered.push(`id="${escapeHtml(id)}"`);
    }
    if (classes.length) {
        rendered.push(`class="${escapeHtml(classes.join(' '))}"`);
    }
    attrs.forEach(([name, value]) => {
        if (!/^[A-Za-z_:][A-Za-z0-9_:.-]*$/.test(name)) {
            return;
        }
        rendered.push(value ? `${name}="${escapeHtml(value)}"` : name);
    });

    return rendered.length ? ` ${rendered.join(' ')}` : '';
}

function renderMarkdownImage(match, alt, src, attrs) {
    return `<img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}"${renderAttributes(attrs)}>`;
}

function normalizeMarkdownExtensions(markdown) {
    return String(markdown || '').replace(
        /!\[([^\]]*)\]\(([^)\s]+)\)\{([^}]*)\}/g,
        renderMarkdownImage
    );
}

function mergeClassAttribute(attrs, className) {
    if (/\sclass\s*=/.test(attrs)) {
        return attrs.replace(/\sclass=(["'])(.*?)\1/i, (match, quote, value) => {
            const classes = new Set(value.split(/\s+/).filter(Boolean));
            classes.add(className);
            return ` class=${quote}${[...classes].join(' ')}${quote}`;
        });
    }

    return `${attrs} class="${className}"`;
}

function getAttributeValue(attrs, name) {
    const match = String(attrs || '').match(new RegExp(`\\s${name}=(["'])(.*?)\\1`, 'i'));
    return match ? match[2] : '';
}

function ensureLazyImageSource(attrs) {
    if (!/\sclass=(["'])(?:(?!\1).)*\blazy-image\b/i.test(attrs) || /\sdata-src=/.test(attrs)) {
        return attrs;
    }

    const src = getAttributeValue(attrs, 'src');
    return src ? `${attrs} data-src="${escapeHtml(src)}"` : attrs;
}

function enhanceProjectMedia(html) {
    return String(html || '')
        .replace(/<img\b([^>]*)>/gi, (match, attrs) => {
            const nextAttrs = ensureLazyImageSource(mergeClassAttribute(attrs, 'img-fluid'));
            return `<img${nextAttrs}>`;
        })
        .replace(/<video\b([^>]*)>/gi, (match, attrs) => {
            let nextAttrs = mergeClassAttribute(attrs, 'img-fluid');
            nextAttrs = mergeClassAttribute(nextAttrs, 'project-video')
                .replace(/\sstyle=["'][^"']*(?:width|max-width)[^"']*["']/gi, '');
            if (!/\splaysinline\b/i.test(nextAttrs)) {
                nextAttrs += ' playsinline';
            }
            return `<video${nextAttrs}>`;
        });
}

function parseProjectMarkdown(markdown, parseMarkdown) {
    const lines = String(markdown || '').split(/\r?\n/);
    const output = [];
    const markdownBuffer = [];
    const stack = [];

    const flushMarkdown = () => {
        const source = markdownBuffer.join('\n').trim();
        markdownBuffer.length = 0;
        if (!source) {
            return;
        }
        output.push(parseMarkdownWithMath(`${normalizeMarkdownExtensions(source)}\n`, parseMarkdown));
    };

    lines.forEach((line) => {
        const openMatch = line.match(/^:::\s*\{([^}]*)\}\s*$/);
        if (openMatch) {
            flushMarkdown();
            output.push(`<div${renderAttributes(openMatch[1])}>`);
            stack.push('div');
            return;
        }

        if (/^:::\s*$/.test(line) && stack.length) {
            flushMarkdown();
            stack.pop();
            output.push('</div>');
            return;
        }

        markdownBuffer.push(line);
    });

    flushMarkdown();
    while (stack.length) {
        stack.pop();
        output.push('</div>');
    }

    return enhanceProjectMedia(output.join('\n'));
}

module.exports = {
    parseProjectMarkdown
};
