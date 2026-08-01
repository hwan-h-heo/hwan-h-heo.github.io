const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const VERSIONED_EXTENSIONS = /\.(?:css|js)$/i;

function collectHtmlFiles(directory, files = []) {
    if (!fs.existsSync(directory)) {
        return files;
    }

    fs.readdirSync(directory, { withFileTypes: true }).forEach((entry) => {
        const filePath = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            collectHtmlFiles(filePath, files);
        } else if (entry.isFile() && entry.name.endsWith('.html')) {
            files.push(filePath);
        }
    });

    return files;
}

function isExternalUrl(value) {
    return /^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(value);
}

function getVersionedAsset(value, htmlPath, distDir, hashCache) {
    if (!value || isExternalUrl(value)) {
        return '';
    }

    const match = value.match(/^([^?#]+)(?:\?[^#]*)?(#.*)?$/);
    if (!match || !VERSIONED_EXTENSIONS.test(match[1])) {
        return '';
    }

    const assetUrl = match[1];
    const fragment = match[2] || '';
    const assetPath = assetUrl.startsWith('/')
        ? path.join(distDir, assetUrl.replace(/^\/+/, ''))
        : path.resolve(path.dirname(htmlPath), assetUrl);
    const relativePath = path.relative(distDir, assetPath);

    if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
        return '';
    }
    if (!fs.existsSync(assetPath) || !fs.statSync(assetPath).isFile()) {
        return '';
    }

    let version = hashCache.get(assetPath);
    if (!version) {
        version = crypto
            .createHash('sha256')
            .update(fs.readFileSync(assetPath))
            .digest('hex')
            .slice(0, 10);
        hashCache.set(assetPath, version);
    }

    return `${assetUrl}?v=${version}${fragment}`;
}

function versionStaticAssetReferences({ distDir }) {
    const hashCache = new Map();
    let updatedPages = 0;
    let updatedReferences = 0;

    collectHtmlFiles(distDir).forEach((htmlPath) => {
        const source = fs.readFileSync(htmlPath, 'utf8');
        let pageReferences = 0;
        const html = source.replace(
            /\b(href|src)=(['"])([^'"]+)\2/g,
            (fullMatch, attribute, quote, value) => {
                const versioned = getVersionedAsset(value, htmlPath, distDir, hashCache);
                if (!versioned) {
                    return fullMatch;
                }
                pageReferences += 1;
                return `${attribute}=${quote}${versioned}${quote}`;
            }
        );

        if (html !== source) {
            fs.writeFileSync(htmlPath, html);
            updatedPages += 1;
            updatedReferences += pageReferences;
        }
    });

    console.log(
        `Versioned static entry assets: ${updatedReferences} CSS/JS references across ${updatedPages} pages.`
    );
}

function findUnversionedStaticAssetReferences({ distDir }) {
    const issues = [];

    collectHtmlFiles(distDir).forEach((htmlPath) => {
        const html = fs.readFileSync(htmlPath, 'utf8');
        const attributes = html.matchAll(/\b(?:href|src)=(['"])([^'"]+)\1/g);
        for (const match of attributes) {
            const value = match[2];
            if (isExternalUrl(value)) {
                continue;
            }
            const assetPath = value.split(/[?#]/, 1)[0];
            if (!VERSIONED_EXTENSIONS.test(assetPath)) {
                continue;
            }
            if (!/\?v=[a-f\d]{10}(?:#|$)/i.test(value)) {
                issues.push(`${path.relative(distDir, htmlPath)}: ${value}`);
            }
        }
    });

    return issues;
}

module.exports = {
    findUnversionedStaticAssetReferences,
    versionStaticAssetReferences
};
