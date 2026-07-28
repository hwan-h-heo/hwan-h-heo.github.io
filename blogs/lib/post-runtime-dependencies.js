const LEGACY_CONTENT_DELIMITER = '--- 여기부터 실제 콘텐츠 ---';

const DEPENDENCY_ALIASES = new Map([
    ['math', 'katex'],
    ['latex', 'katex'],
    ['katex', 'katex'],
    ['code', 'prism'],
    ['syntax', 'prism'],
    ['syntax-highlight', 'prism'],
    ['syntax-highlighting', 'prism'],
    ['prism', 'prism'],
    ['bootstrap', 'bootstrap'],
    ['bootstrap-js', 'bootstrap'],
    ['bootstrap-modal', 'bootstrap'],
    ['bootstrap-collapse', 'bootstrap'],
    ['model-viewer', 'modelViewer'],
    ['modelviewer', 'modelViewer'],
    ['google-model-viewer', 'modelViewer'],
    ['three', 'three'],
    ['threejs', 'three'],
    ['three-js', 'three'],
    ['tween', 'tween'],
    ['tweenjs', 'tween'],
    ['tween-js', 'tween'],
    ['simple-model-viewer', 'simpleModelViewer'],
    ['simplemodelviewer', 'simpleModelViewer'],
    ['custom-model-viewer', 'simpleModelViewer'],
    ['gaussian-splats', 'gaussianSplats'],
    ['gaussian-splatting', 'gaussianSplats'],
    ['gaussiansplats3d', 'gaussianSplats']
]);

const RUNTIME_FEATURES = [
    'katex',
    'prism',
    'bootstrap',
    'modelViewer',
    'three',
    'tween',
    'simpleModelViewer',
    'gaussianSplats'
];

function normalizeDependencyName(value) {
    const key = String(value || '')
        .trim()
        .replace(/^['"]|['"]$/g, '')
        .toLowerCase();

    return DEPENDENCY_ALIASES.get(key) || null;
}

function parseInlineList(value) {
    const normalized = String(value || '').trim();
    if (!normalized) {
        return [];
    }

    const listSource = normalized.startsWith('[') && normalized.endsWith(']')
        ? normalized.slice(1, -1)
        : normalized;

    return listSource
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);
}

function parseSimplePreamble(preamble) {
    const attributes = {};
    const lines = String(preamble || '').split(/\r?\n/);

    for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index];
        const match = line.match(/^([A-Za-z][\w-]*):\s*(.*)$/);
        if (!match) {
            continue;
        }

        const key = match[1];
        const value = match[2].trim();

        if (key === 'dependencies' || key === 'features' || key === 'runtimeDependencies') {
            const items = parseInlineList(value);

            if (!value) {
                let cursor = index + 1;
                while (cursor < lines.length) {
                    const itemMatch = lines[cursor].match(/^\s*-\s+(.+?)\s*$/);
                    if (!itemMatch) {
                        break;
                    }
                    items.push(itemMatch[1]);
                    cursor += 1;
                }
                index = cursor - 1;
            }

            attributes.dependencies = [
                ...(attributes.dependencies || []),
                ...items
            ];
            continue;
        }

        attributes[key] = value;
    }

    return attributes;
}

function parsePostMarkdownSource(markdown) {
    const source = String(markdown || '');

    if (source.startsWith('---\n') || source.startsWith('---\r\n')) {
        const delimiterMatch = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
        if (delimiterMatch) {
            return {
                frontmatter: parseSimplePreamble(delimiterMatch[1]),
                content: source.slice(delimiterMatch[0].length).trim()
            };
        }
    }

    const delimiterIndex = source.indexOf(LEGACY_CONTENT_DELIMITER);
    if (delimiterIndex === -1) {
        return {
            frontmatter: {},
            content: source
        };
    }

    return {
        frontmatter: parseSimplePreamble(source.slice(0, delimiterIndex)),
        content: source.slice(delimiterIndex + LEGACY_CONTENT_DELIMITER.length).trim()
    };
}

function stripCodeLikeContent(source) {
    return String(source || '')
        .replace(/```[\s\S]*?```/g, ' ')
        .replace(/~~~[\s\S]*?~~~/g, ' ')
        .replace(/<pre\b[\s\S]*?<\/pre>/gi, ' ')
        .replace(/`[^`\n]*`/g, ' ');
}

function hasMath(contentSource) {
    const source = stripCodeLikeContent(contentSource);
    return /\$\$|\\\(|\\\[/.test(source)
        || /(^|[^\\])\$[^\n$]{1,240}\$/.test(source);
}

function hasCodeBlocks(contentSource, contentHtml) {
    return /```|~~~|<pre\b|class=(["'])[^"']*\blanguage-[^"']*\1/i.test(contentSource)
        || /<pre\b|class=(["'])[^"']*\blanguage-[^"']*\1/i.test(contentHtml);
}

function hasBootstrapComponents(contentHtml) {
    const html = stripCodeLikeContent(contentHtml);
    return /data-bs-|class=(["'])[^"']*\b(?:modal|accordion|accordion-collapse|accordion-button)\b[^"']*\1/i.test(html);
}

function createFeatureObject(features) {
    const featureSet = new Set(features);

    if (featureSet.has('simpleModelViewer')) {
        featureSet.add('three');
        featureSet.add('tween');
    }

    if (featureSet.has('gaussianSplats')) {
        featureSet.add('three');
    }

    return Object.fromEntries(RUNTIME_FEATURES.map((feature) => [feature, featureSet.has(feature)]));
}

function inferPostRuntimeFeatures({ post, contentSource, contentHtml, frontmatter }) {
    const features = new Set();

    [
        ...(frontmatter?.dependencies || []),
        ...(Array.isArray(post.dependencies) ? post.dependencies : []),
        ...(Array.isArray(post.runtimeDependencies) ? post.runtimeDependencies : [])
    ].forEach((dependency) => {
        const feature = normalizeDependencyName(dependency);
        if (feature) {
            features.add(feature);
        }
    });

    if (hasMath(contentSource)) {
        features.add('katex');
    }

    if (hasCodeBlocks(contentSource, contentHtml)) {
        features.add('prism');
    }

    if (/<model-viewer(?:\s|>)/i.test(contentHtml)) {
        features.add('modelViewer');
    }

    if (/<simple-model-viewer(?:\s|>)/i.test(contentHtml)) {
        features.add('simpleModelViewer');
    }

    if (/id=(["'])threeCanvas\1|id=(["'])canvasContainer\2/i.test(contentHtml) || post.id === '240917_3djs') {
        features.add('gaussianSplats');
    }

    if (hasBootstrapComponents(contentHtml)) {
        features.add('bootstrap');
    }

    return createFeatureObject(features);
}

module.exports = {
    inferPostRuntimeFeatures,
    parsePostMarkdownSource
};
