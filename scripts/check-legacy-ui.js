const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const repoRoot = path.join(__dirname, '..');
const distRoot = path.join(repoRoot, 'blogs', 'dist');
const textExtensions = new Set([
    '.css', '.html', '.js', '.json', '.md', '.mjs', '.svg', '.txt', '.xml'
]);
const ignoredDirectories = new Set(['.git', 'node_modules', 'artifacts']);
const authoredRuntimeRoots = [
    'assets/',
    'blogs/',
    'content/',
    'css/',
    'js/',
    'projects/',
    'scripts/'
];
const authoredRuntimeFiles = new Set(['index.html']);
const selfPath = 'scripts/check-legacy-ui.js';
const permittedProvenanceFiles = new Set([
    'assets/icons/LICENSE.site-icons.txt',
    'assets/icons/site-icons.svg',
    'scripts/build-site-icon-sprite.js',
    'blogs/dist/assets/icons/LICENSE.site-icons.txt',
    'blogs/dist/assets/icons/site-icons.svg'
]);
const permittedEditorDataFiles = new Set([
    'blogs/editor/editor.js',
    'blogs/editor-server.js',
    'blogs/dist/blogs/editor/editor.js'
]);
const prohibitedClassPatterns = [
    /^container(?:-(?:fluid|sm|md|lg|xl|xxl))?$/,
    /^row$/,
    /^col(?:-(?:sm|md|lg|xl|xxl)(?:-(?:auto|[1-9]|1[0-2]))?|-(?:auto|[1-9]|1[0-2]))?$/,
    /^row-cols(?:-(?:sm|md|lg|xl|xxl))?-(?:auto|[1-6])$/,
    /^offset(?:-(?:sm|md|lg|xl|xxl))?-(?:0|[1-9]|1[0-2])$/,
    /^g[xy]?-[0-5]$/,
    /^d-(?:(?:sm|md|lg|xl|xxl)-)?(?:none|inline|inline-block|block|grid|table|flex|inline-flex)$/,
    /^flex-(?:(?:sm|md|lg|xl|xxl)-)?(?:row|column|row-reverse|column-reverse|wrap|nowrap|fill|grow-[01]|shrink-[01])$/,
    /^justify-content-(?:(?:sm|md|lg|xl|xxl)-)?(?:start|end|center|between|around|evenly)$/,
    /^align-items-(?:(?:sm|md|lg|xl|xxl)-)?(?:start|end|center|baseline|stretch)$/,
    /^(?:m|p)[trblxyse]?-(?:(?:sm|md|lg|xl|xxl)-)?(?:0|1|2|3|4|5|auto)$/,
    /^position-(?:static|relative|absolute|fixed|sticky)$/,
    /^img-fluid$/,
    /^rounded(?:-circle|-pill)?$/,
    /^text-(?:center|start|end|muted|white|white-50|break|nowrap)$/,
    /^(?:w|h)-100$/,
    /^navbar(?:-[a-z0-9-]+)?$/,
    /^nav-tabs$/,
    /^tab-pane$/,
    /^accordion(?:-[a-z0-9-]+)?$/,
    /^btn(?:-[a-z0-9-]+)?$/,
    /^badge$/,
    /^bi(?:-[a-z0-9-]+)?$/
];

function normalizePath(filePath) {
    return filePath.split(path.sep).join('/');
}

function isTextFile(filePath) {
    return textExtensions.has(path.extname(filePath).toLowerCase());
}

function collectFiles(directory, files = []) {
    if (!fs.existsSync(directory)) {
        return files;
    }

    fs.readdirSync(directory, { withFileTypes: true }).forEach((entry) => {
        if (entry.isDirectory() && ignoredDirectories.has(entry.name)) {
            return;
        }
        const filePath = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            collectFiles(filePath, files);
        } else if (isTextFile(filePath)) {
            files.push(filePath);
        }
    });
    return files;
}

function listAuthoredRuntimeFiles() {
    const output = execFileSync(
        'git',
        ['ls-files', '--cached', '--others', '--exclude-standard'],
        { cwd: repoRoot, encoding: 'utf8' }
    );

    return output
        .split(/\r?\n/)
        .map((filePath) => normalizePath(filePath.trim()))
        .filter(Boolean)
        .filter((filePath) => filePath !== selfPath)
        .filter((filePath) => !filePath.startsWith('blogs/dist/'))
        .filter((filePath) => authoredRuntimeFiles.has(filePath)
            || authoredRuntimeRoots.some((root) => filePath.startsWith(root)))
        .map((filePath) => path.join(repoRoot, filePath))
        .filter((filePath) => fs.existsSync(filePath) && isTextFile(filePath));
}

function lineNumberAt(source, index) {
    return source.slice(0, index).split('\n').length;
}

function addMatch(errors, filePath, source, index, message) {
    errors.push(`${normalizePath(path.relative(repoRoot, filePath))}:${lineNumberAt(source, index)} ${message}`);
}

function isProhibitedClass(token) {
    return prohibitedClassPatterns.some((pattern) => pattern.test(token));
}

function checkClassTokens(errors, filePath, source) {
    const inspected = new Set();
    const inspectList = (value, index, context) => {
        String(value).split(/\s+/).filter(Boolean).forEach((token) => {
            const key = `${index}:${token}`;
            if (inspected.has(key) || !isProhibitedClass(token)) {
                return;
            }
            inspected.add(key);
            addMatch(errors, filePath, source, index, `uses retired UI class "${token}" in ${context}.`);
        });
    };

    for (const match of source.matchAll(/\bclass(?:Name)?\s*=\s*(["'`])([\s\S]*?)\1/g)) {
        inspectList(match[2], match.index, 'class markup');
    }
    for (const match of source.matchAll(/\.classList\.(?:add|remove|toggle|contains)\(\s*(["'])([^"']+)\1/g)) {
        inspectList(match[2], match.index, 'classList call');
    }
    for (const match of source.matchAll(/\.setAttribute\(\s*(["'])class\1\s*,\s*(["'`])([\s\S]*?)\2/g)) {
        inspectList(match[3], match.index, 'setAttribute call');
    }

    if (path.extname(filePath).toLowerCase() === '.css') {
        for (const match of source.matchAll(/\.(-?[_a-zA-Z]+[_a-zA-Z0-9-]*)/g)) {
            if (isProhibitedClass(match[1])) {
                addMatch(errors, filePath, source, match.index, `defines retired UI selector ".${match[1]}".`);
            }
        }
    }
}

function isPermittedEditorDataLine(line) {
    return /(?:state\.bootstrap|\bbootstrap\s*:|load(?:Public)?Bootstrap|editor-bootstrap|bootstrap data|bootstrap failed|buildBootstrapPayload)/i.test(line);
}

function isPermittedProvenanceLine(line) {
    return /(?:Copyright \(c\) 2019-2024 The Bootstrap Authors|Paths derived from Bootstrap Icons 1\.11\.3)/i.test(line);
}

function checkRetiredReferences(errors, filePath, source) {
    const relativePath = normalizePath(path.relative(repoRoot, filePath));
    const provenanceFile = permittedProvenanceFiles.has(relativePath);
    const editorDataFile = permittedEditorDataFiles.has(relativePath);
    const rules = [
        { pattern: /\biportfolio\b/gi, label: 'iPortfolio reference' },
        {
            pattern: /\bAOS\s*\.\s*(?:init|refresh|refreshHard)\b|data-aos(?:-[a-z0-9-]+)?|(?:assets\/vendor\/aos|\/aos(?:@[\d.]+)?\/[^\s"']*\.js)/gi,
            label: 'AOS runtime reference'
        },
        { pattern: /data-bs-[a-z0-9-]+/gi, label: 'data-bs attribute' },
        { pattern: /--bs-[a-z0-9-]+/gi, label: 'framework CSS variable' },
        { pattern: /bootstrap-icons(?:\.[a-z0-9]+)?/gi, label: 'icon-font reference' },
        { pattern: /\bbootstrap\b/gi, label: 'Bootstrap runtime reference' }
    ];

    rules.forEach(({ pattern, label }) => {
        for (const match of source.matchAll(pattern)) {
            const lineStart = source.lastIndexOf('\n', match.index) + 1;
            const lineEnd = source.indexOf('\n', match.index);
            const line = source.slice(lineStart, lineEnd === -1 ? source.length : lineEnd);
            const permitted = (provenanceFile
                    && /bootstrap/i.test(match[0])
                    && isPermittedProvenanceLine(line))
                || (editorDataFile && /bootstrap/i.test(match[0]) && isPermittedEditorDataLine(line));
            if (!permitted) {
                addMatch(errors, filePath, source, match.index, `contains ${label}.`);
            }
        }
    });

    for (const match of source.matchAll(/(?:font-family\s*:\s*["']?bootstrap icons|bootstrap-icons\.(?:woff2?|ttf|otf)|assets\/vendor\/(?:bootstrap|bootstrap-icons|aos)|https?:\/\/[^\s"']*bootstrap(?:@|\/|\.(?:min\.)?(?:css|js))|\b(?:new\s+)?Bootstrap\.(?:Modal|Collapse|Dropdown|Tab|Toast|Tooltip|Popover|Carousel|Offcanvas)\b)/gi)) {
        addMatch(errors, filePath, source, match.index, 'contains a retired vendor/font/component reference.');
    }
}

function checkVendorTree(errors) {
    const vendorRoot = path.join(repoRoot, 'assets', 'vendor');
    collectFiles(vendorRoot).forEach((filePath) => {
        const relativePath = normalizePath(path.relative(repoRoot, filePath));
        if (/(?:^|\/)(?:bootstrap|bootstrap-icons|aos)(?:\/|$)/i.test(relativePath)
            || /bootstrap-icons\.(?:css|woff2?|ttf|otf)$/i.test(relativePath)) {
            errors.push(`${relativePath} is a retired vendor asset.`);
        }
    });
}

function checkIconSymbols(errors, files) {
    const spritePath = path.join(repoRoot, 'assets', 'icons', 'site-icons.svg');
    if (!fs.existsSync(spritePath)) {
        errors.push('assets/icons/site-icons.svg is missing.');
        return;
    }

    const sprite = fs.readFileSync(spritePath, 'utf8');
    const symbols = Array.from(sprite.matchAll(/<symbol\s+id="([^"]+)"/g), (match) => match[1]);
    const symbolIds = new Set(symbols);
    if (symbols.length === 0 || symbolIds.size !== symbols.length) {
        errors.push('assets/icons/site-icons.svg must contain unique symbol ids.');
    }

    const referencedIds = new Set();
    files.forEach((filePath) => {
        const source = fs.readFileSync(filePath, 'utf8');
        for (const match of source.matchAll(/site-icons\.svg#(icon-[a-z0-9-]+)/g)) {
            referencedIds.add(match[1]);
        }
    });
    const missing = [...referencedIds].filter((id) => !symbolIds.has(id)).sort();
    if (missing.length > 0) {
        errors.push(`Site icon hrefs reference missing symbols: ${missing.join(', ')}.`);
    }
}

function main() {
    if (!fs.existsSync(distRoot)) {
        throw new Error('blogs/dist is missing; run npm run build before check:legacy-ui.');
    }

    const authoredFiles = listAuthoredRuntimeFiles();
    const generatedFiles = collectFiles(distRoot);
    const files = [...new Set([...authoredFiles, ...generatedFiles])];
    const errors = [];

    files.forEach((filePath) => {
        const source = fs.readFileSync(filePath, 'utf8');
        checkRetiredReferences(errors, filePath, source);
        checkClassTokens(errors, filePath, source);
    });
    checkVendorTree(errors);
    checkIconSymbols(errors, files);

    if (errors.length > 0) {
        console.error('Legacy UI check failed.');
        [...new Set(errors)].sort().forEach((error) => console.error(`- ${error}`));
        process.exit(1);
    }

    console.log(
        `Legacy UI check passed: ${authoredFiles.length} authored runtime files and `
        + `${generatedFiles.length} generated files; 0 retired UI dependencies.`
    );
}

main();
