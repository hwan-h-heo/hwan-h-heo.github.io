const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const repoRoot = path.join(__dirname, '..');
const referenceExtensions = /\.(css|html?|js|json|md|txt|xml)$/i;
const ignoredReferenceRoots = [
    'blogs/dist/'
];
const ignoredAssetNames = new Set(['LICENSE.txt']);
const retiredStylesheets = [
    'assets/css/main.css',
    'assets/css/project-legacy.css',
    'assets/css/used.css',
    'blogs/css/blog_post_specific.css',
    'blogs/css/used.css',
    'css/blog_style.css',
    'css/styles.css'
];

function normalizePath(filePath) {
    return filePath.split(path.sep).join('/');
}

function listSourceFiles() {
    const output = execFileSync(
        'git',
        ['ls-files', '--cached', '--others', '--exclude-standard'],
        { cwd: repoRoot, encoding: 'utf8' }
    );

    return output
        .split(/\r?\n/)
        .map((filePath) => normalizePath(filePath.trim()))
        .filter(Boolean)
        .filter((filePath) => referenceExtensions.test(filePath))
        .filter((filePath) => !ignoredReferenceRoots.some((root) => filePath.startsWith(root)))
        .filter((filePath) => fs.existsSync(path.join(repoRoot, filePath)));
}

function collectFiles(directory, options = {}, files = []) {
    const { ignoredDirectories = new Set() } = options;
    if (!fs.existsSync(directory)) {
        return files;
    }

    fs.readdirSync(directory, { withFileTypes: true }).forEach((entry) => {
        if (entry.isDirectory() && ignoredDirectories.has(entry.name)) {
            return;
        }

        const filePath = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            collectFiles(filePath, options, files);
        } else if (!ignoredAssetNames.has(entry.name)) {
            files.push(filePath);
        }
    });

    return files;
}

function collectLocalizedAssetFiles(parentDirectory) {
    if (!fs.existsSync(parentDirectory)) {
        return [];
    }

    return fs.readdirSync(parentDirectory, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .flatMap((entry) => collectFiles(path.join(parentDirectory, entry.name, 'assets')));
}

function collectAuditedAssets() {
    return [
        ...collectFiles(path.join(repoRoot, 'assets'), {
            ignoredDirectories: new Set(['css', 'js', 'vendor'])
        }),
        ...collectFiles(path.join(repoRoot, 'blogs', '3DViewer', 'assets')),
        ...collectLocalizedAssetFiles(path.join(repoRoot, 'blogs', 'posts')),
        ...collectLocalizedAssetFiles(path.join(repoRoot, 'projects'))
    ];
}

const sourceFiles = listSourceFiles();
const sourceText = sourceFiles
    .map((filePath) => fs.readFileSync(path.join(repoRoot, filePath), 'utf8'))
    .join('\n');
const errors = [];

collectAuditedAssets().forEach((filePath) => {
    const fileName = path.basename(filePath);
    if (!sourceText.includes(fileName)) {
        errors.push(`Unreferenced asset: ${normalizePath(path.relative(repoRoot, filePath))}`);
    }
});

sourceFiles.forEach((filePath) => {
    if (filePath === 'scripts/audit-static-assets.js') {
        return;
    }

    const text = fs.readFileSync(path.join(repoRoot, filePath), 'utf8');
    retiredStylesheets.forEach((stylesheet) => {
        if (text.includes(stylesheet)) {
            errors.push(`Retired stylesheet reference in ${filePath}: ${stylesheet}`);
        }
    });
});

if (errors.length > 0) {
    console.error('Static asset audit failed.');
    errors.sort().forEach((error) => console.error(`- ${error}`));
    process.exit(1);
}

console.log(
    `Static asset audit passed: ${collectAuditedAssets().length} media files, `
    + `${sourceFiles.length} source files checked.`
);
