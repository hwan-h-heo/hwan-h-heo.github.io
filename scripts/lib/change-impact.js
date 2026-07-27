const { execFileSync } = require('child_process');
const path = require('path');

const { getPostRoute } = require('../../blogs/lib/site-routes');

const DEFAULT_REPO_ROOT = path.join(__dirname, '..', '..');
const BLOG_STATIC_PREFIXES = [
    'blogs/css/',
    'blogs/js/',
    'blogs/3DViewer/',
    'blogs/search/'
];
const ROOT_STATIC_PREFIXES = [
    'assets/',
    'css/',
    'js/'
];
const BLOG_STATIC_FILES = new Set([
    'blogs/index.html',
    'blogs/redirect-old-site.html',
    'blogs/redirect-legacy-posts.html'
]);
const PRIVATE_EDITOR_PREFIXES = [
    'blogs/editor/drafts/',
    'blogs/editor/draft-assets/',
    'blogs/editor/project-snapshots/'
];

function normalizePath(filePath) {
    return String(filePath || '').trim().replace(/\\/g, '/').replace(/^\.\/+/, '');
}

function unique(values) {
    return [...new Set(values.filter(Boolean))];
}

function runGit(args, repoRoot = DEFAULT_REPO_ROOT) {
    try {
        return execFileSync('git', args, {
            cwd: repoRoot,
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'ignore']
        }).trim();
    } catch (error) {
        return '';
    }
}

function splitLines(value) {
    return String(value || '')
        .split(/\r?\n/)
        .map(normalizePath)
        .filter(Boolean);
}

function getWorkingTreeChanges(repoRoot = DEFAULT_REPO_ROOT) {
    return unique([
        ...splitLines(runGit(['diff', '--name-only', 'HEAD'], repoRoot)),
        ...splitLines(runGit(['ls-files', '--others', '--exclude-standard'], repoRoot))
    ]);
}

function getLastCommitChanges(repoRoot = DEFAULT_REPO_ROOT) {
    return splitLines(runGit(['diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD'], repoRoot));
}

function getChangedFiles(options = {}) {
    if (options.changedFiles && options.changedFiles.length) {
        return unique(options.changedFiles.map(normalizePath));
    }

    const repoRoot = options.repoRoot || DEFAULT_REPO_ROOT;
    const workingTreeChanges = getWorkingTreeChanges(repoRoot);
    if (workingTreeChanges.length) {
        return workingTreeChanges;
    }

    return getLastCommitChanges(repoRoot);
}

function addPostTarget(targets, post, language = '') {
    if (!targets.has(post.id)) {
        targets.set(post.id, new Set());
    }

    const languages = language ? [language] : post.languages;
    languages.forEach((entry) => targets.get(post.id).add(entry));
}

function isToolingFile(filePath) {
    const toolingFiles = new Set([
        'package.json',
        'blogs/package.json'
    ]);
    const toolingPrefixes = [
        'scripts/'
    ];

    return toolingFiles.has(filePath) || toolingPrefixes.some((prefix) => filePath.startsWith(prefix));
}

function isIgnoredFile(filePath) {
    return filePath === 'package-lock.json'
        || filePath === 'blogs/package-lock.json'
        || filePath.startsWith('blogs/dist/');
}

function isStaticCopyFile(filePath) {
    if (BLOG_STATIC_FILES.has(filePath)) {
        return true;
    }

    if (PRIVATE_EDITOR_PREFIXES.some((prefix) => filePath.startsWith(prefix))) {
        return false;
    }

    if (filePath.startsWith('blogs/editor/')) {
        return true;
    }

    return BLOG_STATIC_PREFIXES.some((prefix) => filePath.startsWith(prefix))
        || ROOT_STATIC_PREFIXES.some((prefix) => filePath.startsWith(prefix));
}

function analyzeChangedFiles(changedFiles, siteData) {
    const postById = new Map(siteData.posts.map((post) => [post.id, post]));
    const postTargets = new Map();
    const staticFiles = [];
    const fullBuildReasons = [];

    unique(changedFiles.map(normalizePath)).forEach((filePath) => {
        if (!filePath || isIgnoredFile(filePath) || isToolingFile(filePath)) {
            return;
        }

        if (isStaticCopyFile(filePath)) {
            staticFiles.push(filePath);
            return;
        }

        const postMatch = filePath.match(/^blogs\/posts\/([^/]+)\/(.+)$/);
        if (postMatch) {
            const [, postId, relativePath] = postMatch;
            const post = postById.get(postId);
            if (!post) {
                fullBuildReasons.push(`${filePath} is under an unknown post.`);
                return;
            }

            const contentMatch = relativePath.match(/^content-(eng|kor)\.md$/);
            if (contentMatch) {
                addPostTarget(postTargets, post, contentMatch[1]);
                return;
            }

            if (relativePath.startsWith('assets/')) {
                addPostTarget(postTargets, post);
                return;
            }

            fullBuildReasons.push(`${filePath} is a post structural file.`);
            return;
        }

        fullBuildReasons.push(`${filePath} may affect global output.`);
    });

    if (fullBuildReasons.length || (postTargets.size === 0 && staticFiles.length === 0)) {
        return {
            strategy: 'full',
            reasons: fullBuildReasons.length ? fullBuildReasons : ['No incremental-safe site changes were detected.'],
            postTargets,
            staticFiles: unique(staticFiles),
            routes: []
        };
    }

    const routes = [];
    postTargets.forEach((languages, postId) => {
        const post = postById.get(postId);
        [...languages].forEach((language) => routes.push(getPostRoute(post, language)));
    });

    return {
        strategy: 'incremental',
        reasons: [],
        postTargets,
        staticFiles: unique(staticFiles),
        routes: unique(routes)
    };
}

function serializeChangedFiles(files) {
    return unique(files.map(normalizePath)).join(',');
}

module.exports = {
    analyzeChangedFiles,
    getChangedFiles,
    getLastCommitChanges,
    getWorkingTreeChanges,
    normalizePath,
    serializeChangedFiles,
    unique
};
