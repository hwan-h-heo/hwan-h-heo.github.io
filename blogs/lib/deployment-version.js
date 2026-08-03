const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const VERSION_FILE_NAME = 'deployment-version.json';
const VERSION_META_PATTERN = /\s*<meta\s+name=["']site-deploy-version["'][^>]*>\s*/gi;
const REFRESH_SCRIPT_PATTERN = /\s*<script\s+defer\s+src=["']\/assets\/js\/deploy-refresh\.js(?:\?[^"']*)?["']><\/script>\s*/gi;

function collectFiles(directory, files = []) {
    fs.readdirSync(directory, { withFileTypes: true }).forEach((entry) => {
        const filePath = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            collectFiles(filePath, files);
        } else if (entry.isFile()) {
            files.push(filePath);
        }
    });
    return files;
}

function stripDeploymentStamp(source) {
    return source
        .replace(VERSION_META_PATTERN, '\n')
        .replace(REFRESH_SCRIPT_PATTERN, '\n')
        .replace(/((?:href|src)=["'][^"']+\.(?:css|js))\?v=[^"'#]+/gi, '$1');
}

function computeDeploymentVersion(distDir) {
    const versionPath = path.join(distDir, VERSION_FILE_NAME);
    const files = collectFiles(distDir)
        .filter((filePath) => filePath !== versionPath)
        .sort((left, right) => left.localeCompare(right));
    const hash = crypto.createHash('sha256');

    files.forEach((filePath) => {
        const relativePath = path.relative(distDir, filePath).replace(/\\/g, '/');
        let contents = fs.readFileSync(filePath);
        if (filePath.endsWith('.html')) {
            contents = Buffer.from(stripDeploymentStamp(contents.toString('utf8')));
        }
        hash.update(relativePath);
        hash.update('\0');
        hash.update(contents);
        hash.update('\0');
    });

    return hash.digest('hex').slice(0, 16);
}

function stampDeploymentVersion({ distDir }) {
    const version = computeDeploymentVersion(distDir);
    const versionPath = path.join(distDir, VERSION_FILE_NAME);
    fs.writeFileSync(versionPath, `${JSON.stringify({ version })}\n`);

    let stampedPages = 0;
    collectFiles(distDir)
        .filter((filePath) => filePath.endsWith('.html'))
        .forEach((filePath) => {
            const relativePath = path.relative(distDir, filePath).replace(/\\/g, '/');
            if (relativePath.startsWith('blogs/editor/')) {
                return;
            }

            const source = stripDeploymentStamp(fs.readFileSync(filePath, 'utf8'));
            if (!source.includes('</head>')) {
                return;
            }

            const stamp = [
                `    <meta name="site-deploy-version" content="${version}">`,
                '    <script defer src="/assets/js/deploy-refresh.js"></script>',
                '</head>'
            ].join('\n');
            fs.writeFileSync(filePath, source.replace('</head>', stamp));
            stampedPages += 1;
        });

    console.log(`Stamped deployment version ${version} across ${stampedPages} pages.`);
    return version;
}

module.exports = {
    computeDeploymentVersion,
    stampDeploymentVersion
};
