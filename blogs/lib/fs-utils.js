const fs = require('fs');
const path = require('path');

function ensureDirSync(dirPath) {
    fs.mkdirSync(dirPath, { recursive: true });
}

function copyRecursiveSync(src, dest, options = {}) {
    const shouldCopy = options.shouldCopy || (() => true);
    if (!shouldCopy(src)) {
        return;
    }

    const stats = fs.statSync(src);

    if (stats.isDirectory()) {
        ensureDirSync(dest);
        fs.readdirSync(src).forEach((child) => {
            copyRecursiveSync(path.join(src, child), path.join(dest, child), options);
        });
        return;
    }

    ensureDirSync(path.dirname(dest));
    fs.copyFileSync(src, dest);
}

module.exports = {
    copyRecursiveSync,
    ensureDirSync
};
