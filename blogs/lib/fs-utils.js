const fs = require('fs');
const path = require('path');

function ensureDirSync(dirPath) {
    fs.mkdirSync(dirPath, { recursive: true });
}

function copyRecursiveSync(src, dest) {
    const stats = fs.statSync(src);

    if (stats.isDirectory()) {
        ensureDirSync(dest);
        fs.readdirSync(src).forEach((child) => {
            copyRecursiveSync(path.join(src, child), path.join(dest, child));
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

