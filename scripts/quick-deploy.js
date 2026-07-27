#!/usr/bin/env node

const { spawnSync } = require('child_process');
const path = require('path');

const { loadSiteData } = require('../blogs/lib/site-data');
const {
    analyzeChangedFiles,
    getChangedFiles,
    serializeChangedFiles,
    unique
} = require('./lib/change-impact');

const repoRoot = path.join(__dirname, '..');

function parseArguments(argv) {
    const options = {
        changedFiles: [],
        full: false,
        routes: []
    };

    argv.forEach((argument) => {
        if (argument === '--full') {
            options.full = true;
            return;
        }

        if (argument.startsWith('--changed=')) {
            options.changedFiles.push(...argument.slice('--changed='.length).split(',').filter(Boolean));
            return;
        }

        if (argument.startsWith('--route=')) {
            options.routes.push(normalizeRoute(argument.slice('--route='.length)));
        }
    });

    return options;
}

function normalizeRoute(route) {
    const trimmed = String(route || '').trim();
    if (!trimmed) {
        return '';
    }

    return trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
}

function run(command, args) {
    console.log(`\n$ ${[command, ...args].join(' ')}`);
    const result = spawnSync(command, args, {
        cwd: repoRoot,
        stdio: 'inherit'
    });

    if (result.error) {
        throw result.error;
    }

    if (result.status !== 0) {
        process.exit(result.status || 1);
    }
}

function runFullDeploy() {
    console.log('Running full verification and deploy.');
    run('npm', ['run', 'verify']);
    run('npm', ['--prefix', 'blogs', 'run', 'deploy:dist']);
}

function getStaticSmokeRoutes(staticFiles) {
    const routes = [];

    if (staticFiles.some((filePath) => filePath.startsWith('blogs/'))) {
        routes.push('/blogs/');
    }

    if (staticFiles.some((filePath) => /^(assets|css|js)\//.test(filePath))) {
        routes.push('/');
    }

    return routes;
}

function main() {
    const options = parseArguments(process.argv.slice(2));
    if (options.full) {
        runFullDeploy();
        return;
    }

    const siteData = loadSiteData();
    const changedFiles = getChangedFiles({
        changedFiles: options.changedFiles,
        repoRoot
    });
    const impact = analyzeChangedFiles(changedFiles, siteData);

    console.log(`Quick deploy changed files: ${serializeChangedFiles(changedFiles) || '(none)'}`);

    if (impact.strategy !== 'incremental') {
        console.log('Quick deploy fell back to the full deploy path.');
        impact.reasons.forEach((reason) => console.log(`- ${reason}`));
        runFullDeploy();
        return;
    }

    const routes = unique([
        ...(impact.routes || []),
        ...getStaticSmokeRoutes(impact.staticFiles || []),
        ...options.routes
    ]);

    run('npm', ['run', 'check:content']);
    run('npm', ['run', 'check:assets']);
    run('npm', ['--prefix', 'blogs', 'run', 'build:incremental', '--', `--changed=${serializeChangedFiles(changedFiles)}`]);

    (routes.length ? routes : ['/blogs/']).forEach((route) => {
        run('npm', ['run', 'check:render', '--', `--route=${route}`]);
    });

    run('npm', ['--prefix', 'blogs', 'run', 'deploy:dist']);
    console.log('Quick deploy completed.');
}

main();
