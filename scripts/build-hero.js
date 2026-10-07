const path = require('path');
const fs = require('fs');
const { build } = require('esbuild');

const root = path.resolve(__dirname, '..');
build({
    entryPoints: [path.join(root, 'js/hero.js')],
    outfile: path.join(root, 'assets/js/hero-sculpture.js'),
    bundle: true,
    format: 'iife',
    minify: true,
    legalComments: 'linked',
    target: 'es2022'
}).then(() => {
    fs.appendFileSync(
        path.join(root, 'assets/js/hero-sculpture.js.LEGAL.txt'),
        '\nThree.js — MIT License\n\n' + fs.readFileSync(path.join(root, 'node_modules/three/LICENSE'), 'utf8')
    );
}).catch(error => {
    console.error(error);
    process.exitCode = 1;
});
