const fs = require('fs');
const path = require('path');

const cheerio = require('cheerio');

const repoRoot = path.join(__dirname, '..');
const postsRoot = path.join(repoRoot, 'blogs', 'posts');
const protectedBlockPattern = /```[^\n]*\n[\s\S]*?\n```|~~~[^\n]*\n[\s\S]*?\n~~~|<(pre|script|style)\b[^>]*>[\s\S]*?<\/\1>/gi;
const manualTocPattern = /<nav\b[^>]*class=(['"])[^'"]*\btoc\b[^'"]*\1[^>]*>[\s\S]*?<\/nav>/gi;
const rawHeadingPattern = /<h([234])\b([^>]*)>([\s\S]*?)<\/h\1>/gi;

function parseArguments(argv) {
    const options = {
        postId: '',
        write: false
    };

    argv.forEach((argument) => {
        if (argument === '--write') {
            options.write = true;
        } else if (argument.startsWith('--post=')) {
            options.postId = argument.slice('--post='.length).trim();
        }
    });

    return options;
}

function listContentFiles(postId = '') {
    const postDirectories = fs.readdirSync(postsRoot, { withFileTypes: true })
        .filter((entry) => entry.isDirectory() && (!postId || entry.name === postId));

    if (postId && postDirectories.length === 0) {
        throw new Error(`Unknown post directory: ${postId}`);
    }

    return postDirectories.flatMap((entry) => {
        const directory = path.join(postsRoot, entry.name);
        return fs.readdirSync(directory)
            .filter((fileName) => /^content-(?:eng|kor)\.md$/.test(fileName))
            .map((fileName) => path.join(directory, fileName));
    });
}

function protectBlocks(source) {
    const blocks = [];
    const text = source.replace(protectedBlockPattern, (block) => {
        const token = `\uE000PROTECTED_BLOCK_${blocks.length}\uE001`;
        blocks.push(block);
        return token;
    });

    return {
        restore(value) {
            return value.replace(/\uE000PROTECTED_BLOCK_(\d+)\uE001/g, (match, index) => blocks[Number(index)]);
        },
        text
    };
}

function normalizeLegacyTocLists($) {
    $('ul > ul, ul > ol, ol > ul, ol > ol').each((index, list) => {
        const parentItem = $(list).prev('li');
        if (parentItem.length) {
            parentItem.append(list);
        }
    });
}

function getManualTocHeadingLevels(tocHtml) {
    const $ = cheerio.load(tocHtml, null, false);
    const levels = new Map();

    normalizeLegacyTocLists($);
    $('li').each((index, item) => {
        const anchor = $(item).children('a[href^="#"]').first();
        if (!anchor.length) {
            return;
        }

        const rawId = anchor.attr('href').slice(1);
        let id = rawId;
        try {
            id = decodeURIComponent(rawId);
        } catch (error) {
            id = rawId;
        }

        const nestingDepth = $(item).parents('li').length;
        levels.set(id, Math.min(4, 2 + nestingDepth));
    });

    return levels;
}

function getAttribute(attributes, name) {
    const match = attributes.match(new RegExp(`\\b${name}\\s*=\\s*(['"])(.*?)\\1`, 'i'));
    return match ? match[2] : '';
}

function normalizeHeadingText(value) {
    return String(value || '')
        .replace(/\s+/g, ' ')
        .trim();
}

function normalizeMarkdownHeadingSpacing(source) {
    const lines = source.split(/\r?\n/);
    const output = [];

    lines.forEach((line, index) => {
        if (!/^#{2,4}[ \t]+\S/.test(line)) {
            output.push(line);
            return;
        }

        if (output.length > 0 && output[output.length - 1].trim()) {
            output.push('');
        }
        output.push(line.trimEnd());
        if (index < lines.length - 1 && lines[index + 1].trim()) {
            output.push('');
        }
    });

    return output.join('\n');
}

function convertLegacyMarkdown(source) {
    const protectedSource = protectBlocks(source);
    const tocBlocks = [];
    let text = protectedSource.text.replace(manualTocPattern, (tocHtml) => {
        tocBlocks.push(tocHtml);
        return '';
    });
    const tocHeadingLevels = new Map();
    tocBlocks.forEach((tocHtml) => {
        getManualTocHeadingLevels(tocHtml).forEach((level, id) => {
            tocHeadingLevels.set(id, level);
        });
    });

    let convertedHeadings = 0;
    let remappedHeadings = 0;
    text = text.replace(rawHeadingPattern, (fullMatch, rawLevel, attributes, headingHtml) => {
        const originalLevel = Number(rawLevel);
        const id = getAttribute(attributes, 'id');
        const mappedLevel = id && tocHeadingLevels.get(id);
        const level = mappedLevel || originalLevel;
        const headingText = normalizeHeadingText(headingHtml);

        if (!headingText) {
            return fullMatch;
        }

        convertedHeadings += 1;
        if (level !== originalLevel) {
            remappedHeadings += 1;
        }

        const preservedAttributes = attributes.trim();
        const anchor = preservedAttributes ? `<span ${preservedAttributes}></span>` : '';
        return `${'#'.repeat(level)} ${anchor}${headingText}`;
    });

    text = text.replace(
        /(--- 여기부터 실제 콘텐츠 ---)[ \t]*\r?\n(?:[ \t]*\r?\n){2,}/,
        '$1\n\n'
    );
    if (convertedHeadings > 0 || /^#{2,4}[ \t]+<span\b[^>]*>/m.test(text)) {
        text = normalizeMarkdownHeadingSpacing(text);
    }

    return {
        content: protectedSource.restore(text),
        convertedHeadings,
        remappedHeadings,
        removedTocs: tocBlocks.length
    };
}

function main() {
    const options = parseArguments(process.argv.slice(2));
    const results = [];

    listContentFiles(options.postId).forEach((filePath) => {
        const source = fs.readFileSync(filePath, 'utf8');
        const result = convertLegacyMarkdown(source);
        if (result.content === source) {
            return;
        }

        if (options.write) {
            fs.writeFileSync(filePath, result.content);
        }
        results.push({
            ...result,
            filePath: path.relative(repoRoot, filePath)
        });
    });

    results.forEach((result) => {
        console.log(
            `${options.write ? 'UPDATED' : 'WOULD UPDATE'} ${result.filePath}: `
            + `${result.convertedHeadings} headings, ${result.remappedHeadings} levels remapped, `
            + `${result.removedTocs} manual TOCs removed`
        );
    });

    const totals = results.reduce((summary, result) => ({
        files: summary.files + 1,
        headings: summary.headings + result.convertedHeadings,
        remapped: summary.remapped + result.remappedHeadings,
        tocs: summary.tocs + result.removedTocs
    }), { files: 0, headings: 0, remapped: 0, tocs: 0 });

    console.log(
        `${options.write ? 'Migration complete' : 'Dry run complete'}: ${totals.files} files, `
        + `${totals.headings} headings, ${totals.remapped} level remaps, ${totals.tocs} manual TOCs.`
    );
}

if (require.main === module) {
    main();
}

module.exports = {
    convertLegacyMarkdown,
    getManualTocHeadingLevels,
    listContentFiles,
    protectBlocks,
    normalizeMarkdownHeadingSpacing
};
