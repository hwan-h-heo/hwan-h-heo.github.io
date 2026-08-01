const fs = require('fs');
const { createRequire } = require('module');
const path = require('path');

const cheerio = require('cheerio');
const requireBlogDependency = createRequire(path.join(__dirname, '..', 'blogs', 'package.json'));
const { marked } = requireBlogDependency('marked');
const { parseMarkdownWithMath } = require('../blogs/js/markdown-with-math');
const { parsePostMarkdownSource } = require('../blogs/lib/post-runtime-dependencies');
const { listContentFiles, protectBlocks } = require('./migrate-legacy-markdown');

const repoRoot = path.join(__dirname, '..');
const plainParagraphPattern = /<p\s*>([\s\S]*?)<\/p>/gi;
const htmlTagPattern = /<\/?([a-z][a-z0-9:-]*)\b[^>]*>/gi;
const voidTags = new Set([
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
    'link', 'meta', 'param', 'source', 'track', 'wbr'
]);

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

function updateTagStack(stack, tagMatch) {
    const rawTag = tagMatch[0];
    const tagName = tagMatch[1].toLowerCase();
    const isClosing = /^<\//.test(rawTag);
    const isSelfClosing = /\/\s*>$/.test(rawTag) || voidTags.has(tagName);

    if (isClosing) {
        const matchingIndex = stack.lastIndexOf(tagName);
        if (matchingIndex !== -1) {
            stack.splice(matchingIndex);
        }
        return;
    }

    if (!isSelfClosing) {
        stack.push(tagName);
    }
}

function findTopLevelParagraphs(source) {
    const paragraphMatches = Array.from(source.matchAll(plainParagraphPattern));
    const tagMatches = Array.from(source.matchAll(htmlTagPattern));
    const stack = [];
    const candidates = [];
    let tagIndex = 0;

    paragraphMatches.forEach((paragraphMatch) => {
        while (tagIndex < tagMatches.length && tagMatches[tagIndex].index < paragraphMatch.index) {
            updateTagStack(stack, tagMatches[tagIndex]);
            tagIndex += 1;
        }

        const lineStart = source.lastIndexOf('\n', paragraphMatch.index - 1) + 1;
        const linePrefix = source.slice(lineStart, paragraphMatch.index);
        const paragraphEnd = paragraphMatch.index + paragraphMatch[0].length;
        const nextLineBreak = source.indexOf('\n', paragraphEnd);
        const lineEnd = nextLineBreak === -1 ? source.length : nextLineBreak;
        const lineSuffix = source.slice(paragraphEnd, lineEnd);
        if (stack.length === 0 && linePrefix === '' && !lineSuffix.trim()) {
            candidates.push({
                fullMatch: paragraphMatch[0],
                index: paragraphMatch.index,
                innerHtml: paragraphMatch[1]
            });
        }
    });

    return candidates;
}

function canonicalNode(node) {
    if (node.type === 'text') {
        const text = String(node.data || '').replace(/\s+/g, ' ').trim();
        return text ? `#${text}` : '';
    }
    if (node.type === 'comment') {
        return '';
    }
    if (!node.name) {
        return (node.children || []).map(canonicalNode).join('');
    }

    const attributes = Object.entries(node.attribs || {})
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([name, value]) => ` ${name}=${JSON.stringify(value)}`)
        .join('');
    const children = (node.children || []).map(canonicalNode).join('');
    return `<${node.name}${attributes}>${children}</${node.name}>`;
}

function canonicalHtml(html) {
    const $ = cheerio.load(html, null, false);
    return $.root().contents().toArray().map(canonicalNode).join('');
}

function renderMarkdown(markdown) {
    return parseMarkdownWithMath(markdown, (source) => marked.parse(source));
}

function rendersIdentically(originalHtml, markdown) {
    return canonicalHtml(renderMarkdown(originalHtml)) === canonicalHtml(renderMarkdown(markdown));
}

function replaceLeafTag(source, pattern, createMarkdown) {
    let count = 0;
    const content = source.replace(pattern, (fullMatch, ...groups) => {
        const replacement = createMarkdown(...groups);
        if (!replacement) {
            return fullMatch;
        }
        count += 1;
        return replacement;
    });
    return { content, count };
}

function convertSafeInlineHtml(source) {
    let content = source;
    const counts = {
        code: 0,
        emphasis: 0,
        links: 0,
        strong: 0
    };

    let result = replaceLeafTag(
        content,
        /<strong\s*>\s*<em\s*>([^<>\r\n*_]+)<\/em>\s*<\/strong>/gi,
        (text) => (text.trim() ? `**_${text.trim()}_**` : '')
    );
    content = result.content;
    counts.strong += result.count;
    counts.emphasis += result.count;

    result = replaceLeafTag(
        content,
        /<em\s*>\s*<strong\s*>([^<>\r\n*_]+)<\/strong>\s*<\/em>/gi,
        (text) => (text.trim() ? `_**${text.trim()}**_` : '')
    );
    content = result.content;
    counts.strong += result.count;
    counts.emphasis += result.count;

    result = replaceLeafTag(content, /<code\s*>([^<>\r\n`]*)<\/code>/gi, (text) => (
        text ? `\`${text}\`` : ''
    ));
    content = result.content;
    counts.code += result.count;

    result = replaceLeafTag(content, /<strong\s*>([^<>\r\n*_]+)<\/strong>/gi, (text) => (
        text.trim() ? `**${text}**` : ''
    ));
    content = result.content;
    counts.strong += result.count;

    result = replaceLeafTag(content, /<em\s*>([^<>\r\n*_]+)<\/em>/gi, (text) => (
        text.trim() ? `*${text}*` : ''
    ));
    content = result.content;
    counts.emphasis += result.count;

    result = replaceLeafTag(
        content,
        /<a\s+href=(['"])([^'"\s()]+)\1\s*>([^<>\r\n\[\]\\]+)<\/a>/gi,
        (quote, href, label) => (label.trim() ? `[${label}](${href})` : '')
    );
    content = result.content;
    counts.links += result.count;

    return { content, counts };
}

function sumInlineCounts(left, right) {
    return {
        code: left.code + right.code,
        emphasis: left.emphasis + right.emphasis,
        links: left.links + right.links,
        strong: left.strong + right.strong
    };
}

function convertParagraph(paragraph) {
    const originalHtml = paragraph.fullMatch;
    const plainMarkdown = paragraph.innerHtml.trim().replace(/[ \t]+$/gm, '');
    if (!plainMarkdown || /\uE000PROTECTED_BLOCK_\d+\uE001/.test(plainMarkdown)) {
        return null;
    }

    const inlineResult = convertSafeInlineHtml(plainMarkdown);
    if (rendersIdentically(originalHtml, inlineResult.content)) {
        return {
            inlineCounts: inlineResult.counts,
            markdown: inlineResult.content
        };
    }

    if (rendersIdentically(originalHtml, plainMarkdown)) {
        return {
            inlineCounts: { code: 0, emphasis: 0, links: 0, strong: 0 },
            markdown: plainMarkdown
        };
    }

    return null;
}

function replaceParagraphBlock(source, paragraph, markdown) {
    const paragraphEnd = paragraph.index + paragraph.fullMatch.length;
    const before = source.slice(0, paragraph.index)
        .replace(/[ \t]*(?:\r?\n[ \t]*)*$/, '');
    const after = source.slice(paragraphEnd)
        .replace(/^[ \t]*(?:\r?\n[ \t]*)*/, '');
    const separator = '\n\n';
    return `${before}${before ? separator : ''}${markdown.trim()}${after ? separator : ''}${after}`;
}

function convertSafeMarkdownBlocks(source) {
    const protectedSource = protectBlocks(source);
    const candidates = findTopLevelParagraphs(protectedSource.text);
    const originalContent = parsePostMarkdownSource(source).content;
    const originalRender = canonicalHtml(renderMarkdown(originalContent));
    const totals = {
        candidates: candidates.length,
        code: 0,
        contextRejected: 0,
        emphasis: 0,
        links: 0,
        paragraphs: 0,
        strong: 0
    };
    let content = protectedSource.text;

    candidates.slice().reverse().forEach((paragraph) => {
        const conversion = convertParagraph(paragraph);
        if (!conversion) {
            return;
        }

        const proposedContent = replaceParagraphBlock(content, paragraph, conversion.markdown);
        const restoredProposal = protectedSource.restore(proposedContent);
        const proposalBody = parsePostMarkdownSource(restoredProposal).content;
        if (canonicalHtml(renderMarkdown(proposalBody)) !== originalRender) {
            totals.contextRejected += 1;
            return;
        }

        content = proposedContent;
        totals.paragraphs += 1;
        Object.assign(totals, sumInlineCounts(totals, conversion.inlineCounts));
    });

    content = protectedSource.restore(content);
    const convertedContent = parsePostMarkdownSource(content).content;
    if (!rendersIdentically(originalContent, convertedContent)) {
        throw new Error('Full-document render changed after safe paragraph conversion.');
    }

    return {
        content,
        ...totals
    };
}

function main() {
    const options = parseArguments(process.argv.slice(2));
    const results = [];

    listContentFiles(options.postId).forEach((filePath) => {
        const source = fs.readFileSync(filePath, 'utf8');
        const result = convertSafeMarkdownBlocks(source);
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
            + `${result.paragraphs}/${result.candidates} paragraphs, `
            + `${result.contextRejected} context rejects, `
            + `${result.strong} strong, ${result.emphasis} emphasis, `
            + `${result.code} code, ${result.links} links`
        );
    });

    const totals = results.reduce((summary, result) => ({
        candidates: summary.candidates + result.candidates,
        code: summary.code + result.code,
        contextRejected: summary.contextRejected + result.contextRejected,
        emphasis: summary.emphasis + result.emphasis,
        files: summary.files + 1,
        links: summary.links + result.links,
        paragraphs: summary.paragraphs + result.paragraphs,
        strong: summary.strong + result.strong
    }), { candidates: 0, code: 0, contextRejected: 0, emphasis: 0, files: 0, links: 0, paragraphs: 0, strong: 0 });

    console.log(
        `${options.write ? 'Migration complete' : 'Dry run complete'}: ${totals.files} files, `
        + `${totals.paragraphs}/${totals.candidates} paragraphs, ${totals.contextRejected} context rejects, `
        + `${totals.strong} strong, `
        + `${totals.emphasis} emphasis, ${totals.code} code, ${totals.links} links.`
    );
}

if (require.main === module) {
    main();
}

module.exports = {
    canonicalHtml,
    convertSafeInlineHtml,
    convertSafeMarkdownBlocks,
    findTopLevelParagraphs,
    rendersIdentically
};
