const fs = require('fs');
const path = require('path');

const { createSlug } = require('../blogs/lib/site-data');

const repoRoot = path.join(__dirname, '..');
const projectsDir = path.join(repoRoot, 'projects');

function main() {
    const title = process.argv.slice(2).join(' ').trim();
    if (!title) {
        throw new Error('Usage: npm run new:project -- "Project Name"');
    }

    const slug = createSlug(title).replace(/-/g, '_') || 'new_project';
    let candidate = slug;
    let suffix = 2;
    while (fs.existsSync(path.join(projectsDir, candidate))) {
        candidate = `${slug}_${suffix}`;
        suffix += 1;
    }

    const projectDir = path.join(projectsDir, candidate);
    fs.mkdirSync(path.join(projectDir, 'assets'), { recursive: true });

    const metadata = {
        title,
        heroTitle: title,
        subtitles: ['Project'],
        description: '',
        keywords: ''
    };

    const content = `:::{.container .portfolio-details-container .col-11}\n:::{.row .gy-4}\n:::{.col-lg-8}\n:::{.portfolio-description}\n## Project Overview\n\nStart writing the project page here.\n:::\n:::\n\n:::{.col-lg-4}\n:::{.portfolio-info}\n### Project Details\n\n- **Category**: Project\n- **Skills Demonstrated**: Add skills here\n:::\n:::\n:::\n:::\n`;

    fs.writeFileSync(path.join(projectDir, 'project.json'), `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');
    fs.writeFileSync(path.join(projectDir, 'content.md'), content, 'utf8');

    console.log(`Created project source at projects/${candidate}/`);
    console.log('Add a portfolio card in blogs/data/site-data.json when a card image is ready.');
}

main();
