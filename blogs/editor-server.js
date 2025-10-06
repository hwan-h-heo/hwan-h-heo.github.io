const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = 3030;
const POSTS_DIR = path.join(__dirname, 'posts');
const DRAFTS_DIR = path.join(__dirname, 'editor', 'drafts');
const DRAFT_ASSETS_DIR = path.join(__dirname, 'editor', 'draft-assets');

// Ensure directories exist
if (!fs.existsSync(DRAFTS_DIR)) {
    fs.mkdirSync(DRAFTS_DIR, { recursive: true });
}

if (!fs.existsSync(DRAFT_ASSETS_DIR)) {
    fs.mkdirSync(DRAFT_ASSETS_DIR, { recursive: true });
}

// MIME types
const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.md': 'text/markdown'
};

function serveStaticFile(filePath, res) {
    fs.readFile(filePath, (err, data) => {
        if (err) {
            res.writeHead(404);
            res.end('File not found');
            return;
        }
        const ext = path.extname(filePath);
        const contentType = mimeTypes[ext] || 'application/octet-stream';
        res.writeHead(200, { 'Content-Type': contentType });
        res.end(data);
    });
}

// Parse multipart form data (for image uploads)
function parseMultipart(req, callback) {
    const boundary = req.headers['content-type'].split('boundary=')[1];
    if (!boundary) {
        callback(new Error('No boundary found'));
        return;
    }

    let data = Buffer.alloc(0);
    req.on('data', chunk => {
        data = Buffer.concat([data, chunk]);
    });

    req.on('end', () => {
        const parts = [];
        const boundaryBuffer = Buffer.from(`--${boundary}`);
        let start = 0;

        while (true) {
            const boundaryIndex = data.indexOf(boundaryBuffer, start);
            if (boundaryIndex === -1) break;

            const nextBoundaryIndex = data.indexOf(boundaryBuffer, boundaryIndex + boundaryBuffer.length);
            if (nextBoundaryIndex === -1) break;

            const partData = data.slice(boundaryIndex + boundaryBuffer.length, nextBoundaryIndex);

            // Parse headers and content
            const headerEnd = partData.indexOf(Buffer.from('\r\n\r\n'));
            if (headerEnd !== -1) {
                const headers = partData.slice(0, headerEnd).toString();
                const content = partData.slice(headerEnd + 4, partData.length - 2); // Remove trailing \r\n

                const nameMatch = headers.match(/name="([^"]+)"/);
                const filenameMatch = headers.match(/filename="([^"]+)"/);

                if (nameMatch) {
                    parts.push({
                        name: nameMatch[1],
                        filename: filenameMatch ? filenameMatch[1] : null,
                        data: content
                    });
                }
            }

            start = nextBoundaryIndex;
        }

        callback(null, parts);
    });
}

const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const pathname = parsedUrl.pathname;

    // Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }

    // API Routes
    if (pathname.startsWith('/api/')) {
        // List all drafts
        if (pathname === '/api/drafts' && req.method === 'GET') {
            fs.readdir(DRAFTS_DIR, (err, files) => {
                if (err) {
                    res.writeHead(500, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Failed to read drafts' }));
                    return;
                }
                const mdFiles = files.filter(f => f.endsWith('.md'));
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ drafts: mdFiles }));
            });
            return;
        }

        // Get draft content
        if (pathname.startsWith('/api/draft/') && req.method === 'GET') {
            const draftName = decodeURIComponent(pathname.substring('/api/draft/'.length));
            const draftPath = path.join(DRAFTS_DIR, draftName);

            fs.readFile(draftPath, 'utf8', (err, data) => {
                if (err) {
                    res.writeHead(404, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Draft not found' }));
                    return;
                }
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ content: data }));
            });
            return;
        }

        // Save draft
        if (pathname === '/api/draft' && req.method === 'POST') {
            let body = '';
            req.on('data', chunk => {
                body += chunk.toString();
            });
            req.on('end', () => {
                try {
                    const { filename, content } = JSON.parse(body);
                    if (!filename || !content) {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ error: 'Filename and content required' }));
                        return;
                    }

                    const safeName = filename.endsWith('.md') ? filename : `${filename}.md`;
                    const draftPath = path.join(DRAFTS_DIR, safeName);

                    fs.writeFile(draftPath, content, 'utf8', (err) => {
                        if (err) {
                            res.writeHead(500, { 'Content-Type': 'application/json' });
                            res.end(JSON.stringify({ error: 'Failed to save draft' }));
                            return;
                        }
                        res.writeHead(200, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ success: true, filename: safeName }));
                    });
                } catch (err) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Invalid JSON' }));
                }
            });
            return;
        }

        // Save to posts directory (with asset migration)
        if (pathname === '/api/post' && req.method === 'POST') {
            let body = '';
            req.on('data', chunk => {
                body += chunk.toString();
            });
            req.on('end', () => {
                try {
                    const { postId, language, content } = JSON.parse(body);
                    if (!postId || !language || !content) {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ error: 'postId, language, and content required' }));
                        return;
                    }

                    const postDir = path.join(POSTS_DIR, postId);
                    const postAssetsDir = path.join(postDir, 'assets');

                    // Create directories
                    if (!fs.existsSync(postDir)) {
                        fs.mkdirSync(postDir, { recursive: true });
                    }
                    if (!fs.existsSync(postAssetsDir)) {
                        fs.mkdirSync(postAssetsDir, { recursive: true });
                    }

                    // Find all draft-assets references in content
                    const assetRegex = /\.\/draft-assets\/([^\s\)]+)/g;
                    let updatedContent = content;
                    const matches = [...content.matchAll(assetRegex)];

                    // Migrate assets
                    const migrationPromises = matches.map(match => {
                        return new Promise((resolve, reject) => {
                            const filename = match[1];
                            const sourcePath = path.join(DRAFT_ASSETS_DIR, filename);
                            const destPath = path.join(postAssetsDir, filename);

                            // Copy file
                            fs.copyFile(sourcePath, destPath, (err) => {
                                if (err) {
                                    console.warn(`Failed to copy ${filename}:`, err);
                                    resolve(); // Don't fail the whole operation
                                } else {
                                    // Update path in content
                                    updatedContent = updatedContent.replace(
                                        `./draft-assets/${filename}`,
                                        `./assets/${filename}`
                                    );
                                    resolve();
                                }
                            });
                        });
                    });

                    // Wait for all migrations to complete
                    Promise.all(migrationPromises).then(() => {
                        const filename = `content-${language}.md`;
                        const filePath = path.join(postDir, filename);

                        fs.writeFile(filePath, updatedContent, 'utf8', (err) => {
                            if (err) {
                                res.writeHead(500, { 'Content-Type': 'application/json' });
                                res.end(JSON.stringify({ error: 'Failed to save post' }));
                                return;
                            }
                            res.writeHead(200, { 'Content-Type': 'application/json' });
                            res.end(JSON.stringify({
                                success: true,
                                path: filePath,
                                assetsMigrated: matches.length
                            }));
                        });
                    });
                } catch (err) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Invalid JSON' }));
                }
            });
            return;
        }

        // Get post content
        if (pathname.startsWith('/api/post/') && req.method === 'GET') {
            const parts = pathname.substring('/api/post/'.length).split('/');
            if (parts.length !== 2) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Invalid path format. Use /api/post/{postId}/{language}' }));
                return;
            }

            const [postId, language] = parts.map(decodeURIComponent);
            const filename = `content-${language}.md`;
            const filePath = path.join(POSTS_DIR, postId, filename);

            fs.readFile(filePath, 'utf8', (err, data) => {
                if (err) {
                    res.writeHead(404, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Post not found' }));
                    return;
                }
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ content: data }));
            });
            return;
        }

        // Delete draft
        if (pathname.startsWith('/api/draft/') && req.method === 'DELETE') {
            const draftName = decodeURIComponent(pathname.substring('/api/draft/'.length));
            const draftPath = path.join(DRAFTS_DIR, draftName);

            fs.unlink(draftPath, (err) => {
                if (err) {
                    res.writeHead(404, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Draft not found' }));
                    return;
                }
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ success: true }));
            });
            return;
        }

        // Upload image
        if (pathname === '/api/upload-image' && req.method === 'POST') {
            parseMultipart(req, (err, parts) => {
                if (err) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Invalid multipart data' }));
                    return;
                }

                const imagePart = parts.find(p => p.name === 'image');
                if (!imagePart || !imagePart.filename) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'No image file found' }));
                    return;
                }

                const filename = imagePart.filename;
                const filepath = path.join(DRAFT_ASSETS_DIR, filename);

                fs.writeFile(filepath, imagePart.data, (err) => {
                    if (err) {
                        res.writeHead(500, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ error: 'Failed to save image' }));
                        return;
                    }

                    // Return both server path (for preview) and relative path (for markdown)
                    const serverPath = `/editor/draft-assets/${filename}`;
                    const relativePath = `./draft-assets/${filename}`;

                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({
                        success: true,
                        serverPath: serverPath,      // For immediate preview
                        relativePath: relativePath,  // For markdown content
                        filename: filename
                    }));
                });
            });
            return;
        }

        // API endpoint not found
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'API endpoint not found' }));
        return;
    }

    // Serve static files
    let filePath = '.' + pathname;
    if (filePath === './') {
        filePath = './editor/index.html';
    }

    serveStaticFile(filePath, res);
});

server.listen(PORT, () => {
    console.log(`\n🚀 Editor server running at http://localhost:${PORT}/`);
    console.log(`📝 Edit mode: http://localhost:${PORT}/editor/edit.html`);
    console.log(`💾 Drafts directory: ${DRAFTS_DIR}`);
    console.log(`📂 Posts directory: ${POSTS_DIR}\n`);
});
