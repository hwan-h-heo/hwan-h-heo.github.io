// --- 1. 전역 변수 및 UI 요소 정의 ---
const postLangToggleButton = document.getElementById('lang-toggle-post');

// --- 2. 핵심 함수: 포스트 로딩 및 렌더링 ---

/**
 * 특정 언어의 포스트 콘텐츠를 로드하고 화면에 렌더링하는 메인 함수
 * @param {string} postId - 현재 포스트의 ID
 * @param {string} lang - 로드할 언어 ('eng' 또는 'kor')
 */
function getPostTitle(post, lang) {
    return post[`title_${lang}`] || post.title_eng;
}

function renderSeries(postId, lang) {
    const seriesContainer = document.getElementById('series');
    if (!postsData || !seriesInfo) {
        seriesContainer.style.display = 'none';
        return;
    }

    const currentPost = postsData.find(p => p.id === postId);
    if (!currentPost || !currentPost.series) {
        seriesContainer.style.display = 'none';
        return;
    }

    const seriesId = currentPost.series;
    const postsInSeries = postsData
        .filter(p => p.series === seriesId)
        .sort((a, b) => new Date(a.date) - new Date(b.date)); 

    if (postsInSeries.length <= 1) {
        seriesContainer.style.display = 'none';
        return;
    }

    const seriesTitle = seriesInfo[seriesId]?.[lang] || seriesInfo[seriesId]?.['eng'] || 'Series';

    const listItems = postsInSeries.map(post => {
        const title = getPostTitle(post, lang);
        if (post.id === postId) {
            return `<li><strong>${title}</strong></li>`;
        } else {
            return `<li><a href="?id=${post.id}">${title}</a></li>`;
        }
    }).join('');

    const seriesHtml = `
        <div class="accordion" id="accordionExample">
            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseOne" aria-expanded="false" aria-controls="collapseOne">
                        <strong>${seriesTitle}</strong>
                    </button>
                </h2>
                <div id="collapseOne" class="accordion-collapse collapse" data-bs-parent="#accordionExample">
                    <div class="accordion-body">
                        <ol>${listItems}</ol>
                    </div>
                </div>
            </div>
        </div>
    `;

    seriesContainer.innerHTML = seriesHtml;
    seriesContainer.style.display = 'block';
}

function setupPostNavigation(postId, lang) {
    const navContainer = document.getElementById('post-navigation');
    navContainer.innerHTML = ''; // 기존 내비게이션 초기화

    const currentPost = postsData.find(p => p.id === postId);
    if (!currentPost || !currentPost.series) {
        return; // 시리즈가 없는 경우 내비게이션 생성 안 함
    }

    const postsInSeries = postsData
        .filter(p => p.series === currentPost.series)
        .sort((a, b) => new Date(b.date) - new Date(a.date)); // 최신순 정렬

    const currentIndex = postsInSeries.findIndex(p => p.id === postId);
    const olderPost = postsInSeries[currentIndex + 1];
    const nextPost = postsInSeries[currentIndex - 1];

    const navDiv = document.createElement('div');
    navDiv.className = 'd-flex justify-content-between mb-4';

    if (olderPost) {
        const title = getPostTitle(olderPost, lang);
        const truncatedTitle = title.length > 25 ? title.substring(0, 25) + '...' : title;
        const olderLink = document.createElement('a');
        olderLink.className = 'btn btn-light text-uppercase';
        olderLink.href = `?id=${olderPost.id}`;
        olderLink.innerHTML = `← Older Post<br><small style="font-size: 0.7rem; text-transform: none;">${truncatedTitle}</small>`;

        olderLink.style.width = '40%';
        olderLink.style.textAlign = 'center';

        navDiv.appendChild(olderLink);
    } else {
        if (nextPost) {
            navDiv.appendChild(document.createElement('div'));
       }
    }

    if (nextPost) {
        const title = getPostTitle(nextPost, lang);
        const truncatedTitle = title.length > 25 ? title.substring(0, 25) + '...' : title;
        const nextLink = document.createElement('a');
        nextLink.className = 'btn btn-light text-uppercase';
        nextLink.href = `?id=${nextPost.id}`;
        nextLink.innerHTML = `Next Post →<br><small style="font-size: 0.7rem; text-transform: none;">${truncatedTitle}</small>`;

        nextLink.style.width = '40%';
        nextLink.style.textAlign = 'center';
        navDiv.appendChild(nextLink);
    } else {
        const allSeriesIds = Object.keys(seriesInfo);
        const otherSeriesIds = allSeriesIds.filter(id => id !== currentPost.series);

        if (otherSeriesIds.length > 0) {
            const randomSeriesId = otherSeriesIds[Math.floor(Math.random() * otherSeriesIds.length)];
            const latestPostInRandomSeries = postsData
                .filter(p => p.series === randomSeriesId)
                .sort((a, b) => new Date(a.date) - new Date(b.date))[0];

            if (latestPostInRandomSeries) {
                const seriesTitle = seriesInfo[randomSeriesId][lang] || seriesInfo[randomSeriesId]['eng'];
                const recLink = document.createElement('a');
                recLink.className = 'btn btn-outline-secondary text-uppercase'; 
                recLink.href = `?id=${latestPostInRandomSeries.id}`;
                recLink.innerHTML = `Explore Series<br><small style="font-size: 0.7rem; text-transform: none;">${seriesTitle}</small>`;
                recLink.style.width = '40%';
                navDiv.appendChild(recLink);
            }
        } else {
            navDiv.appendChild(document.createElement('div')); // 공간 차지
        }
    }

    navContainer.appendChild(navDiv);
}

function loadAndRenderPost(postId, lang) {
    if (!postId) {
        console.error("There is no Post ID.");
        return;
    }

    const postContentEl = document.getElementById('post-content');
    updatePostToggleUI(lang);

    const primaryUrl = `${postId}/content-${lang}.md`;
    const fallbackUrl = `${postId}/content-eng.md`;

    fetch(primaryUrl)
        .then(response => {
            if (!response.ok && lang !== 'eng') {
                console.warn(`'${primaryUrl}' Cannot find md file, changed to english`);
                return fetch(fallbackUrl);
            }
            return response; 
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`'${fallbackUrl}' also cannot find.`);
            }
            return response.text();
        })
        .then(text => {
            const parts = text.split('--- 여기부터 실제 콘텐츠 ---');
            const metadata = parts[0];
            const content = parts.length > 1 ? parts[1].trim() : '';
            
            const lines = metadata.split('\n');
            const title = lines[0].replace('title: ', '').trim();
            const date = lines[1].replace('date: ', '').trim();
            const author = lines[2].replace('author: ', '').trim();

            // 페이지 제목 및 메타 정보 업데이트
            document.getElementById('post-title').innerText = title;
            document.title = title;
            document.getElementById('post-meta').innerText = `Posted by ${author} on ${date}`;
            
            let finalHtmlContent;

            if (content.includes('<nav class="toc">')) {
                console.log("exsiting toc.");
                finalHtmlContent = marked.parse(content);
            } else {
                console.log("toc gen.");
                const { tocHtml, contentHtml } = generateTOC(marked.parse(content));
                finalHtmlContent = contentHtml; // 기본값으로 본문 HTML을 설정
                if (tocHtml) {
                    finalHtmlContent = `<nav class="toc">${tocHtml}</nav>` + contentHtml;
                }
            }

            postContentEl.innerHTML = finalHtmlContent;

            const postId = primaryUrl.split('/')[0]
            console.log(postId)
            setupPostNavigation(postId, lang);

            Prism.highlightAllUnder(postContentEl); 
            renderMathInElement(postContentEl, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\(', right: '\\)', display: false},
                        {left: '\\[', right: '\\]', display: true}
                    ],
                    throwOnError : false
                }); 
            if (typeof initializeToc === 'function') initializeToc(); 
            if (typeof initializeShareFunctionality === 'function') initializeShareFunctionality();
            
            // custom script code for three-js viewer
            
            if (postId === '240917_3djs') { 
                console.log('aaaa')
                import('../3DViewer/js/gaussian_viewer.js')
                    .then(module => {
                        module.initGaussianViewer();
                    })
                    .catch(err => {
                        console.error("Failed to load the Gaussian viewer script:", err);
                    });
            }

            
        })
        .catch(error => {
            console.error('Error Loading:', error);
            postContentEl.innerHTML = '<p style="text-align: center;">Faild to Load the Content.</p>';
        });
}


function updatePostToggleUI(lang) {
    if (postLangToggleButton) {
        postLangToggleButton.textContent = (lang === 'eng') ? 'kor' : 'eng';
    }
}

if (postLangToggleButton) {
    postLangToggleButton.addEventListener('click', function() {
        const urlParams = new URLSearchParams(window.location.search);
        const postId = urlParams.get('id');

        const currentLang = localStorage.getItem('language') || 'eng';
        const newLang = (currentLang === 'eng') ? 'kor' : 'eng';
        
        localStorage.setItem('language', newLang);

        renderSeries(postId, newLang); 
        loadAndRenderPost(postId, newLang);
    });
}

document.addEventListener("DOMContentLoaded", function() {
    const urlParams = new URLSearchParams(window.location.search);
    const postId = urlParams.get('id');
    const lang = localStorage.getItem('language') || 'eng';

    if (postId) {
        renderSeries(postId, lang);
        loadAndRenderPost(postId, lang);
    }
});

function generateTOC(htmlContent) {
    const tempElement = document.createElement('div');
    tempElement.innerHTML = htmlContent;
    const headings = tempElement.querySelectorAll('h2, h3');
    if (headings.length === 0) return { tocHtml: '', contentHtml: htmlContent };
    let tocHTML = '<ul>';
    let currentH2 = null;
    headings.forEach((heading, index) => {
        let headingId = heading.id;
        if (!headingId) {
            headingId = `toc-heading-${index}`;
            heading.id = headingId;
        }
        const tagName = heading.tagName;
        const headingText = heading.innerHTML;
        if (tagName === 'H2') {
            if (currentH2) tocHTML += '</ul></li>';
            tocHTML += `<li><a href="#${headingId}">${headingText}</a><ul>`;
            currentH2 = heading;
        } else if (tagName === 'H3' && currentH2) {
            tocHTML += `<li><a href="#${headingId}">${headingText}</a></li>`;
        }
    });
    if (currentH2) tocHTML += '</ul></li>';
    tocHTML += '</ul>';
    return { tocHtml: tocHTML, contentHtml: tempElement.innerHTML };
}

function setupNavigationAndHighlighting() {
    const urlParams = new URLSearchParams(window.location.search);
    const postId = urlParams.get('id');
    const seriesLinks = document.querySelectorAll('#series ol li a');
    if (seriesLinks.length === 0) return;
    const postList = Array.from(seriesLinks).map(anchor => {
        const href = anchor.getAttribute('href');
        const linkUrlParams = new URLSearchParams(href.split('?')[1]);
        let id = linkUrlParams.get('id');
        if (id) {
            id = id.replace(/\/$/, ''); // 정규식을 사용하여 마지막 슬래시만 제거
        }
        return { id, href, anchorElement: anchor };
    }).filter(post => post.id);
    const currentIndex = postList.findIndex(post => post.id === postId);
    if (currentIndex === -1) return;
    const currentPostAnchor = postList[currentIndex].anchorElement;
    const listItem = currentPostAnchor.parentElement;
    listItem.innerHTML = `<strong>${currentPostAnchor.textContent}</strong>`;
    const olderPost = postList[currentIndex - 1];
    const nextPost = postList[currentIndex + 1];
    const navContainer = document.getElementById('post-navigation');
    const navDiv = document.createElement('div');
    navDiv.className = 'd-flex justify-content-between mb-4';
    if (olderPost) {
        const olderLink = document.createElement('a');
        olderLink.className = 'btn btn-primary text-uppercase';
        olderLink.href = olderPost.href;
        olderLink.innerHTML = '← Older Post';
        navDiv.appendChild(olderLink);
    } else {
        navDiv.appendChild(document.createElement('div'));
    }
    if (nextPost) {
        const nextLink = document.createElement('a');
        nextLink.className = 'btn btn-primary text-uppercase';
        nextLink.href = nextPost.href;
        nextLink.innerHTML = 'Next Post →';
        navDiv.appendChild(nextLink);
    }
    navContainer.appendChild(navDiv);
}
