// --- 1. 전역 변수 및 UI 요소 정의 ---
const postLangToggleButton = document.getElementById('lang-toggle-post');

// --- 2. 핵심 함수: 포스트 로딩 및 렌더링 ---

/**
 * 특정 언어의 포스트 콘텐츠를 로드하고 화면에 렌더링하는 메인 함수
 * @param {string} postId - 현재 포스트의 ID
 * @param {string} lang - 로드할 언어 ('eng' 또는 'kor')
 */
function loadAndRenderPost(postId, lang) {
    if (!postId) {
        console.error("포스트 ID가 없습니다.");
        return;
    }

    const postContentEl = document.getElementById('post-content');
    
    // UI를 즉시 업데이트하여 사용자에게 현재 로드 중인 언어를 알려줍니다.
    updatePostToggleUI(lang);

    const primaryUrl = `${postId}/content-${lang}.md`;
    const fallbackUrl = `${postId}/content-eng.md`;

    // 1. 선택된 언어의 마크다운 파일을 먼저 요청합니다.
    fetch(primaryUrl)
        .then(response => {
            // 2. 요청이 실패했고(예: 404 Not Found) 그 언어가 기본 언어(eng)가 아니라면,
            if (!response.ok && lang !== 'eng') {
                console.warn(`'${primaryUrl}' 파일을 찾을 수 없습니다. 기본 언어(영어)로 대체합니다.`);
                // 3. 기본 언어(영어) 파일로 다시 요청을 시도합니다.
                return fetch(fallbackUrl);
            }
            return response; // 처음 요청이 성공했으면 그대로 반환합니다.
        })
        .then(response => {
            // 대체 요청까지 실패했다면 에러를 발생시킵니다.
            if (!response.ok) {
                throw new Error(`'${fallbackUrl}' 파일도 찾을 수 없습니다. 포스트가 존재하지 않을 수 있습니다.`);
            }
            return response.text();
        })
        .then(text => {
            // --- 4. 성공적으로 가져온 텍스트로 페이지를 렌더링합니다. (이 부분은 기존 로직과 유사) ---
            
            // 메타데이터와 콘텐츠 분리
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

            
            if (!document.querySelector('#post-navigation div')) {
                setupNavigationAndHighlighting();
            }

            Prism.highlightAllUnder(postContentEl); // 코드 하이라이팅
            renderMathInElement(postContentEl, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\(', right: '\\)', display: false},
                        {left: '\\[', right: '\\]', display: true}
                    ],
                    throwOnError : false
                }); // 수학 공식 렌더링
            if (typeof initializeToc === 'function') initializeToc(); // 목차 스크롤 기능
            
            // 특정 포스트를 위한 커스텀 스크립트 로드
            const postId = primaryUrl.split('/')[0]
            console.log(postId)
            if (postId === '240917_3djs') { 
                console.log('aaaa')
                import('../3DViewer/js/gaussian_viewer.js')
                    .then(module => {
                        // 모듈 로드가 성공하면, export된 함수를 실행합니다.
                        module.initGaussianViewer();
                    })
                    .catch(err => {
                        console.error("Failed to load the Gaussian viewer script:", err);
                    });
            }

            if (typeof initializeShareFunctionality === 'function') initializeShareFunctionality();
        })
        .catch(error => {
            console.error('포스트 콘텐츠 로딩 중 에러 발생:', error);
            postContentEl.innerHTML = '<p style="text-align: center;">포스트를 불러오는 데 실패했습니다.<br>페이지가 존재하지 않거나, 준비 중일 수 있습니다.</p>';
        });
}


// --- 3. 헬퍼 및 이벤트 핸들러 함수들 ---

/**
 * 포스트 페이지의 언어 토글 버튼 UI를 업데이트하는 함수
 */
function updatePostToggleUI(lang) {
    if (postLangToggleButton) {
        // "한"은 너무 작아서 "KOR" 과 "ENG" 로 변경하는 것을 추천합니다.
        postLangToggleButton.textContent = (lang === 'eng') ? 'kor' : 'eng';
    }
}

/**
 * 언어 토글 버튼 클릭 이벤트 리스너
 */
if (postLangToggleButton) {
    postLangToggleButton.addEventListener('click', function() {
        const urlParams = new URLSearchParams(window.location.search);
        const postId = urlParams.get('id');

        const currentLang = localStorage.getItem('language') || 'eng';
        const newLang = (currentLang === 'eng') ? 'kor' : 'eng';
        
        // 1. localStorage에 새로운 언어 설정을 저장합니다.
        localStorage.setItem('language', newLang);

        // 2. 저장된 새 언어로 콘텐츠를 다시 로드합니다.
        loadAndRenderPost(postId, newLang);
    });
}

/**
 * 페이지가 처음 로드되었을 때 실행되는 메인 로직
 */
document.addEventListener("DOMContentLoaded", function() {
    const urlParams = new URLSearchParams(window.location.search);
    const postId = urlParams.get('id');

    // localStorage에서 언어 설정을 가져오거나, 없으면 'eng'를 기본값으로 사용합니다.
    const lang = localStorage.getItem('language') || 'eng';

    // 가져온 설정으로 포스트를 로드합니다.
    loadAndRenderPost(postId, lang);
});


// --- 4. 기존에 있던 독립적인 함수들 (수정 없이 그대로 사용) ---
// (generateTOC, setupNavigationAndHighlighting 등은 여기에 위치합니다)

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
