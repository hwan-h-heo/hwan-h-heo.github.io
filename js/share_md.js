function initializeShareFunctionality() {

    const copyButton = document.getElementById('copyButton');
    const share_modal = document.getElementById('myshare_modal');
    const share_modal_closeBtn = document.querySelector('.share_modal_close');
    const indicator = document.getElementById('share_modalIndicator');
    
    if (!copyButton || !share_modal || !share_modal_closeBtn || !indicator) {
        console.log("Share functionality elements not found. Skipping initialization.");
        return;
    }

    let animationId; // requestAnimationFrame ID

    function updateShareButtonVisibility() {
        var headerHeight = document.querySelector('.masthead').offsetHeight;
        if (window.innerWidth > 1280 && window.scrollY > headerHeight) {
            copyButton.style.display = 'block'; 
        } else {
            copyButton.style.display = 'none'; 
        }
    }

    function animateIndicator() {
        let startTime = null;
        const duration = 1500; // 1.5초 동안 애니메이션

        function step(timestamp) {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / duration, 1);
            indicator.style.width = `${(1 - progress) * 100}%`;

            if (progress < 1) {
                animationId = requestAnimationFrame(step);
            } else {
                share_modal.style.display = 'none';
            }
        }
        
        cancelAnimationFrame(animationId);
        animationId = requestAnimationFrame(step);
    }

    function closeShareModal() {
        cancelAnimationFrame(animationId);
        share_modal.style.display = 'none';
        indicator.style.width = '0%';
    }

    // 4. 모든 이벤트 리스너를 등록합니다.
    
    // 페이지 로드 시, 스크롤 시, 창 크기 조절 시 버튼 가시성 업데이트
    document.addEventListener('scroll', updateShareButtonVisibility);
    window.addEventListener('resize', updateShareButtonVisibility);
    
    // 링크 복사 버튼 클릭 이벤트
    copyButton.addEventListener('click', () => {
        const url = new URL(window.location.href);
        url.hash = ''; // URL에서 # 이후 부분은 제거
        const urlWithoutHash = url.href;

        navigator.clipboard.writeText(urlWithoutHash)
            .then(() => {
                share_modal.style.display = 'block';
                indicator.style.width = '100%'; // 프로그레스 바 초기화
                animateIndicator();
            })
            .catch(err => {
                console.error('Link Copy Failed:', err);
                alert('링크 복사에 실패했습니다.');
            });
    });

    share_modal_closeBtn.addEventListener('click', closeShareModal);

    window.addEventListener('click', (event) => {
        if (event.target === share_modal) {
            closeShareModal();
        }
    });

    updateShareButtonVisibility();
}