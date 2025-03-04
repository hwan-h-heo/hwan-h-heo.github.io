const copyButton = document.getElementById('copyButton');
const share_modal = document.getElementById('myshare_modal');
const share_modal_closeBtn = document.querySelector('.share_modal_close');
const indicator = document.getElementById('share_modalIndicator');
let timeoutId;
let animationId; // requestAnimationFrame ID 

document.addEventListener("scroll", function() {
    var headerHeight = document.querySelector('.masthead').offsetHeight;
    if (window.scrollY > headerHeight) {
        copyButton.style.display = 'block'; 
    } else {
        copyButton.style.display = 'none'; 
    }
});

function animateIndicator() {
    let startTime = null;
    const duration = 1500; // ms

    function step(timestamp) {
        if (!startTime) startTime = timestamp;
        const progress = Math.min((timestamp - startTime) / duration, 1); 
        indicator.style.width = `${(1 - progress) * 100}%`; 

        if (progress < 1) {
            animationId = requestAnimationFrame(step); 
        } else {
        share_modal.style.display = 'none';
        indicator.style.width = '0%';
        }
    }

    animationId = requestAnimationFrame(step); 
}


copyButton.addEventListener('click', () => {
    const currentURL = window.location.href;

    navigator.clipboard.writeText(currentURL)
    .then(() => {
        clearTimeout(timeoutId);
        cancelAnimationFrame(animationId);

        share_modal.style.display = 'block';
        indicator.style.width = '0%'; 

        animateIndicator(); 

    })
    .catch(err => {
        console.error('Link Copy Failed:', err);
        alert('Link Copy Failed.');
    });
});

share_modal_closeBtn.addEventListener('click', () => {
    clearTimeout(timeoutId);
    cancelAnimationFrame(animationId); 
    share_modal.style.display = 'none';
    indicator.style.width = '0%';
});

window.addEventListener('click', (event) => {
    if (event.target === share_modal) {
    clearTimeout(timeoutId);
    cancelAnimationFrame(animationId); 
    share_modal.style.display = 'none';
    indicator.style.width = '0%';
    }
});