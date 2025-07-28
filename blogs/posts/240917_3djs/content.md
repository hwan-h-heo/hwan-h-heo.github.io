title: Add Gaussian Splatting to Your Website
date: September 17, 2024
author: Hwan Heo
--- 여기부터 실제 콘텐츠 ---

<button id="copyButton">
    <i class="bi bi-share-fill"></i>
</button>

<div id="myshare_modal" class="share_modal">
    <div class="share_modal-content">
        <span class="share_modal_close">×</span>
        <p><strong>Link Copied!</strong></p>
        <div class="copy_indicator-container">
        <div class="copy_indicator" id="share_modalIndicator"></div>
        </div>
    </div>
</div>

<nav class="toc">
    <ul>
        <li>
            <a href="#tl-dr"> TL; DR</a>
        </li>
        <li><a href="#step"> Step-by-Step</a></li>
        <ul>
            <li><a href="#step1"> Step1: CDN setting</a></li>
            <li><a href="#step2"> Step2: HTML Structures</a></li>
            <li><a href="#step3"> Example Result</a></li>
        </ul>
    </ul>
</nav>

<br/>
<h2 id="tl-dr">TL; DR</h2>
<p class="lang eng"> 
    <em>(short article)</em> <br/>
    In this tutorial, I’ll guide you how to integrate 3D Gaussian splatting into your website. 
    Gaussian splatting is a rendering technique that represents point clouds as smooth, detailed "splats" rather than simple points, creating impressive visuals.
</p>
<p class="lang eng"> 
    Don’t worry if you're not a web expert—just follow along and copy-paste the code provided! 
</p>
<p class="lang kor" style="display: none;">
    html 웹페이지에 학습시킨 3D Gaussian Splatting scene 을 띄우는 방법을 알아보자.
    웹알못이어도 상관없이 따라 하기만 하면 된다!
</p>
<h3>
    About the Project
</h3>
<p class="lang eng">
    We’ll be using the <span style="text-decoration: underline;"><a href="https://github.com/mkkellogg/GaussianSplats3D"> 3D Gaussian Splatting for Three.js </a></span> by mkkellogg. 
    This package allows you to render point clouds as smooth, Gaussian splats in a 3D environment using Three.js. 
    We’ll use this package to integrate advanced point-based rendering into our website. 
    Now, let’s walk through the setup process.
</p>
<p class="lang kor" style="display: none;">
    Three-js 을 이용해 구현된 <span style="text-decoration: underline;"><a href="https://github.com/mkkellogg/GaussianSplats3D"> 3D Gaussian Splatting for Three.js </a></span> 을 이용할 것이다. 
    웹잘알이면 패키지 document 만 참고하면 될 듯 하다.
</p>
<br/>

<h2 id="step">
    Step-by-Step
</h2>
<hr/>
<h3 id="step1">Step 1: Project Setup (CDN-based)</h3>
<p class="lang eng">
    For those not familiar with package management or setting up a local development environment, you can use a CDN to load the necessary libraries directly into your HTML file. Here’s how you can set up Three.js and GaussianSplats3D using CDN links:
</p>
<p class="lang kor" style="display: none;">
    npm 등으로 threejs 와 GaussianSplats3D package 를 설치하는 것이 일반적이지만, (저자처럼) 웹에 친숙하지 않은 것을 가정하고 쓰는 글이기 때문에 
    CDN 을 이용해서 package 를 로드하여 사용하는 방법을 기술한다. 
    다음과 같이 필요한 패키지를 로드하여 사용하면 된다. 
</p>
                        <pre class="language-javascript" style="font-size: 16px;"><code>&lt;script type="importmap"&gt;
{
    "imports": {
        "three": "https://unpkg.com/three@0.150.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.157.0/examples/jsm/",
        "GaussianSplats3D": "https://unpkg.com/@mkkellogg/gaussian-splats-3d@0.4.0/build/gaussian-splats-3d.module.js"
    }
}
&lt;/script&gt; </code></pre><br/>

<h3 id="step2">Step 2: HTML Structure</h3>
<p class="lang eng">
We will now initialize the WebGL renderer, set the camera’s position, and use the GaussianSplats3D viewer to load a 3D model (in .ply or .ksplat format, for example). 
</p>
<p class="lang eng">
<strong>Note: </strong>
If you follow the basic usage from the GitHub repo, the 3D content will be rendered to fill the entire page. 
Therefore, I’ve implemented this so the 3D scene appears only within a predefined canvas region. 
You can easily modify the canvas size to suit your needs.
<br/>
The code below handles all of these details:
</p>
<p class="lang kor" style="display: none;">
필요 패키지를 로드하면 이제 description 에 맞게 3D GS scene 을 세팅하여 WebGL renderer 를 불러오면 된다. 
3D GS ply scene 이면 모두 로드 가능하며, 본인의 3D GS scene 에 맡게 카메라나 GS 의 position, rotation 등을 수정하면 된다. 
패키지 basic usgae 처럼 불러오면 화면 전체에 3D content 가 렌더링 되기 때문에 canvas div 를 선언하고 이 영역 내에서만 렌더링 되도록 작업하였다. 
</p>
<div class="accordion accordion-flush" id="accordionFlushExample">
<div class="accordion-item">
<h2 class="accordion-header">
<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#flush-collapseOne" aria-expanded="false" aria-controls="flush-collapseOne">
    <strong><em>Show Code!</em></strong>
</button>
</h2>
<div id="flush-collapseOne" class="accordion-collapse collapse" data-bs-parent="#accordionFlushExample">
<div class="accordion-body">

```javascript
<script type="module">
// Create a new THREE.js scene
import * as GaussianSplats3D from 'GaussianSplats3D';
import * as THREE from 'three';

// Set the desired render width and height
const renderWidth = 640;
const renderHeight = 360;

// Get the canvas container
const rootElement = document.getElementById('canvasContainer');
rootElement.style.width = renderWidth + 'px';
rootElement.style.height = renderHeight + 'px';

// Initialize WebGL renderer
const renderer = new THREE.WebGLRenderer({
    antialias: true
});
renderer.setSize(renderWidth, renderHeight);
renderer.setClearColor(0xf8f9fa, 1);
rootElement.appendChild(renderer.domElement);

// Initialize camera
const camera = new THREE.PerspectiveCamera(65, renderWidth / renderHeight, 0.1, 500);
camera.position.copy(new THREE.Vector3().fromArray([-1.5, -2, 3]));
camera.up = new THREE.Vector3().fromArray([0, -1, -0.6]).normalize();
camera.lookAt(new THREE.Vector3().fromArray([0, 3, 0]));
                
// Initialize the GaussianSplats3D viewer
const viewer = new GaussianSplats3D.Viewer({
    'selfDrivenMode': true,
    'renderer': renderer,
    'camera': camera,
    'useBuiltInControls': true,
    'ignoreDevicePixelRatio': false,
    'gpuAcceleratedSort': true,
    'enableSIMDInSort': true,
    'sharedMemoryForWorkers': false,
    'integerBasedSort': true,
    'halfPrecisionCovariancesOnGPU': true,
    'dynamicScene': false,
    'webXRMode': GaussianSplats3D.WebXRMode.None,
    'renderMode': GaussianSplats3D.RenderMode.OnChange,
    'sceneRevealMode': GaussianSplats3D.SceneRevealMode.Instant,
    'antialiased': false,
    'focalAdjustment': 1.0,
    'logLevel': GaussianSplats3D.LogLevel.None,
    'sphericalHarmonicsDegree': 0,
    'enableOptionalEffects': false,
    'plyInMemoryCompressionLevel': 2,
    'freeIntermediateSplatData': false
});
                
// Load a 3D scene (replace with the actual path to your model)
viewer.addSplatScene('<path/to/your/gs path>', {
    'position': [-0.7, -0.3, 0.9],
    'rotation': [0, 1, 0.2, 0.1],
    'scale': [3, 3, 3]
})
.then(() => {
    requestAnimationFrame(update);
});

// Update function to render the scene

function update() {
    requestAnimationFrame(update);
    viewer.update();
    viewer.render();
}
</script>;
```

</div>
</div>
<p>
    Tip.
</p>
<ul class="lang eng">
    <li>
        Set <code>'sharedMemoryForWorkers': false</code> for preventing CORS error.
    </li>
</ul>
<ul class="lang kor" style="display: none;">
    <li>
        CORS error 방지를 위해  <code>'sharedMemoryForWorkers': false</code> 로 설정하였다. 
    </li>
</ul>
<br/>
<h3 id="step3">
    Example Result of Custom GS Scene
</h3>
<div style="margin-bottom: -20vh;">
    <p class="lang eng">
        Here is the result of the above code. 
        The loaded Gaussian Splatting is my custom captured miniature guitar. 
        You can directly interact with the 3D Scene!
    </p>
    <p class="lang kor" style="display: none;">
        위 방법을 이용해 로드된 3D Gaussian Splatting scene 이다. 
        Interactive 하게 canvas 안의 3d content 를 컨트롤 할 수 있다.
        로드된 scene 은 직접 촬영 후 recon 한 미니어쳐 기타인데, artifacts 나 floaters 가 좀 있기는 하다 ㅠㅠ.  
    </p>
</div>
<div id="canvasContainer" style="margin-bottom: 30vh;">
    <canvas id="threeCanvas"></canvas>
</div>

<h2>Closing</h2>
<p class="lang eng"> 
    For researchers, this topic might not be of direct interest to everyone. 
    However, in an era where personal projects and portfolios are increasingly important, 
    I believe that learning how to add a custom Gaussian splatting scene to personal webpage could be valuable for computer vision researchers who keep an eye on the field of neural rendering. 
    If you're curious about other research related to neural rendering, feel free to check out the other posts on my blog. 
    <br/><br/>
    That’s all for now!
</p>
<p class="lang kor" style="display: none;">
    연구자라면 이 article 이 관심 없는 주제일지도 모르지만, 
    자기 pr 의 시대에 직접 학습시킨 gaussian splatting scene 을 개인 webpage 에 add 하는 방법 또한 누군가에게는 도움이 될 거라고 믿으며 글을 작성하였다.
    :)
</p>

<hr/>
<p>
    You may also like, 
</p>
<ul>
    <li>
        <a href="../240805_gs/">
            <span style="text-decoration: underline;">A Comprehensive Analysis of Gaussian Splatting Rasterization</span>
        </a>
    </li>
    <li>
        <a href="../240823_grt/">
            <span style="text-decoration: underline;">Don't Rasterize But Ray Trace Gaussian</span>
        </a>
    </li>
    <li>
        <a href="../240602_2dgs/">
            <span style="text-decoration: underline;">Under the 3D: Geometrically Accurate 2D Gaussian Splatting </span>
        </a>
    </li>
</ul>
<br/>