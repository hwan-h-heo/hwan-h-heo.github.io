import * as GaussianSplats3D from 'GaussianSplats3D';
import * as THREE from 'three';

// 이 함수는 post.html에서 필요한 시점에 호출될 것입니다.
export function initGaussianViewer() {
    const renderWidth = 640;
    const renderHeight = 360;

    const rootElement = document.getElementById('canvasContainer');
    if (!rootElement) {
        console.error("Canvas container not found!");
        return;
    }
    rootElement.style.width = renderWidth + 'px';
    rootElement.style.height = renderHeight + 'px';

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(renderWidth, renderHeight);
    renderer.setClearColor(0xf8f9fa, 1);
    rootElement.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(65, renderWidth / renderHeight, 0.1, 500);
    camera.position.copy(new THREE.Vector3().fromArray([-1.5, -2, 3]));
    camera.up = new THREE.Vector3().fromArray([0, -1, -0.6]).normalize();
    camera.lookAt(new THREE.Vector3().fromArray([0, 3, 0]));

    const viewer = new GaussianSplats3D.Viewer({
        'selfDrivenMode': false, // selfDrivenMode는 false로 하고 수동으로 update를 호출하는 것이 안정적입니다.
        'renderer': renderer,
        'camera': camera,
        'useBuiltInControls': true,
        // ... 나머지 옵션들은 그대로 ...
    });

    viewer.addSplatScene('https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/guitar_gs.ksplat', {
        'position': [-0.7, -0.3, 0.9],
        'rotation': [0, 1, 0.2, 0.1],
        'scale': [3, 3, 3]
    })
    .then(() => {
        // 로드가 완료되면 렌더링 루프를 시작합니다.
        function update() {
            requestAnimationFrame(update);
            viewer.update();
            viewer.render();
        }
        update();
    });
}