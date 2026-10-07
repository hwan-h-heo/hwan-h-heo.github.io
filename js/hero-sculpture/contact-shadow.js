import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

// Cache an orthographic depth projection of nearby undersides. Refresh only when
// the sculpture pose changes; pointer parallax reuses the texture.
export function contactShadow(renderer, objects, { x, y, z, size, height, opacity }) {
    const scene = new THREE.Scene();
    const root = objects[0].parent, inverse = new THREE.Matrix4();
    const copies = objects.map(object => {
        const copy = object.clone(); copy.matrixAutoUpdate = false;
        scene.add(copy); return copy;
    });
    const camera = new THREE.OrthographicCamera(-size / 2, size / 2, size / 2, -size / 2, .001, height);
    camera.position.set(x, y, z); camera.up.set(0, 0, -1); camera.lookAt(x, y + 1, z);
    const depth = new THREE.MeshDepthMaterial({ side: THREE.DoubleSide });
    depth.onBeforeCompile = shader => {
        shader.fragmentShader = shader.fragmentShader.replace(
            'gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );',
            'gl_FragColor = vec4(0.0, 0.0, 0.0, pow(1.0 - fragCoordZ, 3.0));',
        );
    };
    scene.overrideMaterial = depth;
    const target = new THREE.WebGLRenderTarget(512, 512), temporary = target.clone();
    const blur = new FullScreenQuad(new THREE.ShaderMaterial({
        depthTest: false, depthWrite: false,
        uniforms: { map: { value: target.texture }, direction: { value: new THREE.Vector2(2 / 512, 0) } },
        vertexShader: 'varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}',
        fragmentShader: `uniform sampler2D map;uniform vec2 direction;varying vec2 vUv;
            void main(){
                vec4 c=texture2D(map,vUv)*.227027;
                c+=(texture2D(map,vUv+direction*1.384615)+texture2D(map,vUv-direction*1.384615))*.316216;
                c+=(texture2D(map,vUv+direction*3.230769)+texture2D(map,vUv-direction*3.230769))*.070270;
                gl_FragColor=c;
            }`,
    }));
    function update() {
        root.updateWorldMatrix(true, true); inverse.copy(root.matrixWorld).invert();
        objects.forEach((object, i) => {
            copies[i].matrix.multiplyMatrices(inverse, object.matrixWorld);
            copies[i].matrixWorldNeedsUpdate = true;
        });
        const previous = renderer.getRenderTarget(), clear = renderer.getClearColor(new THREE.Color()), alpha = renderer.getClearAlpha();
        renderer.setClearColor(0, 0);
        renderer.setRenderTarget(target); renderer.render(scene, camera);
        blur.material.uniforms.map.value = target.texture; blur.material.uniforms.direction.value.set(2 / 512, 0);
        renderer.setRenderTarget(temporary); blur.render(renderer);
        blur.material.uniforms.map.value = temporary.texture; blur.material.uniforms.direction.value.set(0, 2 / 512);
        renderer.setRenderTarget(target); blur.render(renderer);
        renderer.setRenderTarget(previous); renderer.setClearColor(clear, alpha);
    }
    update();
    target.texture.repeat.x = -1; target.texture.offset.x = 1;
    const shadow = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({
        map: target.texture, transparent: true, opacity, depthWrite: false, toneMapped: false,
    }));
    shadow.position.set(x, y, z); shadow.rotation.x = -Math.PI / 2; shadow.renderOrder = 1;
    shadow.userData.update = update;
    shadow.userData.setPosition = (x, y, z) => {
        shadow.position.set(x, y, z);
        camera.position.set(x, y, z); camera.lookAt(x, y + 1, z);
    };
    return shadow;
}
