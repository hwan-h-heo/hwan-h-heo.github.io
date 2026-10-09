import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

// Cache linear HDR color AND scene depth. Reusing depth keeps the moving form
// behind the plinth at the right pixels, including during coarse refinement.
export function createCachedStage(renderer) {
    let target;
    const quad = new FullScreenQuad(new THREE.ShaderMaterial({
        depthTest:true,depthWrite:true,depthFunc:THREE.AlwaysDepth,
        uniforms:{colorMap:{value:null},depthMap:{value:null}},
        vertexShader:'varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}',
        fragmentShader:`uniform sampler2D colorMap, depthMap; varying vec2 vUv;
            void main(){
                gl_FragColor = texture2D(colorMap,vUv);
                gl_FragDepth = texture2D(depthMap,vUv).r;
                #include <tonemapping_fragment>
                #include <colorspace_fragment>
            }`,
    }));
    return {
        bake(scene, camera, prepareShadow, hideForm) {
            const size = renderer.getDrawingBufferSize(new THREE.Vector2());
            if (!target || target.width !== size.x || target.height !== size.y) {
                target?.dispose();
                target = new THREE.WebGLRenderTarget(size.x,size.y,{
                    type:renderer.extensions.has('EXT_color_buffer_float') ? THREE.HalfFloatType : THREE.UnsignedByteType,
                    samples:4,depthTexture:new THREE.DepthTexture(size.x,size.y),
                });
                quad.material.uniforms.colorMap.value = target.texture;
                quad.material.uniforms.depthMap.value = target.depthTexture;
            }
            const previous = renderer.getRenderTarget();
            renderer.setRenderTarget(target);
            prepareShadow();
            // One full-pose pass creates the cast shadow. The second stores the
            // architecture with that shadow but without the moving sculpture.
            renderer.render(scene,camera);
            hideForm(true); renderer.render(scene,camera); hideForm(false);
            renderer.setRenderTarget(previous);
        },
        render(scene,camera) {
            const background = scene.background, autoClear = renderer.autoClear;
            scene.background = null; renderer.autoClear = false;
            renderer.clear(); quad.render(renderer); renderer.render(scene,camera);
            scene.background = background; renderer.autoClear = autoClear;
        },
    };
}

export function movingStoneMaterial(texture) {
    const material = new THREE.MeshStandardMaterial({color:'#c2c2c2',metalness:0,roughness:.96,envMapIntensity:0});
    // Keep the mineral albedo and studio lights. During motion omit the height
    // derivative, roughness map, physical specular extension and environment map.
    material.onBeforeCompile = shader => {
        shader.uniforms.stoneTexture = {value:texture};
        shader.vertexShader = 'varying vec3 vMineralPosition; varying vec3 vMineralNormal;\n' + shader.vertexShader;
        shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvMineralPosition=position;vMineralNormal=normal;');
        shader.fragmentShader = 'uniform sampler2D stoneTexture; varying vec3 vMineralPosition; varying vec3 vMineralNormal;\n' + shader.fragmentShader;
        shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>', `#include <map_fragment>
            vec3 w = pow(abs(normalize(vMineralNormal)),vec3(4.0)); w /= max(dot(w,vec3(1.0)),.0001);
            vec3 p = vMineralPosition * .68;
            float mineral = texture2D(stoneTexture,p.yz).r*w.x + texture2D(stoneTexture,p.zx+.37).r*w.y + texture2D(stoneTexture,p.xy+.71).r*w.z;
            diffuseColor.rgb *= mineral;`);
        // Standard uses a stronger default dielectric response than the source
        // physical material; retain the source's restrained specular intensity.
        shader.fragmentShader = shader.fragmentShader.replace('material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );', 'material.specularColor = mix( vec3( 0.0096 ), diffuseColor.rgb, metalnessFactor );');
    };
    material.customProgramCacheKey = () => 'moving-mineral-v29';
    return material;
}
