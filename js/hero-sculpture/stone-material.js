import * as THREE from 'three';

// Packed mineral albedo / roughness / height. These variations live on the
// surface: a grazing light exposes the pores; no grain is placed over the image.
export function makeStoneTexture(renderer) {
    const size = 1024, pixels = new Uint8Array(size * size * 4);
    let seed = 1807;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    function field(n, m = n) {
        const values = Float32Array.from({ length: n * m }, random);
        return (u, v) => {
            const x = u * n, y = v * m, ix = Math.floor(x), iy = Math.floor(y);
            let a = x - ix, b = y - iy;
            a = a * a * (3 - 2 * a); b = b * b * (3 - 2 * b);
            const at = (i, j) => values[((j % m + m) % m) * n + (i % n + n) % n];
            return THREE.MathUtils.lerp(THREE.MathUtils.lerp(at(ix, iy), at(ix + 1, iy), a),
                THREE.MathUtils.lerp(at(ix, iy + 1), at(ix + 1, iy + 1), a), b) * 2 - 1;
        };
    }
    const broad = field(4), medium = field(18), mineral = field(69), fine = field(207), strata = field(9, 72);
    const byte = x => Math.round(THREE.MathUtils.clamp(x, 0, 1) * 255);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
        const u = x / size, v = y / size, i = (y * size + x) * 4;
        const a = broad(u, v), b = medium(u, v), c = mineral(u, v), d = fine(u, v);
        const layer = strata(u + b * .018, v + a * .012);
        const pore = Math.pow(Math.max(0, -d), 4), grain = random() - .5;
        pixels[i] = byte(.77 + a * .18 + b * .075 + c * .03 + layer * .045 - pore * .13);
        pixels[i + 1] = byte(.95 + b * .025 + c * .04);
        pixels[i + 2] = byte(.52 + b * .085 + c * .12 + layer * .09 + d * .055 + grain * .032 - pore * .25);
        pixels[i + 3] = 255;
    }
    // Small sparse pits break the smooth cloud pattern at close viewing distances.
    for (let p = 0; p < 1600; p++) {
        const cx = random() * size, cy = random() * size, radius = 1 + random() * 3.6;
        for (let y = Math.floor(cy - radius); y <= cy + radius; y++) for (let x = Math.floor(cx - radius); x <= cx + radius; x++) {
            const r = Math.hypot(x - cx, (y - cy) * 1.3) / radius;
            if (r >= 1) continue;
            const i = (((y + size) % size) * size + (x + size) % size) * 4, amount = (1 - r) ** 2;
            pixels[i] = Math.max(0, pixels[i] - amount * 32);
            pixels[i + 2] = Math.max(0, pixels[i + 2] - amount * 70);
        }
    }
    const texture = new THREE.DataTexture(pixels, size, size);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.magFilter = THREE.LinearFilter; texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.generateMipmaps = true; texture.colorSpace = THREE.NoColorSpace;
    texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    texture.needsUpdate = true;
    return texture;
}

export function stoneMaterial(texture, { color = '#b6b6b6', scale = .34, relief = .065 } = {}) {
    const material = new THREE.MeshPhysicalMaterial({
        color, metalness: 0, roughness: .94, clearcoat: 0,
        specularIntensity: .24, envMapIntensity: .18,
    });
    material.onBeforeCompile = shader => {
        shader.uniforms.stoneTexture = { value: texture };
        shader.uniforms.stoneScale = { value: scale };
        shader.uniforms.stoneRelief = { value: relief };
        shader.vertexShader = 'varying vec3 vStonePosition; varying vec3 vStoneNormal;\n' + shader.vertexShader;
        shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', `#include <begin_vertex>
            vStonePosition = position; vStoneNormal = normal;`);
        shader.fragmentShader = `uniform sampler2D stoneTexture;
            uniform float stoneScale; uniform float stoneRelief;
            varying vec3 vStonePosition; varying vec3 vStoneNormal;
            vec3 sampleStone(vec3 p, vec3 w) {
                return texture2D(stoneTexture, p.yz).rgb * w.x
                     + texture2D(stoneTexture, p.zx + .37).rgb * w.y
                     + texture2D(stoneTexture, p.xy + .71).rgb * w.z;
            }
            vec3 stoneNormal(vec3 position, vec3 n, float height) {
                vec3 dx = dFdx(position), dy = dFdy(position);
                vec3 rx = cross(dy, n), ry = cross(n, dx);
                float determinant = dot(dx, rx);
                vec3 gradient = sign(determinant) * (dFdx(height) * rx + dFdy(height) * ry);
                return normalize(abs(determinant) * n - gradient);
            }
        ` + shader.fragmentShader;
        shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>', `#include <map_fragment>
            vec3 stoneWeights = pow(abs(normalize(vStoneNormal)), vec3(4.0));
            stoneWeights /= max(dot(stoneWeights, vec3(1.0)), .0001);
            vec3 stoneSample = sampleStone(vStonePosition * stoneScale, stoneWeights);
            diffuseColor.rgb *= stoneSample.r;`);
        shader.fragmentShader = shader.fragmentShader.replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
            roughnessFactor *= stoneSample.g;`);
        shader.fragmentShader = shader.fragmentShader.replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
            normal = stoneNormal(-vViewPosition, normal, stoneSample.b * stoneRelief);`);
    };
    material.customProgramCacheKey = () => 'triplanar-mineral-v18';
    return material;
}
