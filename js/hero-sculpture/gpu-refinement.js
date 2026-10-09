import * as THREE from 'three';
import { buildAncestors } from './mesh-refinement.js';

const LENGTH = 192, RADIAL = 24, STRIDE = RADIAL + 1;
const ease = x => { x = THREE.MathUtils.clamp(x, 0, 1); return x * x * (3 - 2 * x); };
const declarations = `
    uniform float refineTime;
    attribute vec3 ancestor0, ancestor1, ancestor2;
    attribute vec3 ancestorNormal0, ancestorNormal1, ancestorNormal2;
    attribute vec3 refineSchedule;
    vec3 refinementWeights() {
        vec3 w = smoothstep(vec3(0.0), vec3(1.0), (vec3(refineTime) - refineSchedule) / .62);
        w.y = min(w.x, w.y); w.z = min(w.y, w.z); return w;
    }
    vec3 refinedPosition(vec3 w) {
        return ancestor0 + w.x * (ancestor1 - ancestor0) + w.y * (ancestor2 - ancestor1) + w.z * (position - ancestor2);
    }
    vec3 refinedNormal(vec3 w) {
        return normalize(ancestorNormal0 + w.x * (ancestorNormal1 - ancestorNormal0)
            + w.y * (ancestorNormal2 - ancestorNormal1) + w.z * (normal - ancestorNormal2));
    }
`;

// The ancestor surfaces/schedule remain identical. Upload positions and smooth
// endpoint normals once; per-frame work is one uniform, not normals + buffer uploads.
export function createMeshRefinement(mesh) {
    const geometry = mesh.geometry, ancestors = buildAncestors(geometry);
    const count = geometry.attributes.position.count, schedules = new Float32Array(count * 3);
    const normals = ancestors.map((positions, level) => {
        if (level === 3) return geometry.attributes.normal.array.slice();
        const clone = geometry.clone();
        clone.setAttribute('position', new THREE.BufferAttribute(positions.slice(), 3));
        clone.computeVertexNormals();
        const normal = clone.attributes.normal, a = new THREE.Vector3(), b = new THREE.Vector3();
        const join = (i, j) => { a.fromBufferAttribute(normal, i); b.fromBufferAttribute(normal, j); a.add(b).normalize(); normal.setXYZ(i, a.x, a.y, a.z); normal.setXYZ(j, a.x, a.y, a.z); };
        for (let i = 0; i <= LENGTH; i++) join(i * STRIDE, i * STRIDE + RADIAL);
        for (let j = 0; j <= RADIAL; j++) join(j, LENGTH * STRIDE + j);
        const result = normal.array.slice(); clone.dispose(); return result;
    });
    for (let i = 0; i <= LENGTH; i++) for (let j = 0; j <= RADIAL; j++) {
        const index = i * STRIDE + j, u = i % LENGTH, v = j % RADIAL;
        const front = (1 - THREE.MathUtils.clamp(ancestors[0][index * 3 + 1] / 3, 0, 1)) * .60;
        const locality = ((Math.floor(u / 8) * 7 + Math.floor(v / 8) * 3) % 5) * .045;
        schedules[index * 3] = .65 + front + locality;
        for (let level = 1; level < 3; level++) {
            const step = 8 / 2 ** level, rank = (Math.floor(u / step) % 2) * 2 + Math.floor(v / step) % 2;
            schedules[index * 3 + level] = schedules[index * 3 + level - 1] + .70 + rank * .045;
        }
    }
    for (let level = 0; level < 3; level++) {
        geometry.setAttribute('ancestor' + level, new THREE.BufferAttribute(ancestors[level], 3));
        geometry.setAttribute('ancestorNormal' + level, new THREE.BufferAttribute(normals[level], 3));
    }
    geometry.setAttribute('refineSchedule', new THREE.BufferAttribute(schedules, 3));
    const timeUniform = {value:4};
    function applyMaterial(material, wire = false) {
        const previous = material.onBeforeCompile, key = material.customProgramCacheKey();
        material.onBeforeCompile = shader => {
            previous.call(material, shader);
            shader.uniforms.refineTime = timeUniform;
            shader.vertexShader = declarations + (wire ? 'attribute float refineLevel; varying float vRefineOpacity;\n' : '') + shader.vertexShader;
            shader.vertexShader = shader.vertexShader.replace('#include <beginnormal_vertex>', `
                vec3 refineWeights = refinementWeights();
                vec3 objectNormal = refinedNormal(refineWeights);
            `);
            shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', wire ? `
                vec3 refineWeights = refinementWeights();
                vec3 transformed = refinedPosition(refineWeights) + refinedNormal(refineWeights) * .002;
                float reveal = refineLevel < .5 ? 1.0 : refineLevel < 1.5 ? refineWeights.x : refineWeights.y;
                float dissolve = refineLevel < .5 ? refineWeights.x : refineLevel < 1.5 ? refineWeights.y : refineWeights.z;
                vRefineOpacity = max(0.0, reveal - dissolve) * (refineLevel < .5 ? .22 : .15);
            ` : 'vec3 transformed = refinedPosition(refineWeights);');
            if (wire) {
                shader.fragmentShader = 'varying float vRefineOpacity;\n' + shader.fragmentShader;
                shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.a *= vRefineOpacity;');
            }
        };
        material.customProgramCacheKey = () => key + '-gpu-refinement-v29' + (wire ? '-wire' : '');
    }
    applyMaterial(mesh.material);
    const indices = [], levels = [];
    for (let level = 0; level < 3; level++) {
        const step = 8 / 2 ** level;
        for (let i = 0; i < LENGTH; i += step) for (let j = 0; j < RADIAL; j += step) {
            const a = i * STRIDE + j, b = a + step, c = (i + step) * STRIDE + j;
            indices.push(a, b, a, c, b, c); levels.push(level, level, level, level, level, level);
        }
    }
    const wireGeometry = new THREE.BufferGeometry();
    for (const name of ['position','normal','ancestor0','ancestor1','ancestor2','ancestorNormal0','ancestorNormal1','ancestorNormal2','refineSchedule']) {
        const source = geometry.attributes[name].array, data = new Float32Array(indices.length * 3);
        indices.forEach((index, i) => data.set(source.subarray(index * 3, index * 3 + 3), i * 3));
        wireGeometry.setAttribute(name, new THREE.BufferAttribute(data, 3));
    }
    wireGeometry.setAttribute('refineLevel', new THREE.Float32BufferAttribute(levels, 1));
    const wireMaterial = new THREE.LineBasicMaterial({color:'#b6bec0',transparent:true,opacity:1,depthWrite:false});
    applyMaterial(wireMaterial, true);
    const wire = new THREE.LineSegments(wireGeometry, wireMaterial);
    wire.frustumCulled = false; wire.visible = false; wire.renderOrder = 1; mesh.add(wire);
    let active = false, elapsed = 4, sampledTime = 4, lastTime = null, renders = 0, plays = 0;
    function seek(progress) {
        const time = THREE.MathUtils.clamp(progress, 0, 1) * 4;
        if (time === sampledTime) return false;
        sampledTime = time; timeUniform.value = time; wire.visible = time < 4; renders++; return true;
    }
    function samplePosition(i, progress, out) {
        let parent = 1;
        out.fromArray(ancestors[0], i * 3);
        for (let level = 0; level < 3; level++) {
            const weight = Math.min(parent, ease((progress * 4 - schedules[i * 3 + level]) / .62)); parent = weight;
            out.x += weight * (ancestors[level + 1][i * 3] - ancestors[level][i * 3]);
            out.y += weight * (ancestors[level + 1][i * 3 + 1] - ancestors[level][i * 3 + 1]);
            out.z += weight * (ancestors[level + 1][i * 3 + 2] - ancestors[level][i * 3 + 2]);
        }
        return out;
    }
    return {
        applyMaterial, samplePosition, seek,
        start() {active = true; elapsed = 0; lastTime = null; plays++; seek(0);},
        resume() {lastTime = null;},
        update(now) {
            if (active) {
                if (lastTime !== null) elapsed = Math.min(4, elapsed + Math.min((now - lastTime) / 1000, .06));
                lastTime = now;
                if (elapsed === 4) { active = false; lastTime = null; }
            }
            return seek(elapsed / 4);
        },
        finish() {active = false; elapsed = 4; lastTime = null; return seek(1);},
        get active() {return active;},
        state: () => ({active,elapsed,progress:sampledTime/4,renders,plays,fromCoarse:plays>0,levels:[144,576,2304,9216],wireVisible:wire.visible,backend:'gpu',vertexUploadsDuringPlayback:0}),
    };
}
