import * as THREE from 'three';

const ease = x => { x = THREE.MathUtils.clamp(x, 0, 1); return x * x * (3 - 2 * x); };
const STEPS = [8, 4, 2, 1];
const LENGTH = 192, RADIAL = 24, STRIDE = RADIAL + 1;

// The same fine topology samples each ancestor's *triangular* surface. Child
// vertices start inside their parent face, then move toward the finer surface.
// This is a surface-refinement interpretation, not neural/implicit-field inference.
export function buildAncestors(geometry) {
    const source = geometry.attributes.position.array;
    if (source.length !== (LENGTH + 1) * STRIDE * 3) throw new Error('Unexpected closed surface grid');
    return STEPS.map(step => {
        const positions = new Float32Array(source.length);
        for (let i = 0; i <= LENGTH; i++) for (let j = 0; j <= RADIAL; j++) {
            const u = Math.min(Math.floor(i / step) * step, LENGTH - step);
            const v = Math.min(Math.floor(j / step) * step, RADIAL - step);
            const x = (i - u) / step, y = (j - v) / step;
            const a = (u * STRIDE + v) * 3, b = (u * STRIDE + v + step) * 3;
            const c = ((u + step) * STRIDE + v) * 3, d = ((u + step) * STRIDE + v + step) * 3;
            const out = (i * STRIDE + j) * 3;
            for (let axis = 0; axis < 3; axis++) positions[out + axis] = x + y <= 1
                ? source[a + axis] * (1 - x - y) + source[c + axis] * x + source[b + axis] * y
                : source[d + axis] * (x + y - 1) + source[c + axis] * (1 - y) + source[b + axis] * (1 - x);
        }
        return positions;
    });
}

export function createMeshRefinement(mesh) {
    const geometry = mesh.geometry, ancestors = buildAncestors(geometry);
    const finalNormals = geometry.attributes.normal.array.slice();
    const count = geometry.attributes.position.count;
    const schedules = Array.from({ length: 3 }, () => new Float32Array(count));
    for (let i = 0; i <= LENGTH; i++) for (let j = 0; j <= RADIAL; j++) {
        const index = i * STRIDE + j, u = i % LENGTH, v = j % RADIAL;
        const parentU = Math.floor(u / 8), parentV = Math.floor(v / 8);
        // A spatially ordered front, with child ranks nested inside each parent.
        const height = ancestors[0][index * 3 + 1];
        const front = (1 - THREE.MathUtils.clamp(height / 3, 0, 1)) * .60;
        const locality = ((parentU * 7 + parentV * 3) % 5) * .045;
        schedules[0][index] = .65 + front + locality;
        for (let level = 1; level < 3; level++) {
            const step = STEPS[level], rank = (Math.floor(u / step) % 2) * 2 + Math.floor(v / step) % 2;
            schedules[level][index] = schedules[level - 1][index] + .70 + rank * .045;
        }
    }

    const wireIndices = [], wireLevels = [];
    for (let level = 0; level < 3; level++) {
        const step = STEPS[level];
        for (let i = 0; i < LENGTH; i += step) for (let j = 0; j < RADIAL; j += step) {
            const a = i * STRIDE + j, b = a + step, c = (i + step) * STRIDE + j;
            // The parent diagonal makes it clear that each face contains children.
            wireIndices.push(a, b, a, c, b, c);
            for (let k = 0; k < 6; k++) wireLevels.push(level);
        }
    }
    const wireGeometry = new THREE.BufferGeometry();
    const wirePositions = new Float32Array(wireIndices.length * 3), wireVisibility = new Float32Array(wireIndices.length);
    wireGeometry.setAttribute('position', new THREE.BufferAttribute(wirePositions, 3));
    wireGeometry.setAttribute('refineOpacity', new THREE.BufferAttribute(wireVisibility, 1));
    const wireMaterial = new THREE.LineBasicMaterial({ color: '#b6bec0', transparent: true, opacity: 1, depthWrite: false });
    wireMaterial.onBeforeCompile = shader => {
        shader.vertexShader = 'attribute float refineOpacity; varying float vRefineOpacity;\n' + shader.vertexShader;
        shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvRefineOpacity = refineOpacity;');
        shader.fragmentShader = 'varying float vRefineOpacity;\n' + shader.fragmentShader;
        shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.a *= vRefineOpacity;');
    };
    const wire = new THREE.LineSegments(wireGeometry, wireMaterial);
    wire.frustumCulled = false; wire.visible = false; wire.renderOrder = 1;
    mesh.add(wire);
    let active = false, elapsed = 4, sampledTime = 4, lastTime = null, renders = 0, plays = 0;
    const weights = Array.from({ length: 3 }, () => new Float32Array(count));
    const scratch = new THREE.Vector3(), other = new THREE.Vector3();

    function smoothSeams() {
        const normals = geometry.attributes.normal;
        function join(a, b) {
            scratch.fromBufferAttribute(normals, a); other.fromBufferAttribute(normals, b);
            scratch.add(other).normalize();
            normals.setXYZ(a, scratch.x, scratch.y, scratch.z); normals.setXYZ(b, scratch.x, scratch.y, scratch.z);
        }
        for (let i = 0; i <= LENGTH; i++) join(i * STRIDE, i * STRIDE + RADIAL);
        for (let j = 0; j <= RADIAL; j++) join(j, LENGTH * STRIDE + j);
    }

    function sample(time) {
        const positions = geometry.attributes.position.array;
        for (let i = 0; i < count; i++) {
            let parent = 1;
            for (let level = 0; level < 3; level++) {
                const weight = Math.min(parent, ease((time - schedules[level][i]) / .62));
                weights[level][i] = weight; parent = weight;
            }
            for (let axis = 0; axis < 3; axis++) {
                const index = i * 3 + axis;
                let value = ancestors[0][index];
                for (let level = 0; level < 3; level++) value += weights[level][i] * (ancestors[level + 1][index] - ancestors[level][index]);
                positions[index] = value;
            }
        }
        geometry.attributes.position.needsUpdate = true;
        geometry.computeVertexNormals(); smoothSeams();
        const normals = geometry.attributes.normal.array;
        for (let i = 0; i < wireIndices.length; i++) {
            const vertex = wireIndices[i], level = wireLevels[i];
            for (let axis = 0; axis < 3; axis++) wirePositions[i * 3 + axis] = positions[vertex * 3 + axis] + normals[vertex * 3 + axis] * .002;
            const reveal = level === 0 ? 1 : weights[level - 1][vertex];
            wireVisibility[i] = Math.max(0, reveal - weights[level][vertex]) * (level === 0 ? .22 : .15);
        }
        wireGeometry.attributes.position.needsUpdate = true; wireGeometry.attributes.refineOpacity.needsUpdate = true;
        renders++;
    }

    // Both the introduction and reverse scroll sample this exact same surface.
    // Seeking has no independent playback clock and does no work at rest.
    function seek(progress) {
        const time = THREE.MathUtils.clamp(progress, 0, 1) * 4;
        if (time === sampledTime) return false;
        sampledTime = time; wire.visible = time < 4;
        if (time < 4) sample(time);
        else {
            geometry.attributes.position.array.set(ancestors[3]); geometry.attributes.position.needsUpdate = true;
            geometry.attributes.normal.array.set(finalNormals); geometry.attributes.normal.needsUpdate = true;
        }
        return true;
    }
    function finish() {
        active = false; elapsed = 4; lastTime = null;
        return seek(1);
    }
    return {
        start() {
            active = true; elapsed = 0; lastTime = null; plays++;
            seek(0);
        },
        resume() { lastTime = null; },
        update(now, reverseProgress = 0) {
            if (active) {
                if (lastTime !== null) elapsed = Math.min(4, elapsed + Math.min((now - lastTime) / 1000, .06));
                lastTime = now;
                if (elapsed === 4) { active = false; lastTime = null; }
            }
            // Scrolling during the entrance retracts only what has already
            // formed, so it cannot jump forward to a suddenly finished mesh.
            return seek((elapsed / 4) * (1 - THREE.MathUtils.clamp(reverseProgress, 0, 1)));
        },
        finish, seek,
        get active() { return active; },
        state: () => ({ active, elapsed, progress: sampledTime / 4, renders, plays, fromCoarse: plays > 0,
            levels: [144, 576, 2304, 9216], wireVisible: wire.visible }),
    };
}
