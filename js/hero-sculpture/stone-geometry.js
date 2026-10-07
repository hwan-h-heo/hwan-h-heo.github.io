import * as THREE from 'three';

// An original uneven annular body: swollen shoulders, a pinched neck, and a
// recessed opening. The closed 192 x 24 surface retains the refinement lattice.
export function carvedGeometry() {
    const length = 192, radial = 24, positions = [], uvs = [], indices = [];
    for (let i = 0; i <= length; i++) {
        const a = i / length * Math.PI * 2;
        const radius = 1 + .15 * Math.sin(3 * a + .6) + .07 * Math.cos(2 * a - .8);
        const thickness = .29 + .10 * Math.sin(a - .65) + .09 * Math.cos(2 * a + .3);
        for (let j = 0; j <= radial; j++) {
            const b = j / radial * Math.PI * 2;
            const erosion = .014 * Math.sin(11 * a + 3 * b) * Math.sin(7 * a - 2 * b)
                + .008 * Math.cos(19 * a + 5 * b);
            const c = Math.sign(Math.cos(b)) * Math.pow(Math.abs(Math.cos(b)), .78);
            const s = Math.sign(Math.sin(b)) * Math.pow(Math.abs(Math.sin(b)), .78);
            const r = radius + c * (thickness + erosion);
            positions.push(Math.cos(a) * r * .94 + .10 * Math.sin(2 * a),
                Math.sin(a) * r * 1.30,
                s * (thickness * 1.40 + .07) + .16 * Math.cos(2 * a + .4));
            uvs.push(i / length, j / radial);
        }
    }
    for (let i = 0; i <= length; i++) for (let k = 0; k < 3; k++) positions[(i * (radial + 1) + radial) * 3 + k] = positions[i * (radial + 1) * 3 + k];
    for (let j = 0; j <= radial; j++) for (let k = 0; k < 3; k++) positions[(length * (radial + 1) + j) * 3 + k] = positions[j * 3 + k];
    for (let i = 0; i < length; i++) for (let j = 0; j < radial; j++) {
        const a = i * (radial + 1) + j, b = a + radial + 1;
        indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices); geometry.computeVertexNormals();
    const normal = geometry.attributes.normal, n = new THREE.Vector3(), other = new THREE.Vector3();
    const join = (a, b) => { n.fromBufferAttribute(normal, a); other.fromBufferAttribute(normal, b); n.add(other).normalize(); normal.setXYZ(a, n.x, n.y, n.z); normal.setXYZ(b, n.x, n.y, n.z); };
    for (let i = 0; i <= length; i++) join(i * (radial + 1), i * (radial + 1) + radial);
    for (let j = 0; j <= radial; j++) join(j, length * (radial + 1) + j);
    geometry.computeBoundingBox(); geometry.translate(0, -geometry.boundingBox.min.y, 0);
    return geometry;
}

export function apertureGeometry() {
    // A deep flared opening, not a flat extruded ring. The changing wall normal
    // carries a broad light gradient from the lip into the surrounding stone.
    const profile = new THREE.CatmullRomCurve3([
        [3.02, -.85], [3.02, -.15], [3.13, .52], [3.42, .84],
        [3.88, .88], [4.48, .55], [5.15, -.05], [5.62, -.58],
        [5.69, -1.10], [5.30, -1.40], [4.05, -1.43], [3.16, -1.30],
    ].map(([radius, depth]) => new THREE.Vector3(radius, depth, 0)), true, 'centripetal');
    const points = profile.getPoints(48).map(p => new THREE.Vector2(p.x, p.y));
    const geometry = new THREE.LatheGeometry(points, 160);
    geometry.rotateX(Math.PI / 2);
    return geometry;
}

export function stoneBlock(width, height, depth) {
    const shape = new THREE.Shape();
    shape.moveTo(-width / 2, .025); shape.lineTo(width / 2, .025);
    shape.lineTo(width / 2, height - .025); shape.lineTo(-width / 2, height - .025); shape.closePath();
    const geometry = new THREE.ExtrudeGeometry(shape, { depth: depth - .05, bevelEnabled: true,
        bevelSegments: 2, steps: 1, bevelSize: .025, bevelThickness: .025, curveSegments: 1 });
    geometry.translate(0, 0, -depth / 2 + .025);
    return geometry;
}
