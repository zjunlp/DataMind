import React, { useEffect, useRef } from "react";
import { useGSAP } from "@gsap/react";
import { gsap } from "gsap";
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";

gsap.registerPlugin(useGSAP);

const COLORS = {
  blue: 0x2d7f96,
  cyan: 0x59a9b8,
  coral: 0xce6756,
  leaf: 0x6b9b62,
  yellow: 0xd3a64a,
  ink: 0x17211f,
  paper: 0xf3f6f5,
  white: 0xffffff,
  line: 0xc9d5d1,
};

const POSITIONS = {
  clean: [-2.85, 0.1, 0.65],
  filter: [-1.05, 0.3, 0.55],
  mean: [-2.25, 0.55, -0.75],
  median: [-0.45, 0.75, -0.7],
  newData: [0.6, 0.12, -1.55],
  agent: [0.55, 0.35, 0.15],
  initial: [1.55, 0.2, 1.35],
  composition: [2.85, 0.52, -0.35],
};

const TURNS = ["turn1", "turn2", "turn3", "turn4", "turn5"];

const NODE_LABELS = [
  { key: "clean", height: 0.62, offsetX: -8, offsetY: 3 },
  { key: "mean", height: 1.02, offsetX: -6, offsetY: -2 },
  { key: "filter", height: 1.1, offsetX: -10, offsetY: 4 },
  { key: "median", height: 1.2, offsetX: 2, offsetY: -2 },
  { key: "agent", height: 1.22, offsetX: 0, offsetY: 5 },
  { key: "initial", height: 1.12, offsetX: 6, offsetY: 5 },
  { key: "newData", height: 1.0, offsetX: 7, offsetY: 1 },
  { key: "composition", height: 1.2, offsetX: 5, offsetY: 5 },
];

const ACTIVE_TRACKS = [
  [3, 6],
  [0, 5, 6],
  [1, 4, 5],
  [4, 8],
  [2, 5, 6, 7, 9],
];

const ACTIVE_NODES = [
  ["clean", "mean", "agent"],
  ["clean", "filter", "mean", "agent"],
  ["filter", "median", "agent"],
  ["median", "initial", "agent"],
  ["mean", "filter", "newData", "agent", "composition"],
];

const ACTION_NODES = [
  ["clean", "mean"],
  ["filter"],
  ["median"],
  ["initial"],
  ["newData", "composition"],
];

function roundedBox(width, height, depth, radius, color, materialOptions = {}) {
  return new THREE.Mesh(
    new RoundedBoxGeometry(width, height, depth, 5, radius),
    new THREE.MeshStandardMaterial({ color, roughness: 0.62, ...materialOptions }),
  );
}

function createDatasetIcon(color) {
  const group = new THREE.Group();
  const page = roundedBox(0.34, 0.5, 0.08, 0.035, color, { roughness: 0.48 });
  page.position.y = 0.25;
  page.castShadow = true;
  group.add(page);

  const fold = roundedBox(0.12, 0.12, 0.035, 0.02, COLORS.white, { roughness: 0.5 });
  fold.position.set(0.1, 0.41, 0.055);
  group.add(fold);

  [0.17, 0.26, 0.35].forEach((y, index) => {
    const row = roundedBox(index === 1 ? 0.16 : 0.22, 0.025, 0.018, 0.008, COLORS.white, { roughness: 0.45 });
    row.position.set(-0.035, y, 0.052);
    group.add(row);
  });
  return group;
}

function createDataTable(position) {
  const group = new THREE.Group();
  group.position.set(...position);

  const base = roundedBox(1.48, 0.18, 0.9, 0.085, 0xdce9eb, { roughness: 0.72 });
  base.castShadow = true;
  base.receiveShadow = true;
  group.add(base);

  const sheet = roundedBox(1.22, 0.065, 0.68, 0.045, COLORS.white, { roughness: 0.58 });
  sheet.position.y = 0.13;
  sheet.castShadow = true;
  group.add(sheet);

  const cellColors = [COLORS.blue, COLORS.cyan, COLORS.leaf, COLORS.blue];
  const cells = [];
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      const cell = roundedBox(0.3, 0.035, 0.095, 0.012, cellColors[(row + column) % cellColors.length], { roughness: 0.42 });
      cell.position.set(-0.36 + column * 0.36, 0.19, -0.23 + row * 0.15);
      cell.userData.baseY = cell.position.y;
      cell.castShadow = true;
      group.add(cell);
      cells.push(cell);
    }
  }

  const sparkles = [];
  [[0.55, 0.33], [0.66, 0.22], [0.47, 0.19]].forEach(([x, y]) => {
    const sparkle = roundedBox(0.055, 0.055, 0.025, 0.012, COLORS.yellow, { emissive: COLORS.yellow, emissiveIntensity: 0.3 });
    sparkle.rotation.z = Math.PI / 4;
    sparkle.position.set(x, y, -0.29);
    group.add(sparkle);
    sparkles.push(sparkle);
  });

  const scrubber = new THREE.Group();
  const scrubberBody = roundedBox(0.2, 0.11, 0.15, 0.025, COLORS.cyan, { roughness: 0.4 });
  const scrubberPad = roundedBox(0.24, 0.035, 0.19, 0.018, COLORS.white, { roughness: 0.68 });
  scrubberPad.position.y = -0.07;
  scrubber.add(scrubberBody, scrubberPad);
  scrubber.position.set(-0.48, 0.34, -0.23);
  scrubber.userData.basePosition = scrubber.position.clone();
  group.add(scrubber);

  group.userData.animate = (active, time, blend) => {
    cells.forEach((cell, index) => {
      const lift = active ? Math.max(0, Math.sin(time * 6 - index * 0.34)) * 0.045 : 0;
      cell.position.y += (cell.userData.baseY + lift - cell.position.y) * blend;
      cell.material.emissive.copy(cell.material.color);
      cell.material.emissiveIntensity += ((active ? 0.14 : 0) - cell.material.emissiveIntensity) * blend;
    });
    sparkles.forEach((sparkle, index) => {
      const scale = active ? 0.8 + Math.max(0, Math.sin(time * 7 + index * 1.7)) * 0.55 : 0.82;
      sparkle.scale.setScalar(sparkle.scale.x + (scale - sparkle.scale.x) * blend);
      sparkle.rotation.z = active ? Math.PI / 4 + time * 1.8 : Math.PI / 4;
    });
    if (active) {
      const sweep = (time * 0.72) % 1;
      const row = Math.min(3, Math.floor(sweep * 4));
      const rowProgress = (sweep * 4) % 1;
      const direction = row % 2 === 0 ? rowProgress : 1 - rowProgress;
      scrubber.position.x += (-0.48 + direction * 0.96 - scrubber.position.x) * blend;
      scrubber.position.z += (-0.23 + row * 0.15 - scrubber.position.z) * blend;
      scrubber.position.y += (0.34 + Math.sin(time * 12) * 0.012 - scrubber.position.y) * blend;
      scrubber.rotation.y = Math.sin(time * 8) * 0.08;
    } else {
      scrubber.position.lerp(scrubber.userData.basePosition, blend);
      scrubber.rotation.y += (0 - scrubber.rotation.y) * blend;
    }
  };

  return group;
}

function createFilterStation(position) {
  const group = new THREE.Group();
  group.position.set(...position);

  const base = roundedBox(1.34, 0.18, 0.84, 0.085, 0xdcebed, { roughness: 0.7 });
  base.castShadow = true;
  base.receiveShadow = true;
  group.add(base);

  const funnel = new THREE.Mesh(
    new THREE.CylinderGeometry(0.34, 0.1, 0.52, 24, 1, true),
    new THREE.MeshStandardMaterial({ color: COLORS.cyan, roughness: 0.36, transparent: true, opacity: 0.86, side: THREE.DoubleSide }),
  );
  funnel.position.y = 0.53;
  funnel.castShadow = true;
  group.add(funnel);

  const rim = new THREE.Mesh(
    new THREE.TorusGeometry(0.34, 0.035, 12, 28),
    new THREE.MeshStandardMaterial({ color: COLORS.blue, roughness: 0.38 }),
  );
  rim.rotation.x = Math.PI / 2;
  rim.position.y = 0.79;
  group.add(rim);

  const stem = new THREE.Mesh(
    new THREE.CylinderGeometry(0.075, 0.075, 0.16, 18),
    new THREE.MeshStandardMaterial({ color: COLORS.blue, roughness: 0.4 }),
  );
  stem.position.y = 0.2;
  group.add(stem);

  const inputRows = [];
  [-0.2, 0, 0.2].forEach((x, index) => {
    const row = roundedBox(0.18, 0.04, 0.12, 0.012, [COLORS.coral, COLORS.cyan, COLORS.leaf][index]);
    row.position.set(x, 0.89 + Math.abs(index - 1) * 0.05, 0);
    row.userData.basePosition = row.position.clone();
    group.add(row);
    inputRows.push(row);
  });

  const selected = roundedBox(0.42, 0.055, 0.16, 0.018, COLORS.leaf, { emissive: COLORS.leaf, emissiveIntensity: 0.15 });
  selected.position.set(0, 0.14, 0.18);
  selected.userData.baseZ = selected.position.z;
  group.add(selected);

  group.userData.animate = (active, time, blend) => {
    inputRows.forEach((row, index) => {
      if (active) {
        const phase = (time * 0.52 + index * 0.29) % 1;
        const angle = time * 2.2 + index * (Math.PI * 2 / 3);
        row.position.x = Math.cos(angle) * (0.22 * (1 - phase * 0.55));
        row.position.z = Math.sin(angle) * (0.18 * (1 - phase * 0.55));
        row.position.y = 0.92 - phase * 0.55;
      } else {
        row.position.lerp(row.userData.basePosition, blend);
      }
    });
    const squeeze = active ? 1 + Math.sin(time * 5) * 0.045 : 1;
    funnel.scale.y += (squeeze - funnel.scale.y) * blend;
    const outputPhase = active ? (time * 0.7) % 1 : 0;
    const outputTravel = THREE.MathUtils.smoothstep(outputPhase, 0.18, 0.82);
    selected.position.z += (selected.userData.baseZ + (active ? outputTravel * 0.34 : 0) - selected.position.z) * blend;
    const selectedScale = active ? 0.86 + Math.sin(outputPhase * Math.PI) * 0.28 : 1;
    selected.scale.x += (selectedScale - selected.scale.x) * blend;
  };
  return group;
}

function createMeanStation(position) {
  const group = new THREE.Group();
  group.position.set(...position);

  const base = roundedBox(1.42, 0.18, 0.84, 0.085, 0xdce8eb, { roughness: 0.72 });
  base.castShadow = true;
  base.receiveShadow = true;
  group.add(base);

  const post = new THREE.Mesh(
    new THREE.CylinderGeometry(0.055, 0.09, 0.56, 18),
    new THREE.MeshStandardMaterial({ color: COLORS.ink, roughness: 0.46 }),
  );
  post.position.y = 0.39;
  post.castShadow = true;
  group.add(post);

  const beam = roundedBox(1.02, 0.07, 0.11, 0.025, COLORS.blue, { metalness: 0.12, roughness: 0.38 });
  beam.position.y = 0.69;
  beam.castShadow = true;
  group.add(beam);

  const pivot = new THREE.Mesh(
    new THREE.SphereGeometry(0.105, 20, 14),
    new THREE.MeshStandardMaterial({ color: COLORS.yellow, roughness: 0.4 }),
  );
  pivot.position.y = 0.69;
  group.add(pivot);

  const sideParts = [];
  [-0.38, 0.38].forEach((x) => {
    const cord = new THREE.Mesh(
      new THREE.CylinderGeometry(0.012, 0.012, 0.22, 10),
      new THREE.MeshStandardMaterial({ color: COLORS.ink, roughness: 0.5 }),
    );
    cord.position.set(x, 0.56, 0);
    cord.userData.baseY = cord.position.y;
    group.add(cord);
    sideParts.push({ object: cord, side: Math.sign(x) });

    const pan = roundedBox(0.34, 0.055, 0.28, 0.02, COLORS.white, { roughness: 0.52 });
    pan.position.set(x, 0.43, 0);
    pan.userData.baseY = pan.position.y;
    pan.castShadow = true;
    group.add(pan);
    sideParts.push({ object: pan, side: Math.sign(x) });

    [-0.07, 0.07].forEach((offset) => {
      const value = roundedBox(0.1, 0.1, 0.1, 0.018, COLORS.cyan, { roughness: 0.4 });
      value.position.set(x + offset, 0.51, 0);
      value.userData.baseY = value.position.y;
      group.add(value);
      sideParts.push({ object: value, side: Math.sign(x) });
    });
  });

  group.userData.animate = (active, time, blend) => {
    const cycle = (time % 1.65) / 1.65;
    const settle = Math.max(0, 1 - cycle / 0.72);
    const tilt = active ? Math.sin(cycle * Math.PI * 4.5) * 0.2 * settle : 0;
    beam.rotation.z += (tilt - beam.rotation.z) * blend;
    sideParts.forEach(({ object, side }) => {
      const targetY = object.userData.baseY + side * tilt * 0.5;
      object.position.y += (targetY - object.position.y) * blend;
    });
    const balancedPulse = active ? THREE.MathUtils.smoothstep(cycle, 0.64, 0.82) * Math.sin(cycle * Math.PI) : 0;
    const pivotScale = 1 + balancedPulse * 0.13;
    pivot.scale.setScalar(pivot.scale.x + (pivotScale - pivot.scale.x) * blend);
  };

  return group;
}

function createMedianStation(position) {
  const group = new THREE.Group();
  group.position.set(...position);

  const base = roundedBox(1.38, 0.18, 0.84, 0.085, 0xf0e1df, { roughness: 0.72 });
  base.castShadow = true;
  base.receiveShadow = true;
  group.add(base);

  const bars = [];
  [0.2, 0.31, 0.45, 0.58, 0.7].forEach((height, index) => {
    const isMiddle = index === 2;
    const bar = roundedBox(0.16, height, 0.3, 0.025, isMiddle ? COLORS.coral : 0x91bcc5, {
      emissive: isMiddle ? COLORS.coral : COLORS.blue,
      emissiveIntensity: isMiddle ? 0.22 : 0.03,
      roughness: 0.42,
    });
    bar.position.set(-0.44 + index * 0.22, 0.13 + height / 2, 0.02);
    bar.userData.baseY = bar.position.y;
    bar.userData.height = height;
    bar.castShadow = true;
    group.add(bar);
    bars.push(bar);
  });

  const marker = new THREE.Mesh(
    new THREE.TorusGeometry(0.14, 0.026, 12, 26),
    new THREE.MeshStandardMaterial({ color: COLORS.coral, roughness: 0.36 }),
  );
  marker.rotation.x = Math.PI / 2;
  marker.position.set(0, 0.63, 0.02);
  group.add(marker);

  group.userData.animate = (active, time, blend) => {
    bars.forEach((bar, index) => {
      const scan = active ? THREE.MathUtils.smoothstep(time - index * 0.1, 0, 0.45) : 0;
      const lift = active ? Math.sin(scan * Math.PI) * 0.13 : 0;
      const middleBoost = active && index === 2 ? 1 + THREE.MathUtils.smoothstep(time, 0.58, 0.92) * 0.13 : 1;
      bar.position.y += (bar.userData.baseY + lift - bar.position.y) * blend;
      bar.scale.y += (middleBoost - bar.scale.y) * blend;
    });
    const markerProgress = active ? THREE.MathUtils.smoothstep(time, 0.18, 0.95) : 1;
    marker.position.x += ((active ? -0.44 + markerProgress * 0.44 : 0) - marker.position.x) * blend;
    const markerRotation = active ? markerProgress * Math.PI * 2 : 0;
    marker.rotation.y += (markerRotation - marker.rotation.y) * blend;
    const markerScale = active ? 1 + THREE.MathUtils.smoothstep(time, 0.78, 1.08) * 0.18 * Math.max(0, Math.sin(time * 7)) : 1;
    marker.scale.setScalar(marker.scale.x + (markerScale - marker.scale.x) * blend);
  };

  return group;
}

function createNewDataStack(position) {
  const group = new THREE.Group();
  group.position.set(...position);

  const base = roundedBox(1.34, 0.18, 0.84, 0.085, 0xe1ecdf, { roughness: 0.72 });
  base.castShadow = true;
  base.receiveShadow = true;
  group.add(base);

  const pages = [];
  [
    [-0.28, 0.12, 0.08, COLORS.cyan],
    [0, 0.17, 0, COLORS.yellow],
    [0.28, 0.22, -0.08, COLORS.leaf],
  ].forEach(([x, y, z, color], index) => {
    const dataset = createDatasetIcon(color);
    dataset.position.set(x, y, z);
    dataset.rotation.y = (index - 1) * -0.16;
    dataset.userData.baseY = dataset.position.y;
    dataset.userData.baseRotationY = dataset.rotation.y;
    group.add(dataset);
    pages.push(dataset);
  });

  group.userData.animate = (active, time, blend) => {
    pages.forEach((page, index) => {
      const fan = active ? (index - 1) * 0.34 : page.userData.baseRotationY;
      const lift = active ? Math.sin(time * 4.5 + index * 1.3) * 0.065 : 0;
      page.rotation.y += (fan - page.rotation.y) * blend;
      page.position.y += (page.userData.baseY + lift - page.position.y) * blend;
    });
  };

  return group;
}

function createStateRouter(position) {
  const group = new THREE.Group();
  group.position.set(...position);

  const pedestal = roundedBox(1.02, 0.2, 0.82, 0.1, new THREE.Color(COLORS.ink).lerp(new THREE.Color(COLORS.paper), 0.78));
  pedestal.castShadow = true;
  pedestal.receiveShadow = true;
  group.add(pedestal);

  const consoleBody = roundedBox(0.76, 0.34, 0.62, 0.1, COLORS.white, { roughness: 0.5 });
  consoleBody.position.y = 0.36;
  consoleBody.castShadow = true;
  group.add(consoleBody);

  const selectorRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.27, 0.035, 12, 36),
    new THREE.MeshStandardMaterial({ color: COLORS.blue, roughness: 0.38 }),
  );
  selectorRing.rotation.x = Math.PI / 2;
  selectorRing.position.y = 0.58;
  group.add(selectorRing);

  const selector = new THREE.Group();
  selector.position.y = 0.6;
  const pointer = roundedBox(0.075, 0.055, 0.28, 0.018, COLORS.coral, {
    emissive: COLORS.coral,
    emissiveIntensity: 0.28,
    roughness: 0.36,
  });
  pointer.position.z = 0.12;
  selector.add(pointer);
  const pivot = new THREE.Mesh(
    new THREE.CylinderGeometry(0.075, 0.075, 0.08, 18),
    new THREE.MeshStandardMaterial({ color: COLORS.ink, roughness: 0.42 }),
  );
  selector.add(pivot);
  group.add(selector);

  const stateStops = [];
  [COLORS.blue, COLORS.cyan, COLORS.coral, COLORS.yellow, COLORS.leaf].forEach((color, index) => {
    const angle = -Math.PI * 0.62 + index * (Math.PI * 1.24 / 4);
    const stop = new THREE.Mesh(
      new THREE.CylinderGeometry(0.045, 0.045, 0.055, 14),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.04, roughness: 0.4 }),
    );
    stop.position.set(Math.sin(angle) * 0.27, 0.61, Math.cos(angle) * 0.27);
    group.add(stop);
    stateStops.push({ stop, angle });
  });

  const inputPorts = [];
  [-0.22, 0, 0.22].forEach((x, index) => {
    const color = [COLORS.blue, COLORS.cyan, COLORS.leaf][index];
    const port = roundedBox(0.14, 0.055, 0.045, 0.015, color, {
      emissive: color,
      emissiveIntensity: 0.04,
      roughness: 0.4,
    });
    port.position.set(x, 0.3, 0.335);
    group.add(port);
    inputPorts.push(port);
  });

  const outputPort = roundedBox(0.18, 0.07, 0.05, 0.018, COLORS.yellow, {
    emissive: COLORS.yellow,
    emissiveIntensity: 0.08,
    roughness: 0.4,
  });
  outputPort.position.set(0.405, 0.34, 0);
  outputPort.rotation.y = Math.PI / 2;
  group.add(outputPort);

  group.userData.animate = (active, time, blend, stepIndex = 0) => {
    const targetAngle = stateStops[stepIndex]?.angle ?? 0;
    selector.rotation.y += (targetAngle - selector.rotation.y) * blend;
    selectorRing.material.emissive.copy(selectorRing.material.color);
    selectorRing.material.emissiveIntensity += ((active ? 0.2 : 0.03) - selectorRing.material.emissiveIntensity) * blend;
    stateStops.forEach(({ stop }, index) => {
      const selected = index === stepIndex;
      const targetScale = selected && active ? 1.35 + Math.sin(time * 5) * 0.08 : 1;
      stop.scale.setScalar(stop.scale.x + (targetScale - stop.scale.x) * blend);
      stop.material.emissiveIntensity += (((selected && active) ? 0.8 : 0.04) - stop.material.emissiveIntensity) * blend;
    });
    inputPorts.forEach((port, index) => {
      const glow = active ? 0.18 + Math.max(0, Math.sin(time * 6 - index * 0.9)) * 0.55 : 0.04;
      port.material.emissiveIntensity += (glow - port.material.emissiveIntensity) * blend;
    });
    const outputGlow = active && stepIndex === TURNS.length - 1 ? 0.8 + Math.sin(time * 7) * 0.16 : 0.08;
    outputPort.material.emissiveIntensity += (outputGlow - outputPort.material.emissiveIntensity) * blend;
  };

  return group;
}

function createSnapshotDock(position) {
  const group = new THREE.Group();
  group.position.set(...position);

  const base = roundedBox(1.34, 0.18, 0.84, 0.09, 0xf4e8c7, { roughness: 0.7 });
  base.castShadow = true;
  base.receiveShadow = true;
  group.add(base);

  const cabinet = roundedBox(1.02, 0.48, 0.62, 0.065, COLORS.white, { roughness: 0.58 });
  cabinet.position.set(-0.08, 0.32, -0.05);
  cabinet.castShadow = true;
  group.add(cabinet);

  const drawer = roundedBox(0.78, 0.17, 0.5, 0.04, COLORS.yellow, { roughness: 0.5 });
  drawer.position.set(-0.08, 0.23, 0.25);
  drawer.userData.baseZ = drawer.position.z;
  drawer.castShadow = true;
  group.add(drawer);

  const handle = roundedBox(0.24, 0.04, 0.035, 0.012, COLORS.ink, { roughness: 0.42 });
  handle.position.set(-0.08, 0.25, 0.52);
  handle.userData.baseZ = handle.position.z;
  group.add(handle);

  const snapshot = roundedBox(0.58, 0.035, 0.34, 0.025, COLORS.white, { roughness: 0.48 });
  snapshot.position.set(-0.08, 0.35, 0.27);
  snapshot.userData.baseY = snapshot.position.y;
  snapshot.userData.baseZ = snapshot.position.z;
  group.add(snapshot);

  const clock = new THREE.Mesh(
    new THREE.TorusGeometry(0.18, 0.035, 12, 28),
    new THREE.MeshStandardMaterial({ color: COLORS.ink, roughness: 0.4 }),
  );
  clock.position.set(0.38, 0.68, 0.08);
  group.add(clock);

  const hourHand = roundedBox(0.035, 0.13, 0.026, 0.01, COLORS.coral, { roughness: 0.38 });
  hourHand.position.set(0.38, 0.72, 0.105);
  group.add(hourHand);

  const minuteHand = roundedBox(0.13, 0.035, 0.026, 0.01, COLORS.coral, { roughness: 0.38 });
  minuteHand.position.set(0.43, 0.68, 0.105);
  group.add(minuteHand);

  group.userData.animate = (active, time, blend) => {
    const open = active ? 0.22 + Math.sin(time * 4) * 0.025 : 0;
    drawer.position.z += (drawer.userData.baseZ + open - drawer.position.z) * blend;
    handle.position.z += (handle.userData.baseZ + open - handle.position.z) * blend;
    snapshot.position.z += (snapshot.userData.baseZ + open * 0.92 - snapshot.position.z) * blend;
    snapshot.position.y += (snapshot.userData.baseY + (active ? 0.07 : 0) - snapshot.position.y) * blend;
    const hourRotation = active ? -time * 0.7 : 0;
    const minuteRotation = active ? -time * 2.4 : 0;
    hourHand.rotation.z += (hourRotation - hourHand.rotation.z) * blend;
    minuteHand.rotation.z += (minuteRotation - minuteHand.rotation.z) * blend;
    const clockScale = active ? 1 + Math.sin(time * 5) * 0.06 : 1;
    clock.scale.setScalar(clock.scale.x + (clockScale - clock.scale.x) * blend);
  };
  return group;
}

function createOutput(position) {
  const group = new THREE.Group();
  group.position.set(...position);

  const base = roundedBox(1.35, 0.24, 0.92, 0.11, new THREE.Color(COLORS.ink).lerp(new THREE.Color(COLORS.paper), 0.72));
  base.castShadow = true;
  base.receiveShadow = true;
  group.add(base);

  const layers = [];
  [COLORS.blue, COLORS.coral, COLORS.leaf].forEach((color, index) => {
    const layer = roundedBox(0.42, 0.08, 0.48, 0.025, color, { roughness: 0.46 });
    layer.position.set(-0.38, 0.18 + index * 0.1, 0.04);
    layer.userData.baseX = layer.position.x;
    layer.castShadow = true;
    group.add(layer);
    layers.push(layer);
  });

  const report = roundedBox(0.7, 0.07, 0.62, 0.035, COLORS.white, { roughness: 0.5 });
  report.position.set(0.23, 0.21, 0.02);
  report.userData.baseY = report.position.y;
  report.castShadow = true;
  group.add(report);

  const result = new THREE.Group();
  const resultBars = [];
  [0.16, 0.26, 0.38, 0.3].forEach((height, index) => {
    const bar = roundedBox(0.1, height, 0.13, 0.018, [COLORS.blue, COLORS.coral, COLORS.leaf, COLORS.yellow][index], {
      emissive: index === 2 ? COLORS.leaf : COLORS.ink,
      emissiveIntensity: index === 2 ? 0.18 : 0,
      roughness: 0.42,
    });
    bar.position.set(-0.19 + index * 0.14, 0.27 + height / 2, 0.02);
    bar.userData.baseY = bar.position.y;
    bar.userData.height = height;
    bar.castShadow = true;
    result.add(bar);
    resultBars.push(bar);
  });
  result.position.x = 0.23;
  result.userData.resultBeacon = true;
  group.add(result);

  group.userData.animate = (active, time, blend) => {
    layers.forEach((layer, index) => {
      const progress = active ? THREE.MathUtils.smoothstep(time - index * 0.12, 0, 0.65) : 0;
      layer.position.x += (layer.userData.baseX + progress * 0.24 - layer.position.x) * blend;
    });
    resultBars.forEach((bar, index) => {
      const rise = active ? Math.max(0.12, THREE.MathUtils.smoothstep(time - 0.25 - index * 0.1, 0, 0.55)) : 1;
      bar.scale.y += (rise - bar.scale.y) * blend;
      const base = bar.userData.baseY - bar.userData.height / 2;
      bar.position.y = base + bar.userData.height * bar.scale.y / 2;
    });
    const reportLift = active ? Math.sin(Math.min(time, 1) * Math.PI) * 0.045 : 0;
    report.position.y += (report.userData.baseY + reportLift - report.position.y) * blend;
  };
  return group;
}

function createDataPacket(color) {
  const group = new THREE.Group();
  const materials = [];
  const options = { transparent: true, opacity: 0, depthTest: false, depthWrite: false, roughness: 0.38 };

  const sheet = roundedBox(0.25, 0.055, 0.18, 0.025, color, options);
  group.add(sheet);
  materials.push(sheet.material);

  [-0.045, 0.015, 0.06].forEach((z, index) => {
    const row = roundedBox(index === 1 ? 0.12 : 0.17, 0.018, 0.022, 0.007, COLORS.white, options);
    row.position.set(-0.015, 0.045, z);
    group.add(row);
    materials.push(row.material);
  });

  group.traverse((object) => { object.renderOrder = 8; });
  group.userData.materials = materials;
  return group;
}

function createTrack(points, color, radius = 0.038, opacity = 1) {
  const curve = new THREE.CatmullRomCurve3(
    points.map((point) => new THREE.Vector3(...point)),
    false,
    "centripetal",
  );
  const group = new THREE.Group();
  const underlay = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 80, radius + 0.025, 10, false),
    new THREE.MeshStandardMaterial({ color: COLORS.paper, roughness: 0.76, transparent: true, opacity }),
  );
  const rail = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 80, radius, 10, false),
    new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.03,
      roughness: 0.4,
      metalness: 0.1,
      transparent: true,
      opacity: 0.14,
    }),
  );
  const highlight = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 80, radius + 0.055, 10, false),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    }),
  );
  underlay.castShadow = true;
  rail.castShadow = true;
  group.add(underlay, highlight, rail);
  return { curve, group, rail, underlay, highlight, activeOpacity: opacity };
}

export default function StateAtlas({ content, label }) {
  const figureRef = useRef(null);
  const mountRef = useRef(null);
  const labelRefs = useRef([]);
  const nodeLabelRefs = useRef({});
  const activeStepRef = useRef(0);
  const stepStartedAtRef = useRef(0);
  const reduceMotionRef = useRef(false);

  useGSAP(() => {
    const labels = labelRefs.current.filter(Boolean);
    const nodeLabels = nodeLabelRefs.current;
    const media = gsap.matchMedia();
    gsap.set(labels, { clearProps: "transform" });

    media.add(
      {
        all: "all",
        reduceMotion: "(prefers-reduced-motion: reduce)",
      },
      (context) => {
        const { reduceMotion } = context.conditions;
        const activateStep = (index) => {
          activeStepRef.current = index;
          stepStartedAtRef.current = performance.now() / 1000;
          const activeNodes = ACTIVE_NODES[index] ?? [];
          const actionNodes = ACTION_NODES[index] ?? [];
          NODE_LABELS.forEach(({ key }) => {
            const element = nodeLabels[key];
            if (!element) return;
            element.dataset.state = actionNodes.includes(key)
              ? "action"
              : activeNodes.includes(key) ? "context" : "inactive";
          });
        };
        reduceMotionRef.current = reduceMotion;
        gsap.set(labels, { autoAlpha: 0, clipPath: "inset(0 0% 0 0)" });
        if (reduceMotion) {
          activateStep(labels.length - 1);
          gsap.set(labels.at(-1), { autoAlpha: 1 });
          return undefined;
        }

        activateStep(0);
        const timeline = gsap.timeline({
          repeat: -1,
          repeatDelay: 0.25,
          defaults: { ease: "power2.out" },
        });
        timeline.to({}, { duration: 0.2 });

        labels.forEach((element, index) => {
          const step = `state-${index}`;
          const hold = index === labels.length - 1 ? 2.8 : 2.2;
          timeline
            .addLabel(step)
            .call(() => { activateStep(index); }, [], step)
            .fromTo(
              element,
              { autoAlpha: 0, clipPath: "inset(0 100% 0 0)" },
              { autoAlpha: 1, clipPath: "inset(0 0% 0 0)", duration: 0.4, immediateRender: false },
              step,
            )
            .to(element, { autoAlpha: 1, duration: hold, ease: "none" })
            .to(element, { autoAlpha: 0, duration: 0.28, ease: "power1.in" });
        });

        return () => timeline.kill();
      },
    );

    return () => media.revert();
  }, { scope: figureRef, dependencies: [content], revertOnUpdate: true });

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;

    let renderer;
    let scene;
    let camera;
    let world;
    let timer;
    let previousElapsed;
    let tokens = [];
    let tracks = [];
    let labelAnchors = {};
    let sceneNodes = {};
    let resultBeacon;

    const resize = () => {
      if (!renderer || !camera) return;
      const width = mount.clientWidth;
      const height = mount.clientHeight;
      if (!width || !height) return;
      const aspect = width / height;
      const viewWidth = Math.max(8.7, aspect * 4.8);
      const viewHeight = viewWidth / aspect;
      camera.left = -viewWidth / 2;
      camera.right = viewWidth / 2;
      camera.top = viewHeight / 2;
      camera.bottom = -viewHeight / 2;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      renderer.render(scene, camera);
    };

    const initialize = () => {
      if (renderer || mount.clientWidth < 2 || mount.clientHeight < 2) return;

      scene = new THREE.Scene();
      camera = new THREE.OrthographicCamera(-5, 5, 3.5, -3.5, 0.1, 60);
      camera.position.set(7.7, 7.5, 9.4);
      camera.lookAt(0, 0.35, 0);

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
      renderer.setClearColor(0x000000, 0);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFShadowMap;
      renderer.domElement.setAttribute("aria-hidden", "true");
      mount.appendChild(renderer.domElement);

      const hemisphere = new THREE.HemisphereLight(0xffffff, 0xc4d1cd, 2.5);
      const keyLight = new THREE.DirectionalLight(0xffffff, 4.4);
      keyLight.position.set(-3, 8, 5);
      keyLight.castShadow = true;
      keyLight.shadow.mapSize.set(1024, 1024);
      keyLight.shadow.camera.left = -8;
      keyLight.shadow.camera.right = 8;
      keyLight.shadow.camera.top = 7;
      keyLight.shadow.camera.bottom = -7;
      scene.add(hemisphere, keyLight);

      world = new THREE.Group();
      world.rotation.y = -0.05;
      world.position.y = 0.02;
      scene.add(world);

      const grid = new THREE.GridHelper(10.5, 16, COLORS.line, COLORS.line);
      grid.position.y = -0.43;
      grid.material.transparent = true;
      grid.material.opacity = 0.28;
      world.add(grid);

      const shadowPlane = new THREE.Mesh(
        new THREE.PlaneGeometry(11, 7.5),
        new THREE.ShadowMaterial({ color: 0x6f7f7a, opacity: 0.11 }),
      );
      shadowPlane.rotation.x = -Math.PI / 2;
      shadowPlane.position.y = -0.44;
      shadowPlane.receiveShadow = true;
      world.add(shadowPlane);

      labelAnchors = {
        clean: createDataTable(POSITIONS.clean),
        filter: createFilterStation(POSITIONS.filter),
        mean: createMeanStation(POSITIONS.mean),
        median: createMedianStation(POSITIONS.median),
        newData: createNewDataStack(POSITIONS.newData),
        initial: createSnapshotDock(POSITIONS.initial),
        composition: createOutput(POSITIONS.composition),
      };

      Object.values(labelAnchors).forEach((object) => world.add(object));
      const agent = createStateRouter(POSITIONS.agent);
      world.add(agent);
      sceneNodes = { ...labelAnchors, agent };
      Object.values(sceneNodes).forEach((object) => { object.userData.baseWorldY = object.position.y; });
      resultBeacon = labelAnchors.composition.children.find((child) => child.userData.resultBeacon);

      tracks = [
        createTrack([
          [-2.85, 0.48, 0.65],
          [-2.0, 0.7, 0.75],
          [-1.05, 0.68, 0.55],
        ], COLORS.cyan),
        createTrack([
          [-2.25, 0.92, -0.75],
          [-1.35, 1.1, -0.8],
          [-0.45, 1.12, -0.7],
        ], COLORS.coral),
        createTrack([
          [-0.45, 1.12, -0.7],
          [-1.2, 1.62, -1.28],
          [-2.25, 1.0, -0.75],
        ], COLORS.leaf, 0.044),
        createTrack([
          [-2.85, 0.48, 0.65],
          [-1.45, 1.55, 1.5],
          [-0.15, 1.35, 0.95],
          [0.55, 0.9, 0.15],
        ], COLORS.yellow, 0.034, 0.82),
        createTrack([
          [-0.45, 1.12, -0.7],
          [0.12, 1.36, -0.55],
          [0.55, 0.9, 0.15],
        ], COLORS.coral, 0.034, 0.82),
        createTrack([
          [-1.05, 0.68, 0.55],
          [-0.2, 1.05, 0.45],
          [0.55, 0.9, 0.15],
        ], COLORS.cyan, 0.036, 0.9),
        createTrack([
          [-2.25, 0.95, -0.75],
          [-0.7, 1.4, -0.25],
          [0.55, 0.9, 0.15],
        ], COLORS.blue, 0.036, 0.9),
        createTrack([
          [0.6, 0.5, -1.55],
          [0.65, 1.0, -0.7],
          [0.55, 0.9, 0.15],
        ], COLORS.leaf, 0.036, 0.9),
        createTrack([
          [1.55, 0.58, 1.35],
          [1.3, 1.05, 0.78],
          [0.55, 0.9, 0.15],
        ], COLORS.yellow, 0.04),
        createTrack([
          [0.55, 0.9, 0.15],
          [1.55, 1.22, -0.05],
          [2.85, 0.95, -0.35],
        ], COLORS.yellow, 0.046),
      ];
      tracks.forEach(({ group }) => world.add(group));

      const tokenForward = new THREE.Vector3(0, 0, 1);
      const tokenColors = [COLORS.cyan, COLORS.coral, COLORS.leaf, COLORS.yellow, COLORS.coral, COLORS.cyan, COLORS.blue, COLORS.leaf, COLORS.yellow, COLORS.yellow];
      tokens = tracks.map(({ curve }, trackIndex) => {
        const mesh = createDataPacket(tokenColors[trackIndex]);
        world.add(mesh);
        return {
          curve,
          mesh,
          materials: mesh.userData.materials,
          trackIndex,
          tangent: new THREE.Vector3(),
          speed: 0.027 + trackIndex * 0.0025,
          offset: (trackIndex * 0.23) % 1,
        };
      });

      timer = new THREE.Timer();
      timer.connect(document);
      resize();

      const renderFrame = (elapsed) => {
        const delta = previousElapsed === undefined ? 1 : Math.min(0.1, Math.max(0, elapsed - previousElapsed));
        const blend = previousElapsed === undefined ? 1 : 1 - Math.exp(-10 * delta);
        previousElapsed = elapsed;
        const activeTrackIndexes = ACTIVE_TRACKS[activeStepRef.current] ?? [];
        const activeNodeKeys = ACTIVE_NODES[activeStepRef.current] ?? [];
        const actionNodeKeys = ACTION_NODES[activeStepRef.current] ?? [];
        const stepElapsed = Math.max(0, performance.now() / 1000 - stepStartedAtRef.current);
        const actionElapsed = Math.max(0, stepElapsed - 0.5);

        tracks.forEach((track, index) => {
          const isActive = activeTrackIndexes.includes(index);
          const railOpacity = isActive ? track.activeOpacity : 0.045;
          const underlayOpacity = isActive ? 0.34 : 0.025;
          track.rail.material.opacity += (railOpacity - track.rail.material.opacity) * blend;
          track.rail.material.emissiveIntensity += ((isActive ? 0.78 : 0.02) - track.rail.material.emissiveIntensity) * blend;
          track.underlay.material.opacity += (underlayOpacity - track.underlay.material.opacity) * blend;
          track.highlight.material.opacity += ((isActive ? 0.2 : 0) - track.highlight.material.opacity) * blend;
        });

        Object.entries(sceneNodes).forEach(([key, object]) => {
          const isContextNode = activeNodeKeys.includes(key);
          const isActionNode = actionNodeKeys.includes(key);
          const targetScale = isActionNode ? 1.12 : isContextNode ? 1.01 : 0.94;
          const scale = object.scale.x + (targetScale - object.scale.x) * blend;
          object.scale.setScalar(scale);
          const targetY = object.userData.baseWorldY + (isActionNode ? 0.08 : 0);
          object.position.y += (targetY - object.position.y) * blend;
          const shouldAnimate = key === "agent"
            ? isContextNode
            : isActionNode && stepElapsed >= 0.5;
          object.userData.animate?.(!reduceMotionRef.current && shouldAnimate, actionElapsed, blend, activeStepRef.current);
        });

        NODE_LABELS.forEach(({ key, height, offsetX, offsetY }) => {
          const element = nodeLabelRefs.current[key];
          const object = sceneNodes[key];
          if (!element || !object) return;
          const anchor = object.localToWorld(new THREE.Vector3(0, height, 0));
          anchor.project(camera);
          element.style.left = `${mount.offsetLeft + (anchor.x * 0.5 + 0.5) * mount.clientWidth + offsetX}px`;
          element.style.top = `${mount.offsetTop + (-anchor.y * 0.5 + 0.5) * mount.clientHeight + offsetY}px`;
        });

        tokens.forEach((token) => {
          const activeOrder = activeTrackIndexes.indexOf(token.trackIndex);
          const travel = THREE.MathUtils.clamp((stepElapsed - Math.max(0, activeOrder) * 0.08) / 0.72, 0, 1);
          const progress = activeOrder >= 0 ? travel : 0;
          token.mesh.position.copy(token.curve.getPointAt(progress));
          token.curve.getTangentAt(progress, token.tangent).normalize();
          token.mesh.quaternion.setFromUnitVectors(tokenForward, token.tangent);
          token.mesh.position.y += 0.052 + Math.sin(stepElapsed * 8 + token.offset * 10) * 0.02;
          const isActive = activeOrder >= 0;
          const arrivalFade = 1 - THREE.MathUtils.smoothstep(travel, 0.88, 1);
          const targetOpacity = isActive ? arrivalFade : 0;
          token.materials.forEach((material) => {
            material.opacity += (targetOpacity - material.opacity) * blend;
          });
          const tokenScale = token.mesh.scale.x + ((isActive ? 1.25 : 0.7) - token.mesh.scale.x) * blend;
          token.mesh.scale.setScalar(tokenScale);
        });

        if (resultBeacon) {
          const pulse = 1 + Math.sin(elapsed * 2.3) * 0.04;
          resultBeacon.scale.set(pulse, pulse, pulse);
        }

        renderer.render(scene, camera);
      };

      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        renderFrame(0);
      } else {
        renderer.setAnimationLoop(() => {
          timer.update();
          renderFrame(timer.getElapsed());
        });
      }
    };

    const resizeObserver = new ResizeObserver(() => {
      initialize();
      resize();
    });
    resizeObserver.observe(mount);
    initialize();

    return () => {
      resizeObserver.disconnect();
      timer?.dispose();
      if (renderer) {
        renderer.setAnimationLoop(null);
        scene.traverse((object) => {
          object.geometry?.dispose();
          if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose());
          else object.material?.dispose();
        });
        renderer.dispose();
        renderer.forceContextLoss();
        renderer.domElement.remove();
      }
    };
  }, []);

  return (
    <figure className="state-atlas" aria-label={label} ref={figureRef}>
      <div className="state-atlas-canvas" ref={mountRef} />
      <strong className="atlas-heading">{content.heading}</strong>
      {TURNS.map((key, index) => (
        <span
          className="atlas-label atlas-label-request"
          key={key}
          ref={(element) => { labelRefs.current[index] = element; }}
        >
          {content[key]}
        </span>
      ))}
      {NODE_LABELS.map(({ key }) => (
        <span
          className="atlas-node-label"
          data-node={key}
          data-state="inactive"
          key={key}
          ref={(element) => { nodeLabelRefs.current[key] = element; }}
        >
          {content.nodes[key]}
        </span>
      ))}
    </figure>
  );
}
