import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

/* ─── State ─── */
let scene, camera, renderer, controls;
let modelGroup, pcGroup, flashLight, flashTarget, trajLine;
const timer = new THREE.Timer();
let pointCloudVisible = false;
let totalPoints = 0, totalTris = 0;
let metadata = null;

// Robot dog replay
let robotGroup = null;
let robotPath = null;
let robotPlaying = false;
let robotTime = 0;
const robotSpeed = 1.0;

const move = { fwd: false, back: false, left: false, right: false, sprint: false };
const WALK = 3.5, SPRINT = 10.0;
const dir = new THREE.Vector3();

// FPS tracking
let frameCount = 0, lastTime = 0;
const $ = id => document.getElementById(id);
const setProgress = (p, m) => { $('load-bar').style.width = p + '%'; if (m) $('load-status').textContent = m; };

/* ─── Main ─── */
async function main() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080810);
  scene.fog = new THREE.FogExp2(0x080810, 0.028);

  camera = new THREE.PerspectiveCamera(85, innerWidth / innerHeight, 0.05, 120);
  camera.position.set(1.5, 1.65, 4.0);

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: true,
  });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  document.body.appendChild(renderer.domElement);

  controls = new PointerLockControls(camera, document.body);

  // Lighting
  scene.add(new THREE.AmbientLight(0x334466, 0.35));
  scene.add(new THREE.HemisphereLight(0x8899bb, 0x223344, 0.25));

  flashTarget = new THREE.Object3D();
  scene.add(flashTarget);
  flashLight = new THREE.SpotLight(0xffe8cc, 4, 25, Math.PI * 0.35, 0.3, 1.2);
  scene.add(flashLight);

  modelGroup = new THREE.Group();
  scene.add(modelGroup);
  pcGroup = new THREE.Group();
  pcGroup.visible = false;
  scene.add(pcGroup);

  try {
    setProgress(5, 'Downloading assets…');

    const [metaRes, trajRes, robotRes] = await Promise.all([
      fetch('assets/metadata.json').then(r => r.json()),
      fetch('assets/trajectory.json').then(r => r.json()),
      fetch('assets/robot_path.json').then(r => r.json()),
    ]);
    metadata = metaRes;
    robotPath = robotRes;

    setProgress(40, 'Building scene…');
    buildTrajectory(trajRes);
    buildRobotDog();

    setProgress(60, 'Loading 3D model…');
    await loadModel();

    if (metadata?.robotStart) {
      const [x, y] = metadata.robotStart;
      camera.position.set(x, 1.65, -y);
    }

    setProgress(100, '✅ Ready — click to enter');
    await new Promise(r => setTimeout(r, 400));
    $('loading').style.display = 'none';
    $('blocker').style.display = 'flex';
    $('blocker').classList.remove('hidden');

  } catch (e) {
    console.error('Loading failed:', e);
    $('load-status').textContent = `❌ ${e.message || e}`;
    $('load-status').style.color = '#ff6644';
    $('load-bar').style.background = '#ff4444';
  }

  setupEvents();
  initMinimap();
  updateHUD();
  tick();
}

/* ─── Trajectory ─── */
function buildTrajectory(data) {
  const positions = new Float32Array(data.length * 3);
  for (let i = 0; i < data.length; i++) {
    const [x, y, z] = data[i];
    positions[i * 3] = x;
    positions[i * 3 + 1] = z + 0.05;
    positions[i * 3 + 2] = -y;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  trajLine = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x4488ff, transparent: true, opacity: 0.6 }));
  scene.add(trajLine);
}

/* ─── Point Cloud (lazy) ─── */
let pcLoaded = false;

function buildPointCloud(buffer) {
  const view = new DataView(buffer);
  const minX = view.getFloat32(0, true);
  const minY = view.getFloat32(4, true);
  const minZ = view.getFloat32(8, true);
  const scale = view.getFloat32(12, true);
  const n = view.getUint32(16, true);
  totalPoints = n;

  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {
    const off = 20 + i * 9;
    const x = minX + view.getInt16(off, true) * scale;
    const y = minY + view.getInt16(off + 2, true) * scale;
    const z = minZ + view.getInt16(off + 4, true) * scale;
    pos[i * 3] = x; pos[i * 3 + 1] = z; pos[i * 3 + 2] = -y;
    col[i * 3] = view.getUint8(off + 6) / 255;
    col[i * 3 + 1] = view.getUint8(off + 7) / 255;
    col[i * 3 + 2] = view.getUint8(off + 8) / 255;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  pcGroup.add(new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.05, vertexColors: true, sizeAttenuation: true,
    transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending,
  })));
}

/* ─── Model ─── */
async function loadModel() {
  const dracoLoader = new DRACOLoader();
  dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
  dracoLoader.setDecoderConfig({ type: 'js' }); // works everywhere, no WASM needed

  const loader = new GLTFLoader();
  loader.setDRACOLoader(dracoLoader);

  const gltf = await new Promise((resolve, reject) => {
    loader.load(
      'assets/model.glb',
      resolve,
      (p) => {
        if (p.total > 0) {
          const pct = 60 + (p.loaded / p.total) * 35;
          setProgress(Math.min(pct, 95), `Loading 3D model… ${(p.loaded / 1e6).toFixed(1)} MB`);
        }
      },
      reject,
    );
  });

  const model = gltf.scene;
  model.rotation.x = -Math.PI / 2;

  model.traverse(child => {
    if (!child.isMesh) return;
    const geo = child.geometry;
    totalTris += geo.index ? geo.index.count / 3 : geo.attributes.position.count / 3;
    if (child.material) {
      child.material.vertexColors = true;
      child.material.side = THREE.DoubleSide;
      child.material.roughness = 0.8;
      child.material.metalness = 0.1;
    }
  });

  modelGroup.add(model);
  console.log(`Model: ${Math.round(totalTris)} tris`);
}

/* ─── Events ─── */
function setupEvents() {
  $('start-btn').onclick = () => controls.lock();
  $('blocker').onclick = e => { if (e.target === $('blocker')) controls.lock(); };
  controls.addEventListener('lock', () => {
    $('blocker').style.display = 'none';
    $('blocker').classList.add('hidden');
  });
  controls.addEventListener('unlock', () => {
    $('blocker').style.display = 'flex';
    $('blocker').classList.remove('hidden');
  });

  const keyMap = {
    KeyW: 'fwd', ArrowUp: 'fwd', KeyS: 'back', ArrowDown: 'back',
    KeyA: 'left', ArrowLeft: 'left', KeyD: 'right', ArrowRight: 'right',
    ShiftLeft: 'sprint', ShiftRight: 'sprint',
  };

  addEventListener('keydown', e => {
    if (keyMap[e.code]) move[keyMap[e.code]] = true;
    if (e.code === 'KeyP') togglePointCloud();
    if (e.code === 'KeyR') toggleRobot();
    if (e.code === 'F2') { e.preventDefault(); takeScreenshot(); }
  });
  addEventListener('keyup', e => { if (keyMap[e.code]) move[keyMap[e.code]] = false; });

  for (const [id, d] of [['touch-fwd', 'fwd'], ['touch-back', 'back'], ['touch-left', 'left'], ['touch-right', 'right']]) {
    const btn = $(id);
    if (!btn) continue;
    btn.ontouchstart = e => { e.preventDefault(); move[d] = true; };
    btn.ontouchend = e => { e.preventDefault(); move[d] = false; };
  }

  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
}

/* ─── Toggles ─── */
async function togglePointCloud() {
  if (!pcLoaded) {
    const el = $('hint-pts');
    if (el) el.textContent = 'loading...';
    try {
      const buf = await fetch('assets/pointcloud.bin').then(r => r.arrayBuffer());
      buildPointCloud(buf);
      pcLoaded = true;
    } catch { if ($('hint-pts')) $('hint-pts').textContent = 'error'; return; }
  }
  pointCloudVisible = !pointCloudVisible;
  pcGroup.visible = pointCloudVisible;
  const el = $('hint-pts');
  if (el) { el.textContent = pointCloudVisible ? 'on' : 'off'; el.className = pointCloudVisible ? 'on' : ''; }
}

function toggleRobot() {
  robotPlaying = !robotPlaying;
  if (robotPlaying) { robotTime = 0; if (robotGroup) robotGroup.visible = true; }
  else { if (robotGroup) robotGroup.visible = false; }
  const el = $('hint-robot');
  if (el) { el.textContent = robotPlaying ? 'playing' : 'off'; el.className = robotPlaying ? 'on' : ''; }
}

/* ─── HUD ─── */
function updateHUD() {
  const p = camera.position;
  $('hud-pos').textContent = `${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)}`;
  $('hud-pts').textContent = totalPoints.toLocaleString();
  $('hud-tris').textContent = Math.round(totalTris).toLocaleString();
  $('hud-mode').textContent = pointCloudVisible ? 'MODEL+PTS' : 'MODEL';
}

/* ─── Minimap ─── */
let minimapReady = false, minimapImg = null, minimapMeta = null;

async function initMinimap() {
  try {
    const [meta, img] = await Promise.all([
      fetch('assets/minimap_meta.json').then(r => r.json()),
      new Promise(resolve => {
        const i = new Image(); i.onload = () => resolve(i); i.onerror = () => resolve(null);
        i.src = 'assets/minimap.png';
      }),
    ]);
    minimapMeta = meta; minimapImg = img;
    if (img && meta) {
      const c = document.getElementById('minimap-canvas');
      c.width = meta.width; c.height = meta.height;
      minimapReady = true;
    }
  } catch { /* ignore */ }
}

function drawMinimap() {
  if (!minimapReady) return;
  const c = document.getElementById('minimap-canvas');
  const ctx = c.getContext('2d');
  const m = minimapMeta, w = m.width, h = m.height;

  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(minimapImg, 0, 0);

  const px = (camera.position.x - m.origin_x) / m.resolution;
  const py = h - (-camera.position.z - m.origin_y) / m.resolution;
  const look = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const heading = Math.atan2(-look.z, look.x);

  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(-heading + Math.PI / 2);

  // FOV cone
  ctx.beginPath(); ctx.moveTo(0, 0);
  const fr = (80 / 2) * Math.PI / 180, cl = 25;
  ctx.lineTo(Math.sin(-fr) * cl, -Math.cos(-fr) * cl);
  ctx.lineTo(Math.sin(fr) * cl, -Math.cos(fr) * cl);
  ctx.closePath();
  ctx.fillStyle = 'rgba(122,232,180,0.12)'; ctx.fill();

  // Arrow
  ctx.beginPath(); ctx.moveTo(0, -10); ctx.lineTo(-4, 4); ctx.lineTo(4, 4); ctx.closePath();
  ctx.fillStyle = '#7ae8b4'; ctx.fill();
  ctx.restore();

  // Dot
  ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2);
  ctx.fillStyle = '#7ae8b4'; ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
}

/* ─── Robot Dog ─── */
function buildRobotDog() {
  robotGroup = new THREE.Group();
  const bm = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.6, metalness: 0.3 });
  const am = new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.5, metalness: 0.4 });
  const em = new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.8 });
  const lm = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.7 });

  // Body
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.12, 0.22), bm);
  body.position.y = 0.28; robotGroup.add(body);
  // Head
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 0.16), am);
  head.position.set(0.26, 0.30, 0); robotGroup.add(head);
  // Eyes
  for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.012, 6, 6), em);
    eye.position.set(0.32, 0.32, s * 0.05); robotGroup.add(eye);
  }
  // LiDAR
  const lid = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.03, 8), am);
  lid.position.set(0.05, 0.36, 0); robotGroup.add(lid);
  // Legs
  function addPart(geo, mat, px, py, pz) {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(px, py, pz);
    robotGroup.add(m);
  }
  for (const { x, z } of [{ x: .15, z: .10 }, { x: .15, z: -.10 }, { x: -.15, z: .10 }, { x: -.15, z: -.10 }]) {
    addPart(new THREE.BoxGeometry(.03, .14, .03), lm, x, .20, z);
    addPart(new THREE.BoxGeometry(.025, .14, .025), lm, x, .07, z);
    addPart(new THREE.CylinderGeometry(.015, .018, .01, 6), am, x, .005, z);
  }

  robotGroup.visible = false;
  scene.add(robotGroup);
}

function updateRobot(dt) {
  if (!robotPlaying || !robotPath || !robotGroup) return;
  robotTime += dt * robotSpeed;
  const dur = robotPath[robotPath.length - 1].t;
  if (robotTime > dur) robotTime = 0;

  let i = 0;
  for (; i < robotPath.length - 1; i++) if (robotPath[i + 1].t > robotTime) break;
  const j = Math.min(i + 1, robotPath.length - 1);
  const a = robotPath[i], b = robotPath[j];
  const sd = b.t - a.t;
  const f = sd > 0 ? (robotTime - a.t) / sd : 0;

  const x = a.x + (b.x - a.x) * f;
  const y = a.y + (b.y - a.y) * f;
  const z = a.z + (b.z - a.z) * f;
  let dh = b.h - a.h;
  if (dh > Math.PI) dh -= 2 * Math.PI;
  if (dh < -Math.PI) dh += 2 * Math.PI;

  robotGroup.position.set(x, z, -y);
  robotGroup.rotation.y = -(a.h + dh * f) + Math.PI / 2;

  const spd = sd > 0 ? Math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2) / sd : 0;
  if (spd > 0.05) robotGroup.position.y += Math.sin(robotTime * 8) * 0.008;
}

/* ─── Screenshot ─── */
function takeScreenshot() {
  renderer.render(scene, camera);
  renderer.domElement.toBlob(blob => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `robodog-3d_${Date.now()}.png`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, 'image/png');
}

/* ─── Render Loop ─── */
function tick() {
  requestAnimationFrame(tick);
  const now = performance.now();
  timer.update();
  const dt = Math.min(timer.getDelta(), 0.1);

  if (controls.isLocked) {
    const speed = move.sprint ? SPRINT : WALK;
    dir.z = +move.fwd - +move.back;
    dir.x = +move.right - +move.left;
    dir.normalize();
    if (dir.x) controls.moveRight(dir.x * speed * dt);
    if (dir.z) controls.moveForward(dir.z * speed * dt);
    if (camera.position.y < 1.6) camera.position.y = 1.6;
  }

  updateRobot(dt);

  flashLight.position.copy(camera.position);
  const look = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  flashTarget.position.copy(camera.position).add(look.multiplyScalar(8));
  flashLight.target = flashTarget;

  frameCount++;
  if (now - lastTime >= 1000) {
    $('fps-display').textContent = `${Math.round(frameCount * 1000 / (now - lastTime))} FPS`;
    frameCount = 0; lastTime = now;
  }

  drawMinimap();
  if (Math.random() < 0.02) updateHUD();
  renderer.render(scene, camera);
}

main().catch(console.error);
