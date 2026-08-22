import * as THREE from "./vendor/three.module.min.js";

// ---------------------------------------------------------------------------
// Karini Block Sorting Demo - Three.js scene
//
// Draws the RoArm-M2 with a gripper, colored blocks, and target slots.
// All geometry arrives from /api/scene (same as the inspection demo).
// ---------------------------------------------------------------------------

const MM = 0.001; // config is mm, scene is metres

const state = {
  scene: null,
  joints: { base: 0, shoulder: 0, elbow: 0, wrist: 0 },
  target: { base: 0, shoulder: 0, elbow: 0, wrist: 0 },
  gripper: "open",
  heldBlock: null,
  blockPositions: {},
  cellState: "connecting",
  standalone: false,
};

// -- renderer setup -------------------------------------------------------- //

const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);
scene.fog = new THREE.Fog(0x14161a, 2.5, 6.0);

const camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.05, 40);
camera.position.set(0.0, 0.5, 1.4);
camera.lookAt(0.25, 0.15, 0);

// Lighting
scene.add(new THREE.HemisphereLight(0x7d8ba0, 0x0d0f12, 0.5));
const key = new THREE.DirectionalLight(0xfff2e0, 1.4);
key.position.set(1.2, 1.8, 1.0);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -1.5;
key.shadow.camera.right = 1.5;
key.shadow.camera.top = 1.5;
key.shadow.camera.bottom = -1.5;
scene.add(key);
const fill = new THREE.DirectionalLight(0x88a6ff, 0.3);
fill.position.set(-1.0, 0.7, -0.6);
scene.add(fill);

// -- materials ------------------------------------------------------------- //

const matMetal = new THREE.MeshStandardMaterial({ color: 0xb9bfc7, metalness: 0.7, roughness: 0.35 });
const matJoint = new THREE.MeshStandardMaterial({ color: 0x2d3138, metalness: 0.5, roughness: 0.5 });
const matAccent = new THREE.MeshStandardMaterial({ color: 0xf0a53a, metalness: 0.3, roughness: 0.45 });
const matTable = new THREE.MeshStandardMaterial({ color: 0x1c2026, metalness: 0.05, roughness: 0.92 });
const matSlot = new THREE.MeshStandardMaterial({ color: 0x3f4d63, metalness: 0.1, roughness: 0.8, transparent: true, opacity: 0.6 });

// -- static scene ---------------------------------------------------------- //

const table = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.02, 1.0), matTable);
table.position.y = -0.01;
table.receiveShadow = true;
scene.add(table);

const grid = new THREE.GridHelper(1.4, 28, 0x2b313a, 0x21262d);
grid.position.y = 0.002;
scene.add(grid);

// -- arm ------------------------------------------------------------------- //

const armRoot = new THREE.Group();
scene.add(armRoot);
const gBase = new THREE.Group();
const gShoulder = new THREE.Group();
const gElbow = new THREE.Group();
const gWrist = new THREE.Group();
armRoot.add(gBase);
gBase.add(gShoulder);
gShoulder.add(gElbow);
gElbow.add(gWrist);

let gripperLeft = null;
let gripperRight = null;
let gripperOpenWidth = 0.06;

function buildArm(arm) {
  const H = arm.base_height * MM;
  const L1 = arm.upper_arm * MM;
  const L2 = arm.forearm * MM;
  const L3 = arm.eoat * MM;

  // Pedestal
  const pedestal = new THREE.Mesh(
    new THREE.CylinderGeometry(0.06, 0.075, H, 28), matJoint
  );
  pedestal.position.y = H / 2;
  pedestal.castShadow = true;
  gBase.add(pedestal);

  // Collar
  const collar = new THREE.Mesh(
    new THREE.TorusGeometry(0.064, 0.006, 12, 32), matAccent
  );
  collar.rotation.x = Math.PI / 2;
  collar.position.y = H * 0.94;
  gBase.add(collar);

  gShoulder.position.y = H;

  // Shoulder knuckle
  gShoulder.add(new THREE.Mesh(new THREE.SphereGeometry(0.035, 20, 16), matJoint));

  // Upper arm
  const arm1 = link(L1, 0.022, matMetal);
  gShoulder.add(arm1);

  // Elbow
  gElbow.position.x = L1;
  gElbow.add(new THREE.Mesh(new THREE.SphereGeometry(0.028, 20, 16), matJoint));

  // Forearm
  const arm2 = link(L2, 0.018, matMetal);
  gElbow.add(arm2);

  // Wrist
  gWrist.position.x = L2;
  gWrist.add(new THREE.Mesh(new THREE.SphereGeometry(0.022, 16, 12), matJoint));

  // Gripper mount (eoat extends along local +X from wrist)
  const mount = new THREE.Mesh(
    new THREE.CylinderGeometry(0.015, 0.012, L3 * 0.7, 12), matMetal
  );
  mount.position.set(L3 * 0.35, 0, 0);
  mount.rotation.z = Math.PI / 2;
  gWrist.add(mount);

  // Gripper fingers - positioned at the tip of the eoat (L3)
  const fingerGeo = new THREE.BoxGeometry(0.008, L3 * 0.35, 0.02);
  const fingerMat = matAccent;

  gripperLeft = new THREE.Mesh(fingerGeo, fingerMat);
  gripperLeft.position.set(L3, 0.015, 0);
  gripperLeft.castShadow = true;
  gWrist.add(gripperLeft);

  gripperRight = new THREE.Mesh(fingerGeo, fingerMat);
  gripperRight.position.set(L3, -0.015, 0);
  gripperRight.castShadow = true;
  gWrist.add(gripperRight);

  // Invisible marker at gripper tip for world position tracking
  const tipMarker = new THREE.Object3D();
  tipMarker.position.set(L3, 0, 0);
  tipMarker.name = "gripperTip";
  gWrist.add(tipMarker);
}

function link(length, radius, material) {
  const g = new THREE.CylinderGeometry(radius, radius, length, 20);
  g.translate(0, length / 2, 0);
  g.rotateZ(-Math.PI / 2);
  const m = new THREE.Mesh(g, material);
  m.castShadow = true;
  return m;
}

// -- blocks and slots ------------------------------------------------------ //

const blockMeshes = {};
const slotMeshes = {};

function buildBlocks(blocks) {
  for (const b of blocks) {
    const size = b.size * MM;
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(b.color),
      metalness: 0.2,
      roughness: 0.5,
    });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(size, size, size), mat);
    mesh.position.set(b.start_x * MM, size / 2, -b.start_y * MM);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);
    blockMeshes[b.id] = mesh;
  }
}

function buildSlots(slots, blockSize) {
  const size = blockSize * MM;
  for (const s of slots) {
    // Slot marker (flat square outline on table)
    const geo = new THREE.BoxGeometry(size * 1.3, 0.003, size * 1.3);
    const mesh = new THREE.Mesh(geo, matSlot);
    mesh.position.set(s.x * MM, 0.002, -s.y * MM);
    mesh.receiveShadow = true;
    scene.add(mesh);
    slotMeshes[s.id] = mesh;

    // Label
    // (skip text rendering - just use the slot markers visually)
  }
}

function buildEnvelope(env) {
  const [rMin, rMax] = env.radius;
  for (const r of [rMin * MM, rMax * MM]) {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(r - 0.002, r + 0.002, 96),
      new THREE.MeshBasicMaterial({
        color: 0x3f4d63, transparent: true, opacity: 0.5, side: THREE.DoubleSide,
      })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.003;
    scene.add(ring);
  }
}

// -- update loop ----------------------------------------------------------- //

function tick() {
  // Smooth joint interpolation
  const ease = 0.08;
  for (const k of ["base", "shoulder", "elbow", "wrist"]) {
    state.joints[k] += (state.target[k] - state.joints[k]) * ease;
  }

  // Apply joints to arm
  const d2r = Math.PI / 180;
  gBase.rotation.y = state.joints.base * d2r;
  gShoulder.rotation.z = state.joints.shoulder * d2r;
  gElbow.rotation.z = state.joints.elbow * d2r;
  gWrist.rotation.z = state.joints.wrist * d2r;

  // Gripper animation
  if (gripperLeft && gripperRight) {
    const targetOffset = state.gripper === "open" ? 0.015 : 0.006;
    const current = gripperLeft.position.y;
    const next = current + (targetOffset - current) * 0.1;
    gripperLeft.position.y = next;
    gripperRight.position.y = -next;
  }

  // Update block positions from state
  if (state.blockPositions) {
    for (const [id, pos] of Object.entries(state.blockPositions)) {
      const mesh = blockMeshes[id];
      if (mesh) {
        if (id === state.heldBlock) {
          // Block follows gripper tip
          const tipObj = gWrist.getObjectByName("gripperTip");
          const tip = new THREE.Vector3();
          if (tipObj) tipObj.getWorldPosition(tip);
          else gWrist.getWorldPosition(tip);
          mesh.position.x += (tip.x - mesh.position.x) * 0.15;
          mesh.position.z += (tip.z - mesh.position.z) * 0.15;
          mesh.position.y += (tip.y - 0.01 - mesh.position.y) * 0.15;
        } else {
          const size = mesh.geometry.parameters.width;
          const targetX = pos.x * MM;
          const targetZ = -pos.y * MM;
          mesh.position.x += (targetX - mesh.position.x) * 0.1;
          mesh.position.z += (targetZ - mesh.position.z) * 0.1;
          mesh.position.y += (size / 2 - mesh.position.y) * 0.1;
        }
      }
    }
  }

  // Fixed camera viewpoint (no orbit)
  camera.lookAt(0.25, 0.15, 0);

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

// -- resize ---------------------------------------------------------------- //

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

// -- network --------------------------------------------------------------- //

async function boot() {
  let sceneCfg = null;
  try {
    const res = await fetch("/api/scene");
    sceneCfg = await res.json();
  } catch {
    state.standalone = true;
    return; // No bridge
  }

  state.scene = sceneCfg;
  buildArm(sceneCfg.arm);
  buildEnvelope(sceneCfg.envelope);
  buildBlocks(sceneCfg.blocks);
  buildSlots(sceneCfg.slots, sceneCfg.blocks[0].size);

  state.target = { ...sceneCfg.home_joints };
  state.joints = { ...sceneCfg.home_joints };

  resize();
  connect();
  requestAnimationFrame(tick);
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type !== "status") return;
    const d = msg.data;
    state.target = d.joints;
    state.cellState = d.state;
    state.gripper = d.gripper;
    state.heldBlock = d.held_block;
    state.blockPositions = d.block_positions;

    // Update HUD
    document.getElementById("s-state").textContent = d.state;
    document.getElementById("s-backend").textContent = d.backend;
    document.getElementById("s-gripper").textContent = d.gripper;
    document.getElementById("s-held").textContent = d.held_block || "none";
    document.getElementById("narration").textContent = d.current_action || "";

    // Update block list
    updateBlockList(d.block_positions);
  };
  ws.onclose = () => {
    state.cellState = "disconnected";
    setTimeout(connect, 2000);
  };
}

function updateBlockList(positions) {
  const el = document.getElementById("block-list");
  if (!state.scene) return;
  let html = "";
  for (const b of state.scene.blocks) {
    const pos = positions[b.id];
    const posStr = pos ? `(${pos.x.toFixed(0)}, ${pos.y.toFixed(0)})` : "held";
    html += `<div class="block-item">
      <div class="block-dot" style="background:${b.color}"></div>
      <span>${b.label}</span>
      <span class="block-pos">${posStr}</span>
    </div>`;
  }
  el.innerHTML = html;
}

// -- sort controls (browser buttons) --------------------------------------- //

window.runSort = async function () {
  await doSort(["green", "red", "yellow"]);
};

window.runSort2 = async function () {
  await doSort(["yellow", "green", "red"]);
};

window.resetBlocks = async function () {
  addLog("Resetting blocks...");
  const res = await fetch("/api/reset", { method: "POST" });
  const data = await res.json();
  if (data.ok) addLog("Blocks reset.", "success");
  else addLog("Reset failed: " + (data.reason || ""), "error");
};

async function doSort(sequence) {
  addLog(`Planning sort: ${sequence.join(" → ")}...`);

  // Plan
  const planRes = await fetch("/api/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence }),
  });
  const plan = await planRes.json();

  if (!plan.ok) {
    addLog("Plan rejected: " + plan.violations.map(v => v.message).join("; "), "error");
    return;
  }
  addLog(`Plan approved (${plan.est_seconds}s). Executing...`);

  // Execute
  const execRes = await fetch("/api/execute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan_id: plan.plan_id }),
  });
  const exec = await execRes.json();

  if (!exec.ok) {
    addLog("Execution rejected: " + exec.violations.map(v => v.message).join("; "), "error");
    return;
  }

  // Poll until done
  const jobId = exec.job_id;
  const poll = setInterval(async () => {
    const jr = await fetch(`/api/job/${jobId}`);
    const job = await jr.json();
    if (job.state === "done") {
      clearInterval(poll);
      addLog(`Sort complete! ${sequence.join(" → ")}`, "success");
    } else if (job.state === "failed") {
      clearInterval(poll);
      addLog(`Sort failed: ${job.error}`, "error");
    }
  }, 1000);
}

function addLog(msg, cls = "") {
  const el = document.getElementById("log");
  const div = document.createElement("div");
  div.className = "entry " + cls;
  div.textContent = `${new Date().toLocaleTimeString()} ${msg}`;
  el.prepend(div);
  // Keep max 30 entries
  while (el.children.length > 30) el.removeChild(el.lastChild);
}

// -- start ----------------------------------------------------------------- //

boot();
