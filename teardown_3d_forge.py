#!/usr/bin/env python3
"""
teardown_3d_forge.py - Modular 4-Slot 3D Teardown Simulator Forge for Local Models
==================================================================================
Empowers a local LLM (`Gemma-4-31B + gen_022` @ :18935) to build `elvarofficial/teardown`-style
interactive 3D Exploded-View Mechanical Simulators (`Mellis HB-7` / `Robotic Mechanism` aesthetic)
by decomposing the page into 4 orthogonal JS slots (`kin.js`, `machine.js`, `console.js`, `tour.js`)
and assembling them into a single standalone HTML5 + Three.js page!
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENDPOINT = "http://127.0.0.1:18935"


MASTER_HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__MACHINE_TITLE__ // 3D Teardown & Field Simulator</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  :root {
    --bg-center: #f7f4ee;
    --bg-edge: #e6e1d6;
    --card-bg: rgba(249, 247, 242, 0.88);
    --card-border: #d5cfc2;
    --ink: #1b1e24;
    --muted: #7c786f;
    --accent: #ff4f1f;
    --accent-soft: rgba(255, 79, 31, 0.14);
    --ok: #109868;
  }
  * { box-sizing: border-box; user-select: none; }
  body {
    margin: 0; padding: 0; width: 100vw; height: 100vh; overflow: hidden;
    background: radial-gradient(circle at 50% 50%, var(--bg-center) 0%, var(--bg-edge) 100%);
    color: var(--ink);
    font-family: "JetBrains Mono", "SF Mono", "Consolas", monospace;
    font-size: 11px;
  }
  #webgl-container { position: absolute; inset: 0; z-index: 1; cursor: grab; }
  #webgl-container:active { cursor: grabbing; }
  /* CAD Polar Reticle Overlay */
  .cad-reticle {
    position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
    width: 640px; height: 640px; border-radius: 50%;
    border: 1px dashed rgba(27, 30, 36, 0.16);
    pointer-events: none; z-index: 2;
  }
  .cad-reticle::before {
    content: ""; position: absolute; inset: 85px; border-radius: 50%;
    border: 1px solid rgba(27, 30, 36, 0.1);
  }
  /* Top Banner */
  .top-bar {
    position: absolute; top: 0; left: 0; right: 0; height: 64px; z-index: 10;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 22px 0 0; border-bottom: 1px solid var(--card-border);
    background: rgba(247, 244, 238, 0.82); backdrop-filter: blur(8px);
  }
  .brand-block { display: flex; align-items: center; height: 100%; gap: 16px; }
  .orange-tag {
    height: 100%; background: var(--accent); color: #fff; font-weight: 800;
    padding: 0 20px; display: flex; align-items: center; gap: 10px; font-size: 14px; letter-spacing: 1px;
  }
  .title-stack .sub { font-size: 9.5px; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; }
  .title-stack .main { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; color: var(--ink); }
  .title-stack .main span { color: var(--accent); }
  .top-meta { display: flex; gap: 28px; align-items: center; font-size: 10px; }
  .meta-item .k { color: var(--muted); font-size: 9px; text-transform: uppercase; }
  .meta-item .v { font-weight: 700; font-size: 12px; margin-top: 2px; }
  /* Side Columns */
  .col-left, .col-right {
    position: absolute; top: 78px; bottom: 88px; width: 285px; z-index: 10;
    display: flex; flex-direction: column; gap: 10px; pointer-events: none;
  }
  .col-left { left: 18px; }
  .col-right { right: 18px; }
  .hud-card {
    background: var(--card-bg); border: 1px solid var(--card-border);
    border-radius: 6px; padding: 11px 13px; pointer-events: auto;
    backdrop-filter: blur(10px); box-shadow: 0 4px 18px rgba(0,0,0,0.04);
  }
  .card-hdr {
    display: flex; justify-content: space-between; align-items: center;
    font-weight: 800; font-size: 10px; letter-spacing: 1px; margin-bottom: 8px;
    border-left: 3px solid var(--ink); padding-left: 7px; text-transform: uppercase;
  }
  .card-hdr .num { color: var(--muted); font-weight: 400; }
  .big-readout { display: flex; justify-content: space-between; align-items: baseline; margin: 4px 0 8px 0; }
  .big-val { font-size: 24px; font-weight: 800; letter-spacing: -1px; }
  .big-unit { font-size: 11px; color: var(--muted); margin-left: 3px; }
  .pill-grid { display: flex; flex-wrap: wrap; gap: 5px; }
  .pill-btn {
    background: #fff; border: 1px solid var(--card-border); border-radius: 3px;
    padding: 4px 8px; font-family: inherit; font-size: 9.5px; font-weight: 700;
    color: var(--ink); cursor: pointer; text-transform: uppercase; transition: 0.15s;
  }
  .pill-btn:hover { border-color: var(--accent); }
  .pill-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .sub-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px dashed rgba(0,0,0,0.08); cursor: pointer;
  }
  .sub-row:last-child { border-bottom: none; }
  .sub-badge {
    font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 3px;
    background: #e2ddd2; color: var(--ink);
  }
  .sub-badge.detached { background: var(--accent); color: #fff; }
  .bar-track { flex: 1; height: 5px; background: #e2ddd2; border-radius: 2px; margin: 0 8px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--accent); transition: width 0.1s; }
  /* Bottom Timeline Bar */
  .bottom-bar {
    position: absolute; left: 320px; right: 320px; bottom: 18px; z-index: 10;
    background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 6px;
    padding: 10px 16px; display: flex; align-items: center; gap: 14px;
    backdrop-filter: blur(10px);
  }
  .play-btn {
    width: 34px; height: 34px; border-radius: 50%; border: none;
    background: var(--accent); color: #fff; font-weight: bold; cursor: pointer; font-size: 13px;
  }
  input[type="range"].scrub { flex: 1; accent-color: var(--accent); cursor: pointer; }
  .mode-watermark {
    position: absolute; left: 325px; bottom: 86px; z-index: 5; pointer-events: none;
  }
  .mode-watermark .lbl { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
  .mode-watermark .val { font-size: 24px; font-weight: 900; color: var(--accent); letter-spacing: 1px; }
</style>
</head>
<body>

<div class="cad-reticle"></div>
<div id="webgl-container"></div>

<div class="top-bar">
  <div class="brand-block">
    <div class="orange-tag" id="hdr-code">⚙️ HB-7</div>
    <div class="title-stack">
      <div class="sub" id="hdr-category">ROBOTIC ACTUATOR // PROCEDURAL 3D TEARDOWN SIMULATOR</div>
      <div class="main" id="hdr-title">Mellis HB-7 <span>/</span></div>
    </div>
  </div>
  <div class="top-meta">
    <div class="meta-item"><div class="k">EXPLODE SEPARATION</div><div class="v" id="meta-explode">0%</div></div>
    <div class="meta-item"><div class="k">SIM ID</div><div class="v">KL-022-THOR</div></div>
    <div class="meta-item"><div class="k">RUNTIME</div><div class="v" id="meta-runtime">00:00.0</div></div>
    <div class="meta-item"><div class="k">STATUS</div><div class="v" style="color:var(--accent)" id="meta-status">● OPERATIONAL</div></div>
  </div>
</div>

<div class="col-left">
  <div class="hud-card">
    <div class="card-hdr"><span>PRIMARY KINEMATICS</span><span class="num">01</span></div>
    <div class="big-readout">
      <div><span class="big-val" id="kin-freq">0.0</span><span class="big-unit" id="kin-unit1">HZ</span></div>
      <div><span class="big-val" id="kin-amp">±0</span><span class="big-unit" id="kin-unit2">DEG</span></div>
    </div>
    <canvas id="wave-canvas" width="255" height="58" style="width:100%; height:58px; background:rgba(0,0,0,0.03); border-radius:3px;"></canvas>
    <div style="display:flex; justify-content:space-between; color:var(--muted); font-size:9px; margin-top:5px;">
      <span>DRIVE SPEED</span>
      <input type="range" id="sl-speed" min="0" max="2.5" step="0.05" value="1.0" style="width:120px; accent-color:var(--accent);">
    </div>
  </div>

  <div class="hud-card">
    <div class="card-hdr"><span>POWER & THERMAL</span><span class="num">02</span></div>
    <div class="big-readout">
      <div><span class="big-val" id="pwr-pct">88%</span><span class="big-unit">LOAD</span></div>
      <div id="pwr-bars" style="display:flex; gap:3px;"></div>
    </div>
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:6px; border-top:1px solid var(--card-border); padding-top:6px;">
      <div><div style="color:var(--muted);font-size:8.5px">DRAW</div><div style="font-weight:800" id="pwr-watts">14.2 W</div></div>
      <div><div style="color:var(--muted);font-size:8.5px">BUS</div><div style="font-weight:800" id="pwr-volts">24.0 V</div></div>
      <div><div style="color:var(--muted);font-size:8.5px">CORE TEMP</div><div style="font-weight:800" id="pwr-temp">41.5 °C</div></div>
    </div>
  </div>

  <div class="hud-card" style="flex:1; display:flex; flex-direction:column;">
    <div class="card-hdr"><span>ACTUATOR TORQUE CHANNELS</span><span class="num">03</span></div>
    <div id="torque-list" style="display:flex; flex-direction:column; gap:7px; margin-top:4px;"></div>
  </div>
</div>

<div class="col-right">
  <div class="hud-card">
    <div class="card-hdr"><span>SPATIAL ORIENTATION</span><span class="num">04</span></div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:10px;">
      <div>PITCH: <b id="ori-pitch">0.0°</b></div>
      <div>ROLL: <b id="ori-roll">0.0°</b></div>
      <div>YAW: <b id="ori-yaw">0.0°</b></div>
      <div>STRESS: <b id="ori-stress">12 MPa</b></div>
    </div>
  </div>

  <div class="hud-card">
    <div class="card-hdr"><span>ISOLATE ASSEMBLIES</span><span class="num">05</span></div>
    <div class="pill-grid" id="assembly-pills"></div>
  </div>

  <div class="hud-card">
    <div class="card-hdr"><span>DISPLAY & TEARDOWN MODE</span><span class="num">06</span></div>
    <div class="pill-grid">
      <button class="pill-btn" id="btn-exploded" onclick="toggleExplode()">EXPLODED VIEW</button>
      <button class="pill-btn active" id="btn-turntable" onclick="toggleTurntable()">TURNTABLE</button>
      <button class="pill-btn" id="btn-wireframe" onclick="toggleWireframe()">X-RAY WIRE</button>
      <button class="pill-btn active" id="btn-autotour" onclick="toggleAutoTour()">AUTO TOUR</button>
    </div>
    <div style="margin-top:8px; display:flex; align-items:center; gap:8px; font-size:9.5px;">
      <span>EXPLODE:</span>
      <input type="range" id="sl-explode" min="0" max="1" step="0.01" value="0.45" style="flex:1; accent-color:var(--accent);" oninput="onExplodeSlider(this.value)">
    </div>
  </div>

  <div class="hud-card" style="flex:1; overflow-y:auto;">
    <div class="card-hdr"><span>SUBSYSTEM TEARDOWN (CLICK TO DETACH)</span><span class="num">07</span></div>
    <div id="subsystems-list"></div>
  </div>
</div>

<div class="mode-watermark">
  <div class="lbl">TEARDOWN STATE</div>
  <div class="val" id="watermark-state">REASSEMBLY</div>
</div>

<div class="bottom-bar">
  <button class="play-btn" id="btn-play" onclick="togglePlay()">❚❚</button>
  <div style="flex:1; display:flex; flex-direction:column; gap:3px;">
    <div style="display:flex; justify-content:space-between; font-size:9px; color:var(--muted);">
      <span>00:00 TAKE-OFF / ASSEMBLED</span>
      <span id="tour-caption" style="color:var(--accent); font-weight:700;">AUTO-TOUR ACTIVE // CLICK ANY PART OR BUTTON TO TAKE CONTROL</span>
      <span>24:00 FULL TEARDOWN</span>
    </div>
    <input type="range" class="scrub" id="sl-timeline" min="0" max="24" step="0.05" value="0" oninput="onScrubTimeline(this.value)">
  </div>
</div>

<script>
// === GLOBAL SIMULATOR STATE ===
const simState = {
  t: 0,
  playing: true,
  speed: 1.0,
  explode: 0.45,
  targetExplode: 0.45,
  turntable: true,
  wireframe: false,
  autoTour: true,
  isolatedId: 'ALL',
  detachedMap: {},
  waveHistory: new Array(64).fill(0),
};

// === SLOT 1: KINEMATICS & SUBSYSTEM DEFINITIONS (Generated by LocalSleep-MoE) ===
__SLOT_1_KINEMATICS__

// === SLOT 2: PROCEDURAL 3D GEOMETRY & POSE UPDATER (Generated by LocalSleep-MoE) ===
__SLOT_2_GEOMETRY__

// === THREE.JS STAGE & INTERACTIVE CONTROLLER ===
let scene, camera, renderer, rootGroup, subsystemGroups = {}, leaderLines = [];
let isDragging = false, prevMouse = {x: 0, y: 0}, spherical = {theta: 0.6, phi: 1.15, radius: 9.5};

function initStage() {
  const container = document.getElementById('webgl-container');
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 100);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  container.appendChild(renderer.domElement);

  const amb = new THREE.AmbientLight(0xffffff, 0.85);
  scene.add(amb);
  const dir1 = new THREE.DirectionalLight(0xfff8f0, 0.9);
  dir1.position.set(8, 14, 10);
  scene.add(dir1);
  const dir2 = new THREE.DirectionalLight(0xff4f1f, 0.35);
  dir2.position.set(-8, -5, -6);
  scene.add(dir2);

  rootGroup = new THREE.Group();
  scene.add(rootGroup);

  subsystemGroups = buildMachineGeometry(THREE, rootGroup);

  // Build orange dashed leader lines for exploded view
  const lineMat = new THREE.LineBasicMaterial({ color: 0xff4f1f, transparent: true, opacity: 0.55 });
  Object.keys(subsystemGroups).forEach(k => {
    const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const line = new THREE.Line(geo, lineMat);
    rootGroup.add(line);
    leaderLines.push({ key: k, line });
  });

  // Mouse Orbit
  container.addEventListener('mousedown', e => { isDragging = true; prevMouse = {x: e.clientX, y: e.clientY}; simState.autoTour = false; updateButtons(); });
  window.addEventListener('mouseup', () => isDragging = false);
  window.addEventListener('mousemove', e => {
    if (!isDragging) return;
    spherical.theta -= (e.clientX - prevMouse.x) * 0.008;
    spherical.phi = Math.max(0.25, Math.min(2.8, spherical.phi - (e.clientY - prevMouse.y) * 0.008));
    prevMouse = {x: e.clientX, y: e.clientY};
  });
  container.addEventListener('wheel', e => {
    spherical.radius = Math.max(4.5, Math.min(18, spherical.radius + e.deltaY * 0.005));
  });
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  initHudControls();
  requestAnimationFrame(animateLoop);
}

function initHudControls() {
  if (typeof MACHINE_META !== 'undefined') {
    document.getElementById('hdr-code').textContent = '⚙️ ' + (MACHINE_META.code || 'HB-7');
    document.getElementById('hdr-category').textContent = MACHINE_META.category || 'PROCEDURAL 3D SIMULATOR';
    document.getElementById('hdr-title').innerHTML = (MACHINE_META.title || 'Mellis HB-7') + ' <span>/</span>';
    document.getElementById('kin-unit1').textContent = MACHINE_META.primaryUnit || 'HZ';
    document.getElementById('kin-unit2').textContent = MACHINE_META.secondaryUnit || 'DEG';
  }
  const pillsDiv = document.getElementById('assembly-pills');
  const subListDiv = document.getElementById('subsystems-list');
  pillsDiv.innerHTML = '<button class="pill-btn active" onclick="selectIsolate(\\'ALL\\')">ALL</button>';
  subListDiv.innerHTML = '';

  (SUBSYSTEM_DEFS || []).forEach((s, idx) => {
    simState.detachedMap[s.id] = true;
    const b = document.createElement('button');
    b.className = 'pill-btn';
    b.dataset.id = s.id;
    b.textContent = s.shortName || s.id;
    b.onclick = () => selectIsolate(s.id);
    pillsDiv.appendChild(b);

    const row = document.createElement('div');
    row.className = 'sub-row';
    row.onclick = () => toggleSubsystemDetach(s.id);
    row.innerHTML = `
      <div>
        <div style="font-weight:800">${String(idx+1).padStart(2,'0')} ${s.name}</div>
        <div style="font-size:9px;color:var(--muted)">${s.spec || ''}</div>
      </div>
      <span class="sub-badge detached" id="badge-${s.id}">DETACHED</span>
    `;
    subListDiv.appendChild(row);
  });

  const pwrBars = document.getElementById('pwr-bars');
  for (let i = 0; i < 10; i++) {
    const d = document.createElement('div');
    d.style.cssText = 'width:7px;height:16px;border-radius:1px;background:' + (i < 7 ? 'var(--ok)' : (i < 9 ? 'var(--accent)' : '#d5cfc2'));
    pwrBars.appendChild(d);
  }
  document.getElementById('sl-speed').addEventListener('input', e => simState.speed = parseFloat(e.target.value));
}

function selectIsolate(id) {
  simState.isolatedId = id;
  simState.autoTour = false;
  document.querySelectorAll('#assembly-pills .pill-btn').forEach(b => {
    b.classList.toggle('active', (b.dataset.id || 'ALL') === id);
  });
  updateButtons();
}

function toggleSubsystemDetach(id) {
  simState.autoTour = false;
  simState.detachedMap[id] = !simState.detachedMap[id];
  const badge = document.getElementById('badge-' + id);
  if (badge) {
    badge.textContent = simState.detachedMap[id] ? 'DETACHED' : 'ATTACHED';
    badge.classList.toggle('detached', simState.detachedMap[id]);
  }
  updateButtons();
}

function toggleExplode() {
  simState.autoTour = false;
  simState.targetExplode = simState.targetExplode > 0.2 ? 0.0 : 0.75;
  document.getElementById('sl-explode').value = simState.targetExplode;
  updateButtons();
}
function onExplodeSlider(val) {
  simState.autoTour = false;
  simState.targetExplode = parseFloat(val);
  simState.explode = simState.targetExplode;
  updateButtons();
}
function toggleTurntable() { simState.turntable = !simState.turntable; updateButtons(); }
function toggleWireframe() {
  simState.wireframe = !simState.wireframe;
  rootGroup.traverse(obj => { if (obj.isMesh && obj.material) obj.material.wireframe = simState.wireframe; });
  updateButtons();
}
function toggleAutoTour() { simState.autoTour = !simState.autoTour; updateButtons(); }
function togglePlay() {
  simState.playing = !simState.playing;
  document.getElementById('btn-play').textContent = simState.playing ? '❚❚' : '▶';
}
function onScrubTimeline(val) {
  simState.t = parseFloat(val);
  simState.autoTour = false;
  updateButtons();
}
function updateButtons() {
  document.getElementById('btn-exploded').classList.toggle('active', simState.targetExplode > 0.15);
  document.getElementById('btn-turntable').classList.toggle('active', simState.turntable);
  document.getElementById('btn-wireframe').classList.toggle('active', simState.wireframe);
  document.getElementById('btn-autotour').classList.toggle('active', simState.autoTour);
}

let lastFrameMs = performance.now();
function animateLoop(nowMs) {
  const dt = Math.min(0.05, (nowMs - lastFrameMs) / 1000);
  lastFrameMs = nowMs;

  if (simState.playing) {
    simState.t = (simState.t + dt * simState.speed) % 24.0;
    document.getElementById('sl-timeline').value = simState.t.toFixed(2);
  }

  if (simState.autoTour) {
    const phase = simState.t / 24.0;
    simState.targetExplode = phase < 0.25 ? 0.0 : (phase < 0.65 ? 0.72 : 0.35);
    document.getElementById('sl-explode').value = simState.targetExplode.toFixed(2);
    document.getElementById('tour-caption').textContent =
      phase < 0.25 ? 'PHASE 1 // ASSEMBLED NOMINAL OPERATION' :
      (phase < 0.65 ? 'PHASE 2 // FULL SUBSYSTEM EXPLODED TEARDOWN' : 'PHASE 3 // DYNAMIC REASSEMBLY DIAGNOSTIC');
  }

  simState.explode += (simState.targetExplode - simState.explode) * 0.08;
  if (simState.turntable && !isDragging) spherical.theta += dt * 0.35;

  camera.position.x = spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
  camera.position.y = spherical.radius * Math.cos(spherical.phi);
  camera.position.z = spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
  camera.lookAt(0, 0, 0);

  const kin = computeKinematics(simState.t, simState);
  updateMachinePose(subsystemGroups, kin, simState);

  // Update subsystem explode positions & leader lines
  (SUBSYSTEM_DEFS || []).forEach(s => {
    const grp = subsystemGroups[s.id];
    if (!grp) return;
    const effExplode = simState.detachedMap[s.id] ? simState.explode : 0.0;
    const hx = (s.homePos && s.homePos[0]) || 0;
    const hy = (s.homePos && s.homePos[1]) || 0;
    const hz = (s.homePos && s.homePos[2]) || 0;
    const ex = hx + ((s.explodeVec && s.explodeVec[0]) || 0) * effExplode;
    const ey = hy + ((s.explodeVec && s.explodeVec[1]) || 0) * effExplode;
    const ez = hz + ((s.explodeVec && s.explodeVec[2]) || 0) * effExplode;
    grp.position.set(ex, ey, ez);

    const visible = (simState.isolatedId === 'ALL' || simState.isolatedId === s.id);
    grp.visible = visible;

    const lObj = leaderLines.find(l => l.key === s.id);
    if (lObj) {
      lObj.line.visible = visible && effExplode > 0.05;
      const posArr = lObj.line.geometry.attributes.position.array;
      posArr[0] = hx; posArr[1] = hy; posArr[2] = hz;
      posArr[3] = ex; posArr[4] = ey; posArr[5] = ez;
      lObj.line.geometry.attributes.position.needsUpdate = true;
    }
  });

  // Update HUD Readouts (Number-safe coercion for LLM string/number outputs)
  const num = (v, def = 0) => { const n = parseFloat(Array.isArray(v) ? v[0] : v); return Number.isFinite(n) ? n : def; };
  document.getElementById('meta-explode').textContent = Math.round(simState.explode * 100) + '%';
  document.getElementById('meta-runtime').textContent = '00:' + simState.t.toFixed(1).padStart(4, '0');
  document.getElementById('watermark-state').textContent = simState.explode > 0.35 ? 'REASSEMBLY' : 'OPERATIONAL';
  document.getElementById('kin-freq').textContent = num(kin.primaryVal, 0).toFixed(1);
  const secV = num(kin.secondaryVal, 0);
  document.getElementById('kin-amp').textContent = (secV >= 0 ? '+' : '') + secV.toFixed(1);
  document.getElementById('pwr-pct').textContent = Math.round(num(kin.loadPct, 80)) + '%';
  document.getElementById('pwr-watts').textContent = num(kin.powerWatts, 12.4).toFixed(1) + ' W';
  document.getElementById('pwr-volts').textContent = num(kin.busVolts, 24.0).toFixed(1) + ' V';
  document.getElementById('pwr-temp').textContent = num(kin.tempC, 42.0).toFixed(1) + ' °C';
  document.getElementById('ori-pitch').textContent = num(kin.pitch, 0).toFixed(1) + '°';
  document.getElementById('ori-roll').textContent = num(kin.roll, 0).toFixed(1) + '°';
  document.getElementById('ori-yaw').textContent = ((spherical.theta * 57.3) % 360).toFixed(0) + '°';
  document.getElementById('ori-stress').textContent = num(kin.stressMpa, 18.5).toFixed(1) + ' MPa';

  // Torque Channels
  const tList = document.getElementById('torque-list');
  if (kin.channels && tList) {
    tList.innerHTML = kin.channels.map(ch => `
      <div style="display:flex; align-items:center; font-size:9.5px;">
        <span style="width:68px;font-weight:700">${ch.label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, Math.max(5, num(ch.pct, 50)))}%"></div></div>
        <span style="width:52px;text-align:right;color:var(--muted)">${ch.val}</span>
      </div>
    `).join('');
  }

  // Draw Waveform Canvas
  simState.waveHistory.push(num(kin.waveSample, Math.sin(simState.t * 6)));
  if (simState.waveHistory.length > 64) simState.waveHistory.shift();
  const cvs = document.getElementById('wave-canvas');
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  ctx.strokeStyle = '#ff4f1f';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  simState.waveHistory.forEach((v, i) => {
    const x = (i / 63) * cvs.width;
    const y = cvs.height * 0.5 - v * (cvs.height * 0.38);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  renderer.render(scene, camera);
  requestAnimationFrame(animateLoop);
}

window.addEventListener('DOMContentLoaded', initStage);
</script>
</body>
</html>
"""


def strip_js_fences(txt: str) -> str:
    s = txt.strip()
    if "```" in s:
        for block in s.split("```")[1:]:
            lines = block.splitlines()
            if lines and lines[0].strip().lower() in ("js", "javascript"):
                lines = lines[1:]
            cand = "\n".join(lines).strip()
            if cand:
                return cand
    return s


def call_local_moe(endpoint: str, sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
    payload = {
        "model": "localsleep-moe",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 16384,
        "temperature": 0.4,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {
        "content": strip_js_fences(data["choices"][0]["message"]["content"]),
        "telemetry": data.get("localsleep_telemetry") or {},
    }


def forge_teardown_simulator(machine_topic: str, out_html: Path, endpoint: str = DEFAULT_ENDPOINT) -> Dict[str, Any]:
    t0 = time.time()
    sys_p = (
        "你是 `elvarofficial/teardown` 3D 機械拆解模擬器的核心工程師。請只輸出純 JavaScript 程式碼（不要 HTML 標籤，不要省略任何邏輯）。"
    )

    # Slot 1: Kinematics & Subsystem Definitions
    p_slot1 = (
        f"我們正在建置『{machine_topic}』的 3D 爆炸拆解模擬器。\n"
        "請輸出 JavaScript 定義以下 3 個全域物件/函式：\n"
        "1. `const MACHINE_META = { code: '...', title: '...', category: '...', primaryUnit: 'HZ', secondaryUnit: 'DEG' };`\n"
        "2. `const SUBSYSTEM_DEFS = [ ... ];` 必須包含 6 個正交機械子系統，每個元素格式為：\n"
        "   `{ id: 'SUB_1', shortName: 'OPTICS', name: 'OPTIC HEAD', spec: '2x 18MM LENS · IR', homePos: [0, 0, 0], explodeVec: [dx, dy, dz] }`\n"
        "   （注意：`homePos` 預設為 `[0,0,0]`，`explodeVec` 為爆炸拆解時往外飛出的 3D 位移向量，範圍約 -3.5 到 +3.5）。\n"
        "3. `function computeKinematics(t, simState)`：根據時間 `t` (秒) 與 `simState.speed`，計算並回傳：\n"
        "   `{ primaryVal, secondaryVal, loadPct, powerWatts, busVolts, tempC, pitch, roll, stressMpa, waveSample, angles: { ... }, channels: [ {label: 'CH1', pct: 75, val: '4.2 Nm'}, ... (共6條) ] }`\n"
        "請直接輸出這段完整的 JavaScript 程式碼："
    )
    print("🔧 [Slot 1/2] Asking LocalSleep-MoE (:18935) to forge Kinematics & 6 Subsystem Contracts...")
    r1 = call_local_moe(endpoint, sys_p, p_slot1)
    slot1_js = r1["content"]
    print(f"   ✅ Slot 1 done ({len(slot1_js)} chars | {r1['telemetry'].get('tps')} tok/s)")

    # Slot 2: Procedural 3D Geometry & Pose Updater
    p_slot2 = (
        f"以下是『{machine_topic}』在 Slot 1 定義的子系統與運動學合約：\n"
        f"```javascript\n{slot1_js}\n```\n"
        "現在請輸出 Slot 2 的兩個純 JavaScript 函式（使用傳入的 `THREE` 物件）：\n"
        "1. `function buildMachineGeometry(THREE, rootGroup)`：\n"
        "   - 建立 3 種標準工業材質：\n"
        "     • 消光碳黑主體：`new THREE.MeshStandardMaterial({color: 0x22262e, roughness: 0.45, metalness: 0.6})`\n"
        "     • 精密亮銀軸承/連桿：`new THREE.MeshStandardMaterial({color: 0xd8dee9, roughness: 0.3, metalness: 0.8})`\n"
        "     • 國際信號橘重點作動件：`new THREE.MeshStandardMaterial({color: 0xff4f1f, roughness: 0.35, metalness: 0.2})`\n"
        "   - 針對 `SUBSYSTEM_DEFS` 中的每一個 `s.id`，建立一個 `const grp = new THREE.Group(); rootGroup.add(grp); groups[s.id] = grp;`\n"
        "   - 在每個子系統 `grp` 內，使用豐富的幾何體 (`CylinderGeometry`, `SphereGeometry`, `TorusGeometry`, `BoxGeometry`, `ConeGeometry`) 組合出具有真實機械結構感（齒輪環、活塞連桿、光學鏡頭、散熱鰭片、多關節支架、仿生翼/致動器）的 3D 零件，並將可動子零件掛在 `grp.userData` 上。\n"
        "   - 回傳 `groups` 字典。\n"
        "2. `function updateMachinePose(groups, kin, simState)`：\n"
        "   - 根據 `kin` 與 `simState.t`，讓各子系統內的齒輪、翅膀、連桿、鏡頭或轉子產生生動的機械運轉動畫！\n"
        "請直接輸出這兩個完整函式的 JavaScript 程式碼："
    )
    print("🎨 [Slot 2/2] Asking LocalSleep-MoE (:18935) to forge Procedural 3D Three.js Geometry & Pose Animator...")
    r2 = call_local_moe(endpoint, sys_p, p_slot2)
    slot2_js = r2["content"]
    print(f"   ✅ Slot 2 done ({len(slot2_js)} chars | {r2['telemetry'].get('tps')} tok/s)")

    final_html = (
        MASTER_HTML_SHELL.replace("__MACHINE_TITLE__", machine_topic)
        .replace("__SLOT_1_KINEMATICS__", slot1_js)
        .replace("__SLOT_2_GEOMETRY__", slot2_js)
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(final_html, encoding="utf-8")
    return {
        "output_html": str(out_html),
        "total_bytes": len(final_html.encode("utf-8")),
        "slot1_chars": len(slot1_js),
        "slot2_chars": len(slot2_js),
        "elapsed_sec": round(time.time() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge an interactive 3D Teardown Simulator via LocalSleep-MoE (:18935)")
    parser.add_argument(
        "--machine",
        default="Mellis HB-7 Flapping-Wing Robotic Pollinator Drone",
        help="Target machine to build a 3D teardown simulator for",
    )
    parser.add_argument(
        "--out",
        default=str(BASE_DIR / "exports" / "teardown_mellis_hb7_sim.html"),
        help="Output HTML file path",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()

    res = forge_teardown_simulator(args.machine, Path(args.out), endpoint=args.endpoint)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
