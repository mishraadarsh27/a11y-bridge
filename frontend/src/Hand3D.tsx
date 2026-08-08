import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { POSES, DEFAULT, Spec } from './signPoses'

const COLORS = [0xf43f5e, 0x3b82f6, 0x22c55e, 0xf59e0b, 0xa855f7]

interface Finger { joints: THREE.Object3D[]; spread: number }

function mat(c: number, glow = 0.3) {
  return new THREE.MeshStandardMaterial({ color: c, roughness: 0.35, metalness: 0.25, emissive: c, emissiveIntensity: glow })
}

function buildHand(group: THREE.Group): Finger[] {
  const palm = new THREE.Mesh(new THREE.BoxGeometry(1.15, 1.0, 0.28), mat(0x94a3b8, 0.08))
  group.add(palm)
  const wrist = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.28, 0.6, 16), mat(0x64748b, 0.08))
  wrist.position.y = -0.78
  group.add(wrist)

  const defs = [
    { x: -0.62, y: -0.15, lens: [0.45, 0.35, 0.3], spread: 0 },
    { x: -0.38, y: 0.5, lens: [0.5, 0.35, 0.28], spread: 0.14 },
    { x: -0.13, y: 0.55, lens: [0.56, 0.4, 0.3], spread: 0 },
    { x: 0.13, y: 0.5, lens: [0.5, 0.35, 0.28], spread: -0.14 },
    { x: 0.38, y: 0.42, lens: [0.4, 0.3, 0.24], spread: -0.28 },
  ]
  const fingers: Finger[] = []
  defs.forEach((d, fi) => {
    const base = new THREE.Object3D()
    base.position.set(d.x, d.y, 0)
    group.add(base)
    const joints = [base]
    let parent: THREE.Object3D = base
    d.lens.forEach((len, si) => {
      const j = new THREE.Object3D()
      j.position.y = si === 0 ? 0 : d.lens[si - 1]
      parent.add(j)
      const bone = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.055, len, 12), mat(COLORS[fi]))
      bone.position.y = len / 2
      j.add(bone)
      const knob = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 16), mat(COLORS[fi], 0.6))
      j.add(knob)
      parent = j
      joints.push(j)
    })
    fingers.push({ joints, spread: d.spread })
  })
  return fingers
}

function targetRotations(fingers: Finger[], spec: Spec) {
  const spread = spec.spread ?? 0.6
  const targets: { obj: THREE.Object3D; x: number; z: number }[] = []
  for (let fi = 0; fi < 5; fi++) {
    const f = fingers[fi]
    let ext = spec.f[fi]
    let baseZ = f.spread * spread
    if (fi === 0) {
      const m = spec.thumb ?? 'tuck'
      baseZ = m === 'out' ? 1.0 : m === 'up' ? 0.3 : m === 'across' ? -1.35 : 0.5
      ext = m === 'tuck' ? 0.15 : m === 'across' ? 0.9 : spec.f[0]
    }
    const bend = (1 - ext) * 1.5
    targets.push({ obj: f.joints[0], x: bend * 0.5, z: baseZ })
    targets.push({ obj: f.joints[1], x: bend * 0.6, z: 0 })
    targets.push({ obj: f.joints[2], x: bend * 0.7, z: 0 })
  }
  return targets
}

export function Hand3D({ letter }: { letter: string }) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const apiRef = useRef<{ setLetter: (l: string) => void } | null>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / Math.max(mount.clientHeight, 1), 0.1, 100)
    camera.position.set(0, 0.4, 4.2)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    mount.appendChild(renderer.domElement)

    scene.add(new THREE.AmbientLight(0xffffff, 0.8))
    const dl = new THREE.DirectionalLight(0xffffff, 1.4); dl.position.set(2, 3, 4); scene.add(dl)
    const pl = new THREE.PointLight(0x22d3ee, 0.8); pl.position.set(-3, -1, 3); scene.add(pl)

    const spin = new THREE.Group(); scene.add(spin)
    const hand = new THREE.Group(); spin.add(hand)
    const fingers = buildHand(hand)

    let targets = targetRotations(fingers, DEFAULT)
    apiRef.current = {
      setLetter: (L: string) => {
        const spec = POSES[L.toUpperCase()] ?? DEFAULT
        targets = targetRotations(fingers, spec)
        hand.rotation.z = -((spec.rot ?? 0) * Math.PI) / 180
      },
    }

    let dragging = false, px = 0, py = 0
    const onDown = (e: PointerEvent) => { dragging = true; px = e.clientX; py = e.clientY }
    const onMove = (e: PointerEvent) => {
      if (!dragging) return
      spin.rotation.y += (e.clientX - px) * 0.01
      spin.rotation.x = Math.max(-0.8, Math.min(0.8, spin.rotation.x + (e.clientY - py) * 0.005))
      px = e.clientX; py = e.clientY
    }
    const onUp = () => { dragging = false }
    renderer.domElement.addEventListener('pointerdown', onDown)
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)

    let raf = 0
    const loop = () => {
      targets.forEach(t => {
        t.obj.rotation.x += (t.x - t.obj.rotation.x) * 0.12
        t.obj.rotation.z += (t.z - t.obj.rotation.z) * 0.12
      })
      if (!dragging) spin.rotation.y += 0.004
      renderer.render(scene, camera)
      raf = requestAnimationFrame(loop)
    }
    loop()

    const onResize = () => {
      renderer.setSize(mount.clientWidth, mount.clientHeight)
      camera.aspect = mount.clientWidth / Math.max(mount.clientHeight, 1)
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(onResize); ro.observe(mount)

    return () => {
      cancelAnimationFrame(raf); ro.disconnect()
      renderer.domElement.removeEventListener('pointerdown', onDown)
      window.removeEventListener('pointermove', onMove); window.removeEventListener('pointerup', onUp)
      renderer.dispose()
      if (renderer.domElement.parentElement === mount) mount.removeChild(renderer.domElement)
    }
  }, [])

  useEffect(() => { apiRef.current?.setLetter(letter) }, [letter])

  return <div ref={mountRef} style={{ width: '100%', height: 340, cursor: 'grab', borderRadius: 12, background: 'radial-gradient(400px 200px at 50% 40%, rgba(59,130,246,.12), transparent)' }} />
}