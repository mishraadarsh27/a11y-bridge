export interface Spec { f: number[]; thumb?: 'out' | 'up' | 'across' | 'tuck'; rot?: number; spread?: number }

// ASL fingerspelling: [thumb, index, middle, ring, pinky] extension 0-1
export const POSES: Record<string, Spec> = {
  A: { f: [1, 0, 0, 0, 0], thumb: 'up' },
  B: { f: [0, 1, 1, 1, 1], thumb: 'across' },
  C: { f: [.5, .5, .5, .5, .5], thumb: 'out' },
  D: { f: [0, 1, .15, .15, .15], thumb: 'across' },
  E: { f: [0, .25, .25, .25, .25], thumb: 'across' },
  F: { f: [.3, .3, 1, 1, 1], thumb: 'across' },
  G: { f: [.6, 1, 0, 0, 0], thumb: 'out', rot: 90 },
  H: { f: [0, 1, 1, 0, 0], rot: 90 },
  I: { f: [0, 0, 0, 0, 1], thumb: 'tuck' },
  J: { f: [0, 0, 0, 0, 1], thumb: 'tuck', rot: 25 },
  K: { f: [.7, 1, .8, 0, 0], thumb: 'up' },
  L: { f: [1, 1, 0, 0, 0], thumb: 'out' },
  M: { f: [0, .3, .3, .3, 0], thumb: 'tuck' },
  N: { f: [0, .3, .3, 0, 0], thumb: 'tuck' },
  O: { f: [.45, .45, .45, .45, .45], thumb: 'out' },
  P: { f: [.7, 1, .8, 0, 0], thumb: 'up', rot: 160 },
  Q: { f: [.6, 1, 0, 0, 0], thumb: 'out', rot: 160 },
  R: { f: [0, 1, 1, 0, 0], spread: .15 },
  S: { f: [0, 0, 0, 0, 0], thumb: 'across' },
  T: { f: [0, .1, 0, 0, 0], thumb: 'up' },
  U: { f: [0, 1, 1, 0, 0], spread: 0 },
  V: { f: [0, 1, 1, 0, 0], spread: 1 },
  W: { f: [0, 1, 1, 1, 0], spread: 1 },
  X: { f: [0, .5, 0, 0, 0], thumb: 'tuck' },
  Y: { f: [1, 0, 0, 0, 1], thumb: 'out' },
  Z: { f: [0, 1, 0, 0, 0], thumb: 'tuck' },
}
export const DEFAULT: Spec = { f: [.8, .8, .8, .8, .8], thumb: 'out' }