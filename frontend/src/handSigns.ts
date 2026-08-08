export interface HandSign {
  letter: string
  rightHand: string
  description: string
  useBothHands: boolean
}

// Local images: frontend/public/hand-signs/ folder me A.png ... Z.png rakhna
const IMG = (L: string) => `/hand-signs/${L}.png`

export const HAND_SIGNS: Record<string, HandSign> = {
  A: { letter: 'A', rightHand: IMG('A'), description: 'Fist with thumb beside index finger', useBothHands: false },
  B: { letter: 'B', rightHand: IMG('B'), description: 'Four fingers up, thumb across palm', useBothHands: false },
  C: { letter: 'C', rightHand: IMG('C'), description: 'Curved hand like holding a cup', useBothHands: false },
  D: { letter: 'D', rightHand: IMG('D'), description: 'Index up, other fingers touch thumb', useBothHands: false },
  E: { letter: 'E', rightHand: IMG('E'), description: 'All fingers curled, thumb across', useBothHands: false },
  F: { letter: 'F', rightHand: IMG('F'), description: 'OK sign with 3 fingers up', useBothHands: false },
  G: { letter: 'G', rightHand: IMG('G'), description: 'Index and thumb pointing sideways', useBothHands: false },
  H: { letter: 'H', rightHand: IMG('H'), description: 'Index and middle pointing sideways', useBothHands: false },
  I: { letter: 'I', rightHand: IMG('I'), description: 'Pinky up, fist closed', useBothHands: false },
  J: { letter: 'J', rightHand: IMG('J'), description: 'Pinky traces J shape', useBothHands: false },
  K: { letter: 'K', rightHand: IMG('K'), description: 'Index, middle up, thumb between', useBothHands: false },
  L: { letter: 'L', rightHand: IMG('L'), description: 'Index and thumb form L', useBothHands: false },
  M: { letter: 'M', rightHand: IMG('M'), description: 'Three fingers over thumb', useBothHands: false },
  N: { letter: 'N', rightHand: IMG('N'), description: 'Two fingers over thumb', useBothHands: false },
  O: { letter: 'O', rightHand: IMG('O'), description: 'All fingers touch thumb forming O', useBothHands: false },
  P: { letter: 'P', rightHand: IMG('P'), description: 'Like K but pointing down', useBothHands: false },
  Q: { letter: 'Q', rightHand: IMG('Q'), description: 'Like G but pointing down', useBothHands: false },
  R: { letter: 'R', rightHand: IMG('R'), description: 'Index and middle crossed', useBothHands: false },
  S: { letter: 'S', rightHand: IMG('S'), description: 'Fist with thumb over fingers', useBothHands: false },
  T: { letter: 'T', rightHand: IMG('T'), description: 'Index between thumb and fingers', useBothHands: false },
  U: { letter: 'U', rightHand: IMG('U'), description: 'Index and middle up together', useBothHands: false },
  V: { letter: 'V', rightHand: IMG('V'), description: 'Index and middle in V shape', useBothHands: false },
  W: { letter: 'W', rightHand: IMG('W'), description: 'Index, middle, ring up in W', useBothHands: false },
  X: { letter: 'X', rightHand: IMG('X'), description: 'Index finger hooked', useBothHands: false },
  Y: { letter: 'Y', rightHand: IMG('Y'), description: 'Thumb and pinky out', useBothHands: false },
  Z: { letter: 'Z', rightHand: IMG('Z'), description: 'Index traces Z shape', useBothHands: false },
}

export const DEFAULT_SIGN: HandSign = HAND_SIGNS.A