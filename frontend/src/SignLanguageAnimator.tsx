import React, { useEffect, useRef, useState } from 'react'

interface SignLanguageAnimatorProps {
  text: string
  isPlaying: boolean
  onComplete?: () => void
}

// Map letters/words to sign language gestures
const signMappings: Record<string, string> = {
  'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'h': 'H',
  'i': 'I', 'j': 'J', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O', 'p': 'P',
  'q': 'Q', 'r': 'R', 's': 'S', 't': 'T', 'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X',
  'y': 'Y', 'z': 'Z',
  'hello': 'HELLO', 'hi': 'HI', 'yes': 'YES', 'no': 'NO', 'thank you': 'THANK YOU',
  'please': 'PLEASE', 'sorry': 'SORRY', 'goodbye': 'GOODBYE', 'how are you': 'HOW ARE YOU'
}

// Hand landmark positions for sign language (21 points: wrist + 5 fingers with 4 joints each)
interface HandPose {
  wrist: { x: number; y: number }
  thumb: { x: number; y: number }[]
  index: { x: number; y: number }[]
  middle: { x: number; y: number }[]
  ring: { x: number; y: number }[]
  pinky: { x: number; y: number }[]
}

// Sign language gestures - normalized coordinates (0-1)
const signGestures: Record<string, HandPose> = {
  'A': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.55, y: 0.6 }, { x: 0.55, y: 0.5 }, { x: 0.55, y: 0.4 }, { x: 0.55, y: 0.35 }],
    index: [{ x: 0.5, y: 0.55 }, { x: 0.5, y: 0.45 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.25 }],
    middle: [{ x: 0.5, y: 0.55 }, { x: 0.5, y: 0.45 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.25 }],
    ring: [{ x: 0.5, y: 0.55 }, { x: 0.5, y: 0.45 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.25 }],
    pinky: [{ x: 0.5, y: 0.55 }, { x: 0.5, y: 0.45 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.25 }]
  },
  'B': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.45, y: 0.65 }, { x: 0.45, y: 0.55 }, { x: 0.45, y: 0.5 }, { x: 0.45, y: 0.48 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }]
  },
  'C': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.55, y: 0.65 }, { x: 0.58, y: 0.55 }, { x: 0.6, y: 0.48 }, { x: 0.62, y: 0.45 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.52, y: 0.55 }, { x: 0.54, y: 0.48 }, { x: 0.56, y: 0.42 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.52, y: 0.55 }, { x: 0.54, y: 0.48 }, { x: 0.56, y: 0.42 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.52, y: 0.55 }, { x: 0.54, y: 0.48 }, { x: 0.56, y: 0.42 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.52, y: 0.55 }, { x: 0.54, y: 0.48 }, { x: 0.56, y: 0.42 }]
  },
  'HELLO': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.45, y: 0.65 }, { x: 0.45, y: 0.58 }, { x: 0.45, y: 0.52 }, { x: 0.45, y: 0.48 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }]
  },
  'HI': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.45, y: 0.65 }, { x: 0.45, y: 0.58 }, { x: 0.45, y: 0.52 }, { x: 0.45, y: 0.48 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.2 }]
  },
  'YES': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.55, y: 0.65 }, { x: 0.55, y: 0.55 }, { x: 0.55, y: 0.45 }, { x: 0.55, y: 0.35 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }]
  },
  'NO': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.55, y: 0.65 }, { x: 0.58, y: 0.6 }, { x: 0.6, y: 0.55 }, { x: 0.6, y: 0.52 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.25 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }]
  },
  'THANK YOU': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.55, y: 0.65 }, { x: 0.55, y: 0.55 }, { x: 0.55, y: 0.45 }, { x: 0.55, y: 0.4 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.25 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.25 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }]
  },
  'PLEASE': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.48, y: 0.68 }, { x: 0.46, y: 0.66 }, { x: 0.45, y: 0.64 }, { x: 0.44, y: 0.63 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }]
  },
  'SORRY': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.55, y: 0.65 }, { x: 0.55, y: 0.58 }, { x: 0.55, y: 0.52 }, { x: 0.55, y: 0.48 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }]
  },
  // Add more letters
  'D': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.55, y: 0.65 }, { x: 0.58, y: 0.6 }, { x: 0.6, y: 0.55 }, { x: 0.62, y: 0.52 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.25 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }]
  },
  'E': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.55, y: 0.65 }, { x: 0.55, y: 0.58 }, { x: 0.55, y: 0.52 }, { x: 0.55, y: 0.48 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }]
  },
  'L': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.45, y: 0.65 }, { x: 0.45, y: 0.58 }, { x: 0.45, y: 0.52 }, { x: 0.45, y: 0.48 }],
    index: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.35 }, { x: 0.5, y: 0.25 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }]
  },
  'O': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.52, y: 0.65 }, { x: 0.54, y: 0.6 }, { x: 0.56, y: 0.55 }, { x: 0.58, y: 0.52 }],
    index: [{ x: 0.48, y: 0.65 }, { x: 0.46, y: 0.6 }, { x: 0.44, y: 0.55 }, { x: 0.42, y: 0.52 }],
    middle: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    ring: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }],
    pinky: [{ x: 0.5, y: 0.65 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.48 }]
  },
  // Default/relaxed hand
  'DEFAULT': {
    wrist: { x: 0.5, y: 0.7 },
    thumb: [{ x: 0.45, y: 0.68 }, { x: 0.48, y: 0.65 }, { x: 0.5, y: 0.62 }, { x: 0.52, y: 0.6 }],
    index: [{ x: 0.5, y: 0.68 }, { x: 0.5, y: 0.6 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.45 }],
    middle: [{ x: 0.5, y: 0.68 }, { x: 0.5, y: 0.58 }, { x: 0.5, y: 0.48 }, { x: 0.5, y: 0.4 }],
    ring: [{ x: 0.5, y: 0.68 }, { x: 0.5, y: 0.6 }, { x: 0.5, y: 0.52 }, { x: 0.5, y: 0.45 }],
    pinky: [{ x: 0.5, y: 0.68 }, { x: 0.5, y: 0.62 }, { x: 0.5, y: 0.56 }, { x: 0.5, y: 0.5 }]
  }
}

function drawHand(ctx: CanvasRenderingContext2D, pose: HandPose, width: number, height: number) {
  ctx.clearRect(0, 0, width, height)

  // Set drawing style
  ctx.strokeStyle = '#1e40af'
  ctx.fillStyle = '#3b82f6'
  ctx.lineWidth = 5
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  const wristX = pose.wrist.x * width
  const wristY = pose.wrist.y * height

  // Draw wrist (larger and more visible)
  ctx.fillStyle = '#1e3a8a'
  ctx.fillRect(wristX - 20, wristY - 12, 40, 24)
  ctx.strokeStyle = '#1e3a8a'
  ctx.lineWidth = 2
  ctx.strokeRect(wristX - 20, wristY - 12, 40, 24)

  // Draw finger function - draws joints and connecting lines
  const drawFinger = (points: { x: number; y: number }[], startX: number, startY: number, color: string) => {
    if (points.length === 0) return

    ctx.strokeStyle = color
    ctx.fillStyle = color
    ctx.lineWidth = 5

    // Draw connecting lines
    ctx.beginPath()
    ctx.moveTo(startX, startY)
    for (const point of points) {
      const x = point.x * width
      const y = point.y * height
      ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Draw joints (larger dots)
    for (const point of points) {
      const x = point.x * width
      const y = point.y * height
      ctx.beginPath()
      ctx.arc(x, y, 8, 0, Math.PI * 2)
      ctx.fill()
      // Add highlight
      ctx.fillStyle = '#ffffff'
      ctx.beginPath()
      ctx.arc(x - 2, y - 2, 3, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = color
    }
  }

  // Draw fingers with different colors for visibility
  drawFinger(pose.thumb, wristX - 15, wristY, '#ef4444')  // Red for thumb
  drawFinger(pose.index, wristX - 2, wristY - 8, '#3b82f6')  // Blue for index
  drawFinger(pose.middle, wristX, wristY, '#10b981')  // Green for middle
  drawFinger(pose.ring, wristX, wristY + 8, '#f59e0b')  // Orange for ring
  drawFinger(pose.pinky, wristX + 15, wristY, '#8b5cf6')  // Purple for pinky

  // Draw palm outline for better visualization
  ctx.strokeStyle = '#60a5fa'
  ctx.fillStyle = 'rgba(59, 130, 246, 0.2)'
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.moveTo(wristX - 15, wristY)
  ctx.lineTo(wristX + 15, wristY)
  ctx.lineTo(pose.index[0]?.x ? pose.index[0].x * width : wristX, pose.index[0]?.y ? pose.index[0].y * height : wristY - 10)
  ctx.lineTo(pose.pinky[0]?.x ? pose.pinky[0].x * width : wristX + 15, pose.pinky[0]?.y ? pose.pinky[0].y * height : wristY)
  ctx.closePath()
  ctx.fill()
  ctx.stroke()
}

export function SignLanguageAnimator({ text, isPlaying, onComplete }: SignLanguageAnimatorProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [currentSign, setCurrentSign] = useState<string>('')
  const [signSequence, setSignSequence] = useState<string[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)

  // Parse text into sign sequence
  useEffect(() => {
    if (!text.trim()) {
      setSignSequence([])
      return
    }

    const words = text.toLowerCase().trim().split(/\s+/)
    const sequence: string[] = []

    for (const word of words) {
      if (signMappings[word]) {
        sequence.push(signMappings[word])
      } else {
        for (const char of word) {
          if (signMappings[char]) {
            sequence.push(signMappings[char])
          }
        }
      }
    }

    setSignSequence(sequence)
    setCurrentIndex(0)
  }, [text])

  // Draw hand animation
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width || 400
    const height = canvas.height || 300

    const drawFrame = () => {
      // Always draw something - use current sign if available, otherwise default
      const gesture = (currentSign && signGestures[currentSign])
        ? signGestures[currentSign]
        : signGestures['DEFAULT']
      drawHand(ctx, gesture, width, height)
    }

    // Initial draw
    drawFrame()

    // Set up animation loop
    const interval = setInterval(drawFrame, 50) // Update every 50ms for smooth animation

    return () => clearInterval(interval)
  }, [isAnimating, currentSign])

  // Ensure canvas is properly sized on mount
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const container = canvas.parentElement
    if (container) {
      const updateSize = () => {
        const rect = container.getBoundingClientRect()
        canvas.width = rect.width
        canvas.height = 300
      }

      updateSize()
      const resizeObserver = new ResizeObserver(updateSize)
      resizeObserver.observe(container)

      return () => resizeObserver.disconnect()
    }
  }, [])

  // Play sign sequence
  useEffect(() => {
    if (!isPlaying || signSequence.length === 0) {
      setIsAnimating(false)
      return
    }

    setIsAnimating(true)
    setCurrentIndex(0)
    setCurrentSign(signSequence[0] || 'DEFAULT')

    let index = 0
    const timeouts: NodeJS.Timeout[] = []

    const playNextSign = () => {
      if (index >= signSequence.length) {
        setIsAnimating(false)
        setCurrentSign('DEFAULT')
        if (onComplete) onComplete()
        return
      }

      setCurrentSign(signSequence[index])
      setCurrentIndex(index)
      index++

      const timeout = setTimeout(playNextSign, 2000) // 2 seconds per sign
      timeouts.push(timeout)
    }

    playNextSign()

    return () => {
      timeouts.forEach(clearTimeout)
    }
  }, [isPlaying, signSequence, onComplete])

  // Set canvas size
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const resize = () => {
      const container = canvas.parentElement
      if (container) {
        canvas.width = container.clientWidth
        canvas.height = 300
      }
    }

    resize()
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
  }, [])

  if (signSequence.length === 0) {
    return (
      <div style={{
        width: '100%',
        height: 300,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: '1px solid #ccc',
        borderRadius: 8,
        background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
        position: 'relative'
      }}>
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%'
          }}
        />
        <div className="small" style={{ opacity: 0.7, zIndex: 1, background: 'rgba(255,255,255,0.8)', padding: '8px 16px', borderRadius: 8 }}>
          Enter text to see sign language animation
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{
        width: '100%',
        height: 300,
        border: '2px solid #3b82f6',
        borderRadius: 8,
        overflow: 'hidden',
        background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        position: 'relative'
      }}>
        <canvas
          ref={canvasRef}
          style={{
            display: 'block',
            width: '100%',
            height: '100%'
          }}
        />
        {currentSign && (
          <div style={{
            position: 'absolute',
            top: 10,
            left: 10,
            background: 'rgba(59, 130, 246, 0.9)',
            color: 'white',
            padding: '6px 12px',
            borderRadius: 6,
            fontWeight: 'bold',
            fontSize: '18px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
          }}>
            {currentSign}
          </div>
        )}
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <div className="small" style={{ fontWeight: 600 }}>
          Showing: <b style={{ color: '#1e40af' }}>{currentSign || 'Ready'}</b>
        </div>
        <div className="small" style={{ opacity: 0.6 }}>
          ({currentIndex + 1}/{signSequence.length})
        </div>
        {isAnimating && (
          <div className="small" style={{ color: '#16a34a', display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{
              display: 'inline-block',
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: '#16a34a',
              animation: 'pulse 1s infinite'
            }} /> Animating
          </div>
        )}
      </div>
      <div className="small" style={{ opacity: 0.7, padding: 8, background: '#f0f9ff', borderRadius: 6 }}>
        Sequence: <strong>{signSequence.join(' → ')}</strong>
      </div>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  )
}
