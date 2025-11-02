import React, { useEffect, useRef, useState } from 'react'
import { Logo } from './Logo'
import { SignLanguageAnimator } from './SignLanguageAnimator'

function TranslateWidget() {
  const [text, setText] = useState('Hello, how are you?')
  const [source, setSource] = useState('auto')
  const [target, setTarget] = useState('hi')
  const [model, setModel] = useState('llama3.2:latest')
  const [out, setOut] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const langs = [
    { code: 'auto', name: 'Auto-detect' },
    { code: 'en', name: 'English' },
    { code: 'hi', name: 'Hindi' },
    { code: 'ur', name: 'Urdu' },
    { code: 'pa', name: 'Punjabi' },
    { code: 'bn', name: 'Bengali' },
    { code: 'ta', name: 'Tamil' },
    { code: 'te', name: 'Telugu' },
    { code: 'mr', name: 'Marathi' },
    { code: 'gu', name: 'Gujarati' },
    { code: 'kn', name: 'Kannada' },
    { code: 'ml', name: 'Malayalam' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'ar', name: 'Arabic' },
    { code: 'zh', name: 'Chinese' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
  ]

  const doTranslate = async () => {
    setLoading(true)
    setError(null)
    setOut('')
    try {
      const base = (import.meta as any).env.VITE_API_URL || ''
      const res = await fetch(`${base || ''}/translate`.replace(/([^:]?)\/\//g, '$1/'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source_lang: source, target_lang: target, model }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setOut(data?.translated ?? '')
    } catch (e: any) {
      setError(e?.message || 'Failed to translate')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="section" style={{ display: 'grid', gap: 8 }}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <select className="select" value={source} onChange={(e) => setSource(e.target.value)}>
          {langs.map((l) => <option key={l.code} value={l.code}>{l.name}</option>)}
        </select>
        <span className="small" aria-hidden>→</span>
        <select className="select" value={target} onChange={(e) => setTarget(e.target.value)}>
          {langs.filter(l => l.code !== 'auto').map((l) => <option key={l.code} value={l.code}>{l.name}</option>)}
        </select>
        <input className="input" style={{ minWidth: 160 }} value={model} onChange={(e) => setModel(e.target.value)} placeholder="Model (e.g., llama3.2:latest)" />
        <button className="button primary" onClick={doTranslate} disabled={loading}>
          {loading ? 'Translating…' : 'Translate'}
        </button>
      </div>
      <textarea className="input" rows={3} value={text} onChange={(e) => setText(e.target.value)} placeholder="Enter text" />
      {error && <div className="small" style={{ color: '#ef4444' }}>Error: {error}</div>}
      <div className="section" style={{ background: 'var(--panel)', padding: 8, borderRadius: 6 }}>
        <div className="small" style={{ opacity: 0.8 }}>Output</div>
        <div aria-live="polite">{out || <span className="small" style={{ opacity: 0.6 }}>(translation will appear here)</span>}</div>
      </div>
    </div>
  )
}

function useWebSocket(url: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const [messages, setMessages] = useState<string[]>([])
  const [status, setStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')

  useEffect(() => {
    console.log('WebSocket: useEffect triggered, url =', url)
    if (!url) {
      console.log('WebSocket: URL is null, not connecting')
      return
    }
    console.log('WebSocket: Connecting to', url)
    setStatus('connecting')
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket: Connected!')
      setStatus('connected')
      // Probe backend identity
      setTimeout(() => {
        try {
          if (ws.readyState === WebSocket.OPEN) {
            console.log('WebSocket: Sending health check')
            ws.send(JSON.stringify({ type: 'health' }))
          }
        } catch (e) {
          console.error('WebSocket: Failed to send health', e)
        }
      }, 100)

      // Keep-alive ping every 30 seconds
      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          try {
            ws.send(JSON.stringify({ type: 'health' }))
          } catch {}
        } else {
          clearInterval(pingInterval)
        }
      }, 30000)

      // Clear interval on close
      ws.addEventListener('close', () => clearInterval(pingInterval))
    }
    ws.onmessage = (evt) => {
      console.log('WebSocket: Message received:', evt.data)
      setMessages((m) => [...m, String(evt.data)])
    }
    ws.onerror = (err) => {
      console.error('WebSocket: Error', err)
      setStatus('disconnected')
    }
    ws.onclose = (evt) => {
      console.log('WebSocket: Closed', evt.code, evt.reason)
      setStatus('disconnected')
    }

    return () => {
      console.log('WebSocket: Cleanup - closing connection')
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
      wsRef.current = null
    }
  }, [url])

  const send = (text: string) => wsRef.current?.readyState === WebSocket.OPEN && wsRef.current.send(text)

  return { messages, status, send, wsRef }
}

function SignStatus({ messages }: { messages: string[] }) {
  const [numHands, setNumHands] = useState<number>(0)
  const [gestures, setGestures] = useState<string[]>([])
  const [handsInfo, setHandsInfo] = useState<Array<{label?: string; score?: number | null}>>([])
  const [sign, setSign] = useState<{ id?: number | null; label?: string; score?: number | null } | null>(null)
  const [labelMap, setLabelMap] = useState<string[]>(() => {
    try {
      const savedRaw = localStorage.getItem('sign_labels')
      if (savedRaw) {
        const saved = JSON.parse(savedRaw)
        if (Array.isArray(saved)) {
          // sanitize and cap length
          return saved.map((x) => String(x ?? '')).slice(0, 64)
        }
      }
    } catch {}
    // Default mapping requested: 1->yes, 2->no, 3..28->A..Z
    const arr = new Array(64).fill('') as string[]
    arr[1] = 'yes'
    arr[2] = 'no'
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for (let i = 0; i < 26; i++) arr[3 + i] = letters[i]
    try { localStorage.setItem('sign_labels', JSON.stringify(arr)) } catch {}
    return arr
  })
  const [error, setError] = useState<string | null>(null)

  const editLabels = () => {
    const raw = window.prompt('Enter labels as JSON array or newline-separated (index -> label)')
    if (!raw) return
    let arr: string[] = []
    try {
      const j = JSON.parse(raw)
      if (Array.isArray(j)) arr = j.map((x) => String(x))
    } catch {
      arr = raw.split(/\r?\n|,/).map((s) => s.trim()).filter(Boolean)
    }
    setLabelMap(arr)
    try { localStorage.setItem('sign_labels', JSON.stringify(arr)) } catch {}
  }

  useEffect(() => {
    if (!messages.length) return
    const last = messages[messages.length - 1]
    try {
      const j = JSON.parse(last)
      if (j?.type === 'sign_status') {
        setNumHands(Number(j?.payload?.num_hands || 0))
        setGestures(Array.isArray(j?.payload?.gestures) ? j.payload.gestures : [])
        setHandsInfo(Array.isArray(j?.payload?.hands) ? j.payload.hands : [])
        setError(typeof j?.payload?.error === 'string' ? j.payload.error : null)
        if (j?.payload?.sign) {
          const s = j.payload.sign
          let label: string | undefined = s?.label
          const id: number | undefined = typeof s?.id === 'number' ? s.id : (typeof label === 'string' && /^\d+$/.test(label) ? parseInt(label, 10) : undefined)
          if (id != null && labelMap[id]) label = labelMap[id]
          setSign({ id: id ?? null, label, score: typeof s?.score === 'number' ? s.score : null })
        }
      }
    } catch {}
  }, [messages, labelMap])

  return (
    <div style={{ padding: 8, border: '1px solid #ccc', borderRadius: 8, minWidth: 200 }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>Hand Detection</div>
      <div>Hands: <b>{numHands}</b></div>
      <div style={{ marginTop: 6 }}>
        <span style={{
          display: 'inline-block',
          width: 10, height: 10, borderRadius: 9999,
          background: numHands > 0 ? '#16a34a' : '#ef4444'
        }} /> {numHands > 0 ? 'Detected' : 'No hands'}
      </div>
      {sign?.label && (
        <div className="small" style={{ marginTop: 6 }}>
          Sign: <b>{sign.label}</b>{typeof sign.score === 'number' ? ` (${sign.score.toFixed(2)})` : ''}{typeof sign.id === 'number' ? ` [id: ${sign.id}]` : ''}
        </div>
      )}
      {handsInfo.length > 0 && (
        <div className="small" style={{ marginTop: 6 }}>
          {handsInfo.map((h, i) => <div key={i}>{h.label} {h.score != null ? `(${h.score.toFixed(2)})` : ''}</div>)}
        </div>
      )}
      {gestures.length > 0 && (
        <div className="small" style={{ marginTop: 6 }}>Gestures: {gestures.join(', ')}</div>
      )}
      <div className="small" style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
        <button className="button" onClick={editLabels} style={{ height: 26, padding: '0 8px' }}>Set Labels</button>
        <button className="button" onClick={() => {
          try { localStorage.removeItem('sign_labels') } catch {}
          // force default mapping
          const arr = new Array(64).fill('') as string[]
          arr[1] = 'yes'; arr[2] = 'no'
          const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
          for (let i = 0; i < 26; i++) arr[3 + i] = letters[i]
          setLabelMap(arr)
          try { localStorage.setItem('sign_labels', JSON.stringify(arr)) } catch {}
        }} style={{ height: 26, padding: '0 8px' }}>Reset Labels</button>
        {labelMap.length > 0 && <span>labels: {labelMap.length}</span>}
      </div>
      {error && (
        <div className="small" style={{ marginTop: 6, color: '#ef4444' }}>Error: {error}</div>
      )}
    </div>
  )
}

type Theme = 'system' | 'light' | 'dark'

export function App() {
  const [input, setInput] = useState('hello')
  const [connected, setConnected] = useState(false) // manually connect
  const [lang, setLang] = useState('en-US')
  const [signAnimationText, setSignAnimationText] = useState('')
  const [isAnimating, setIsAnimating] = useState(false)
  const [autoConvert, setAutoConvert] = useState(true) // Master toggle for automatic conversion
  const lastProcessedSignRef = useRef<string | null>(null)
  const lastProcessedTextRef = useRef<string>('')
  const lastProcessedVoiceRef = useRef<string>('')
  // Build URLs from environment for prod; fall back to localhost in dev
  const API_URL = (import.meta as any).env.VITE_API_URL || ''
  const WS_URL = (import.meta as any).env.VITE_WS_URL || ''
  const wsBase = WS_URL || (location.protocol === 'https:' ? `wss://${location.host}` : `ws://${location.hostname}:8000`)
  // Force Sign-MNIST letter pipeline by default (A–Y static letters)
  const url = connected ? `${wsBase.replace(/\/$/, '')}/ws?sign_model=signmnist` : null
  const { messages, status, send, wsRef } = useWebSocket(url)

  // Theme management
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem('theme') as Theme) || 'system')
  useEffect(() => {
    const root = document.documentElement
    const meta = document.querySelector('meta[name="color-scheme"]') as HTMLMetaElement | null
    if (theme === 'system') {
      root.removeAttribute('data-theme')
      if (meta) meta.content = 'light dark'
    } else {
      root.setAttribute('data-theme', theme)
      if (meta) meta.content = theme
    }
    localStorage.setItem('theme', theme)
  }, [theme])

  // Browser TTS (with anti-loop gating)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const speak = (text: string) => {
    const u = new SpeechSynthesisUtterance(text)
    u.lang = lang
    u.onstart = () => setIsSpeaking(true)
    u.onend = () => setIsSpeaking(false)
    u.onerror = () => setIsSpeaking(false)
    window.speechSynthesis.speak(u)
    // notify backend for logging/ack
    send(JSON.stringify({ type: 'tts_text', payload: { text } }))
  }

  // Browser STT (Chrome/Edge: Web Speech API)
  const recRef = useRef<any>(null)
  const sttSupported = typeof (window as any).SpeechRecognition !== 'undefined' || typeof (window as any).webkitSpeechRecognition !== 'undefined'
  const [autoSpeak, setAutoSpeak] = useState<boolean>(false)
  const [autoSpeakSign, setAutoSpeakSign] = useState<boolean>(false)
  const signHistRef = useRef<Array<{ label: string; score: number; t: number }>>([])
  const lastSpokenSignRef = useRef<string | null>(null)
  const lastSpokenAtRef = useRef<number>(0)
  const startSTT = () => {
    if (!sttSupported) return
    // stop any previous session
    recRef.current?.stop?.()
    const Rec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    const rec = new Rec()
    rec.continuous = true
    rec.interimResults = true
    rec.lang = lang
    rec.onresult = (e: any) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i]
        if (res.isFinal) {
          const txt = res[0].transcript.trim()
          // Ignore results while our own TTS is speaking to avoid feedback loops
          if (isSpeaking) continue
          send(JSON.stringify({ type: 'stt_result', payload: { text: txt } }))

          // Auto conversion: Voice → Text + Sign Animation
          if (autoConvert && txt && txt !== lastProcessedVoiceRef.current) {
            lastProcessedVoiceRef.current = txt

            // 1. Update text field
            if (txt !== lastProcessedTextRef.current) {
              setInput(txt)
              lastProcessedTextRef.current = txt
            }

            // 2. Update sign animation
            if (txt !== signAnimationText) {
              setSignAnimationText(txt)
              setIsAnimating(true)
            }

            // 3. Speak (only if autoSpeak is enabled, to avoid double-speaking)
            if (autoSpeak && txt) {
              speak(txt)
            }
          } else if (autoSpeak && txt) {
            // Legacy behavior if autoConvert is off
            speak(txt)
          }
        }
      }
    }
    rec.start()
    recRef.current = rec
  }
  const stopSTT = () => {
    recRef.current?.stop?.()
    recRef.current = null
  }

  // Sign capture (camera -> WS frames)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [camOn, setCamOn] = useState(false)
  const [sending, setSending] = useState(false)
  const [collecting, setCollecting] = useState(false)
  const [collectLabel, setCollectLabel] = useState('A')
  const captureInterval = useRef<number | null>(null)

  useEffect(() => {
    if (!camOn) return
    const startCam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
      } catch (e) {
        console.error('Camera error', e)
        setCamOn(false)
      }
    }
    startCam()
    return () => {
      const v = videoRef.current
      const s = v && (v.srcObject as MediaStream | null)
      s?.getTracks().forEach((t) => t.stop())
      if (v) v.srcObject = null
    }
  }, [camOn])

  const startSending = () => {
    if (!videoRef.current || !canvasRef.current) return
    const vid = videoRef.current
    const cvs = canvasRef.current
    cvs.width = vid.videoWidth || 640
    cvs.height = vid.videoHeight || 480
    const ctx = cvs.getContext('2d')!
    setSending(true)
    captureInterval.current = window.setInterval(() => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
      ctx.drawImage(vid, 0, 0, cvs.width, cvs.height)
      const dataUrl = cvs.toDataURL('image/jpeg', 0.8)
      try {
        const payload: any = { image: dataUrl }
        if (collecting && collectLabel.trim()) payload.collect = { label: collectLabel.trim() }
        wsRef.current.send(JSON.stringify({ type: 'sign_frame', payload }))
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn('WS send error', e)
      }
    }, 150)
  }
  const stopSending = () => {
    if (captureInterval.current) window.clearInterval(captureInterval.current)
    captureInterval.current = null
    setSending(false)
  }

  // Auto-start sending when camera is on and WS is connected
  useEffect(() => {
    if (camOn && status === 'connected' && !sending) {
      startSending()
    }
  }, [camOn, status, sending])

  // Auto conversion: Sign → Text + Voice + Animation
  useEffect(() => {
    if (!messages.length || !autoConvert) return
    const raw = messages[messages.length - 1]
    try {
      const j = JSON.parse(typeof raw === 'string' && raw.startsWith('echo:') ? raw.slice(5) : raw)
      if (j?.type === 'sign_status' && j?.payload?.sign) {
        const label = String(j.payload.sign.label ?? '')
        const score = typeof j.payload.sign.score === 'number' ? j.payload.sign.score : 0
        // Increased threshold from 0.5 to 0.6 for better accuracy
        if (!label || label === 'unknown' || score < 0.6) return
        const now = Date.now()
        signHistRef.current.push({ label, score, t: now })
        if (signHistRef.current.length > 30) signHistRef.current.shift() // Increased buffer

        // Improved smoothing: use last 10 samples with score >= 0.65
        const recent = signHistRef.current.slice(-10).filter((x) => x.score >= 0.65)
        if (recent.length < 5) return // Need at least 5 confident samples

        const counts: Record<string, number> = {}
        const scores: Record<string, number[]> = {}
        for (const r of recent) {
          counts[r.label] = (counts[r.label] || 0) + 1
          if (!scores[r.label]) scores[r.label] = []
          scores[r.label].push(r.score)
        }

        // Find most common label with at least 60% occurrence
        let stable: string | null = null
        let maxc = 0
        for (const [k, v] of Object.entries(counts)) {
          if (v > maxc && v >= Math.ceil(recent.length * 0.6)) {
            maxc = v
            stable = k
          }
        }

        if (stable && maxc >= 5) { // Increased from 3 to 5 for better stability
          const avgScore = scores[stable] ? scores[stable].reduce((a, b) => a + b, 0) / scores[stable].length : score
          const since = now - (lastSpokenAtRef.current || 0)

          // Only process if it's a new sign or enough time has passed (increased to 2 seconds)
          if (stable !== lastProcessedSignRef.current || since > 2000) {
            // 1. Update text field
            if (stable !== lastProcessedTextRef.current) {
              setInput(stable)
              lastProcessedTextRef.current = stable
            }

            // 2. Update sign animation
            if (stable !== signAnimationText) {
              setSignAnimationText(stable)
              setIsAnimating(true)
            }

            // 3. Speak (voice output) - only if score is high enough
            if (!isSpeaking && avgScore >= 0.65 && (stable !== lastSpokenSignRef.current || since > 2000)) {
              speak(stable)
              lastSpokenSignRef.current = stable
              lastSpokenAtRef.current = now
            }

            lastProcessedSignRef.current = stable
          }
        }
      }
    } catch {}
  }, [messages, autoConvert, isSpeaking, signAnimationText])

  // Auto conversion: Text → Sign Animation + Voice
  useEffect(() => {
    if (!autoConvert || !input.trim()) return
    // Only process if text has changed
    if (input === lastProcessedTextRef.current) return

    // Prevent loop: if this text just came from sign recognition, skip voice/sign update
    // (it's already been handled by sign → text conversion)
    if (input === lastProcessedSignRef.current) {
      lastProcessedTextRef.current = input
      return
    }

    // Prevent loop: if this text just came from voice recognition
    if (input === lastProcessedVoiceRef.current) {
      lastProcessedTextRef.current = input
      return
    }

    lastProcessedTextRef.current = input

    // 1. Update sign animation
    if (input !== signAnimationText) {
      setSignAnimationText(input)
      setIsAnimating(true)
    }

    // 2. Speak (voice output) - but only if not already speaking
    if (!isSpeaking && input !== lastProcessedVoiceRef.current) {
      speak(input)
      lastProcessedVoiceRef.current = input
    }
  }, [input, autoConvert, isSpeaking, signAnimationText])

  // Auto conversion: Voice → Text + Sign Animation
  useEffect(() => {
    // This is handled in the STT onresult callback
    // We'll update it to also trigger sign animation
  }, [])

  // Legacy: Auto TTS for recognized signs (for backward compatibility with checkbox)
  useEffect(() => {
    if (autoSpeakSign && !autoConvert) {
      // Keep the old behavior if autoConvert is off but autoSpeakSign is on
      // This effect handles that case
    }
  }, [autoSpeakSign, autoConvert])

  return (
    <div className="container" style={{ fontFamily: 'system-ui, sans-serif', lineHeight: 1.5 }}>
      <div className="header">
        <div className="brand">
          <Logo />
          <h1>VaaniSetu (CommuniBridge)</h1>
        </div>
        <div className="right">
          <div className="segmented" role="group" aria-label="Theme">
            <button className={theme === 'system' ? 'active' : ''} onClick={() => setTheme('system')}>Sys</button>
            <button className={theme === 'light' ? 'active' : ''} onClick={() => setTheme('light')}>Light</button>
            <button className={theme === 'dark' ? 'active' : ''} onClick={() => setTheme('dark')}>Dark</button>
          </div>
          <span className={`badge ${status === 'connected' ? 'ok' : 'err'}`}>{status}</span>
          <select className="select" value={lang} onChange={(e) => setLang(e.target.value)}>
            <option value="en-US">English (US)</option>
            <option value="en-GB">English (UK)</option>
            <option value="hi-IN">Hindi (hi-IN)</option>
            <option value="bn-IN">Bengali (bn-IN)</option>
            <option value="ta-IN">Tamil (ta-IN)</option>
            <option value="te-IN">Telugu (te-IN)</option>
            <option value="mr-IN">Marathi (mr-IN)</option>
            <option value="gu-IN">Gujarati (gu-IN)</option>
            <option value="kn-IN">Kannada (kn-IN)</option>
            <option value="pa-IN">Punjabi (pa-IN)</option>
            <option value="ml-IN">Malayalam (ml-IN)</option>
            <option value="or-IN">Odia (or-IN)</option>
          </select>
          <button className="button" onClick={() => {
            console.log('Button clicked, connected was:', connected)
            setConnected((v) => !v)
          }}>
            {connected ? 'Disconnect' : 'Connect'}
          </button>
        </div>
      </div>

      <div className="grid">
        {/* Sign Panel */}
        <div className="panel">
          <h2>Sign</h2>
          <div className="section">
            <button className="button" onClick={() => setCamOn((v) => !v)}>{camOn ? 'Stop Camera' : 'Start Camera'}</button>
            <button className="button primary" onClick={sending ? stopSending : startSending} disabled={!camOn || status !== 'connected'}>
              {sending ? 'Stop Sending' : 'Start Sending'}
            </button>
          </div>
          <div className="section" style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <input className="input" style={{ width: 80 }} value={collectLabel} onChange={(e) => setCollectLabel(e.target.value)} placeholder="Label" />
            <button className="button" onClick={() => setCollecting((v) => !v)} disabled={!camOn || status !== 'connected'}>{collecting ? 'Stop Collect' : 'Start Collect'}</button>
            <button className="button" onClick={async () => { try { const base = (import.meta as any).env.VITE_API_URL || ''; await fetch(`${base || ''}/sign/static_train`.replace(/([^:]?)\/\//g, '$1/'), { method: 'POST' }) } catch {} }}>Train Static</button>
          </div>
          <video ref={videoRef} className="video" muted playsInline></video>
          <div style={{ marginTop: 8 }}>
            <SignStatus messages={messages} />
          </div>
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>

        {/* Voice Panel */}
        <div className="panel">
          <h2>Voice</h2>
          <div className="section">
            <button className="button" onClick={startSTT} disabled={!sttSupported}>Start Mic</button>
            <button className="button" onClick={stopSTT}>Stop Mic</button>
            {!sttSupported && <span className="small">STT not supported in this browser</span>}
          </div>
          <div className="section" style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <label className="small" htmlFor="auto-convert" style={{ fontWeight: 'bold' }}>
              <input id="auto-convert" type="checkbox" checked={autoConvert} onChange={(e) => setAutoConvert(e.target.checked)} />
              <span style={{ color: autoConvert ? '#16a34a' : '#666' }}>Auto-Convert All Modes</span>
            </label>
            <label className="small" htmlFor="auto-speak">
              <input id="auto-speak" type="checkbox" checked={autoSpeak} onChange={(e) => setAutoSpeak(e.target.checked)} /> Auto-speak STT
            </label>
            <label className="small" htmlFor="auto-speak-sign">
              <input id="auto-speak-sign" type="checkbox" checked={autoSpeakSign} onChange={(e) => setAutoSpeakSign(e.target.checked)} /> Auto-speak Sign
            </label>
          </div>
          {autoConvert && (
            <div className="small" style={{ marginTop: 4, padding: 6, background: '#e0f2fe', borderRadius: 4, color: '#0369a1' }}>
              ✓ Automatic conversion enabled: Sign ↔ Text ↔ Voice
            </div>
          )}
          <div className="section">
            <input className="input" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Text to speak" />
            <button className="button" onClick={() => speak(input)}>Speak</button>
          </div>
        </div>

        {/* Text Panel */}
        <div className="panel">
          <h2>Text</h2>
          <div className="section">
            <input className="input" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Type a message" />
            <button className="button primary" onClick={() => send(JSON.stringify({ type: 'text', payload: { text: input } }))} disabled={status !== 'connected'}>
              Send
            </button>
          </div>
          <div className="scroll">
            {messages.map((m, i) => {
              try {
                let raw = m
                // Handle servers that prefix with 'echo:' and then JSON
                if (typeof raw === 'string' && raw.startsWith('echo:')) raw = raw.slice(5)
                const j = JSON.parse(raw)
                const label = j?.type ?? 'msg'
                const body = j?.payload?.text ? j.payload.text : JSON.stringify(j.payload ?? {})
                return <div className="log-item" key={i}><b>{label}:</b> {body}</div>
              } catch {
                return <div className="log-item" key={i}>{m}</div>
              }
            })}
          </div>
        </div>

        {/* Translate Panel (LLM via Ollama) */}
        <div className="panel">
          <h2>Translate (LLM)</h2>
          <TranslateWidget />
        </div>

        {/* Sign Language Animation Panel */}
        <div className="panel">
          <h2>Sign Language Animation</h2>
          <div className="section">
            <input
              className="input"
              value={signAnimationText}
              onChange={(e) => setSignAnimationText(e.target.value)}
              placeholder="Enter text to animate (e.g., 'hello', 'yes', 'thank you')"
            />
            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
              <button
                className="button primary"
                onClick={() => {
                  if (signAnimationText.trim()) {
                    setIsAnimating(true)
                  }
                }}
                disabled={!signAnimationText.trim() || isAnimating}
              >
                {isAnimating ? 'Animating...' : 'Animate Sign'}
              </button>
              <button
                className="button"
                onClick={() => {
                  setIsAnimating(false)
                  setSignAnimationText('')
                }}
              >
                Stop
              </button>
            </div>
          </div>
          <div className="section">
            <SignLanguageAnimator
              text={signAnimationText}
              isPlaying={isAnimating}
              onComplete={() => {
                setIsAnimating(false)
              }}
            />
          </div>
          <div className="small" style={{ marginTop: 8, opacity: 0.7 }}>
            <div><strong>Supported words:</strong> hello, hi, yes, no, thank you, please, sorry, goodbye, how are you</div>
            <div style={{ marginTop: 4 }}><strong>Fingerspelling:</strong> Any text will be fingerspelled letter by letter (A-Z)</div>
          </div>
        </div>
      </div>
    </div>
  )}
