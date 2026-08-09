import React, { useEffect, useRef, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { SignLanguageAnimator } from '../SignLanguageAnimator'

export function Bridge() {
  const [connected, setConnected] = useState(false)
  const [status, setStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')
  const [lang, setLang] = useState('en-US')
  const [input, setInput] = useState('')
  const [autoConvert, setAutoConvert] = useState(true)
  const [history, setHistory] = useState<{ type: string; text: string; time: string }[]>([])
  const [signLabel, setSignLabel] = useState('No Hand')
  const [signScore, setSignScore] = useState(0)
  const [numHands, setNumHands] = useState(0)
  const [camOn, setCamOn] = useState(false)
  const [animText, setAnimText] = useState('')
  const [animating, setAnimating] = useState(false)
  const [turbo, setTurbo] = useState(false)
  const [bright, setBright] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const recRef = useRef<any>(null)
  const captureRef = useRef<number | null>(null)
  const lastSignRef = useRef<string | null>(null)
  const lastVoiceRef = useRef('')

  const wsBase = (import.meta as any).env.VITE_WS_URL || 'ws://localhost:8000'
  const url = connected ? `${wsBase}/ws` : null

  const addHistory = (type: string, text: string) => setHistory(h => [...h, { type, text, time: new Date().toLocaleTimeString() }])
  const sendWS = (obj: any) => wsRef.current?.readyState === WebSocket.OPEN && wsRef.current.send(JSON.stringify(obj))

  useEffect(() => {
    if (!url) return
    setStatus('connecting')
    const ws = new WebSocket(url); wsRef.current = ws
    ws.onopen = () => { setStatus('connected'); ws.send(JSON.stringify({ type: 'health' })) }
    ws.onclose = () => setStatus('disconnected')
    ws.onerror = () => setStatus('disconnected')
    ws.onmessage = evt => {
      try {
        const msg = JSON.parse(evt.data)
        if (msg.type === 'sign_status') {
          setNumHands(msg.payload?.num_hands || 0)
          const sign = msg.payload?.sign
          if (sign) {
            setSignLabel(sign.label || 'Unknown'); setSignScore(sign.score || 0)
            if (autoConvert && sign.score >= 0.8 && sign.label !== lastSignRef.current && !['Unknown', 'No Hand'].includes(sign.label)) {
              lastSignRef.current = sign.label
              setInput(sign.label); setAnimText(sign.label); setAnimating(true); addHistory('Sign', sign.label)
            }
          }
        } else if (msg.type === 'stt_ack') addHistory('Voice', msg.payload?.text || '')
        else if (msg.type === 'text_echo') addHistory('Text', msg.payload?.text || '')
      } catch {}
    }
    return () => ws.close()
  }, [url, autoConvert])

  useEffect(() => {
    if (!camOn) return
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      .then(s => { if (videoRef.current) { videoRef.current.srcObject = s; videoRef.current.play() } })
      .catch(e => { console.error(e); setCamOn(false) })
    return () => { const v = videoRef.current; (v?.srcObject as MediaStream | null)?.getTracks().forEach(t => t.stop()); if (v) v.srcObject = null }
  }, [camOn])

  useEffect(() => {
    if (camOn && status === 'connected' && videoRef.current && canvasRef.current) {
      const vid = videoRef.current, cvs = canvasRef.current
      cvs.width = 640; cvs.height = 480
      const ctx = cvs.getContext('2d')!
      captureRef.current = window.setInterval(() => {
        ctx.drawImage(vid, 0, 0, 640, 480)
        sendWS({ type: 'sign_frame', payload: { image: cvs.toDataURL('image/jpeg', 0.8) } })
      }, turbo ? 150 : 300)
    } else if (captureRef.current) window.clearInterval(captureRef.current)
    return () => { if (captureRef.current) window.clearInterval(captureRef.current) }
  }, [camOn, status, turbo])

  const speak = (text: string) => { const u = new SpeechSynthesisUtterance(text); u.lang = lang; speechSynthesis.speak(u); sendWS({ type: 'tts_text', payload: { text } }) }

  const sttSupported = typeof (window as any).SpeechRecognition !== 'undefined' || typeof (window as any).webkitSpeechRecognition !== 'undefined'
  const startSTT = () => {
    if (!sttSupported) return
    const Rec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    const rec = new Rec(); rec.continuous = true; rec.interimResults = true; rec.lang = lang
    rec.onresult = (e: any) => {
      for (let i = e.resultIndex; i < e.results.length; i++) if (e.results[i].isFinal) {
        const txt = e.results[i][0].transcript.trim()
        if (txt && txt !== lastVoiceRef.current) {
          lastVoiceRef.current = txt
          sendWS({ type: 'stt_result', payload: { text: txt } })
          setInput(txt); setAnimText(txt); setAnimating(true)
          if (autoConvert) speak(txt)
        }
      }
    }
    rec.start(); recRef.current = rec
  }
  const stopSTT = () => { recRef.current?.stop?.(); recRef.current = null }

  const downloadLog = () => {
    const blob = new Blob([history.map(h => `${h.time} [${h.type}] ${h.text}`).join('\n')], { type: 'text/plain' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'communibridge-log.txt'; a.click()
  }
  const goFullscreen = () => {
    const el = document.querySelector('.cam-box') as HTMLElement | null
    el?.requestFullscreen?.().catch(() => {})
  }
  const logIcon = (t: string) => t === 'Sign' ? ['li-green', '🖐'] : t === 'Voice' ? ['li-purple', '🎤'] : ['li-blue', '💬']

  return (
    <div className="bridge">
      <aside className="sidebar">
        <NavLink to="/" className={({ isActive }) => 'side-btn' + (isActive ? ' active' : '')} title="Home">🏠</NavLink>
        <NavLink to="/bridge" className={({ isActive }) => 'side-btn' + (isActive ? ' active' : '')} title="Live Bridge">🌉</NavLink>
        <NavLink to="/animator" className={({ isActive }) => 'side-btn' + (isActive ? ' active' : '')} title="Animator">🖐️</NavLink>
        <NavLink to="/translate" className={({ isActive }) => 'side-btn' + (isActive ? ' active' : '')} title="Translate">🈺</NavLink>
        <NavLink to="/about" className={({ isActive }) => 'side-btn' + (isActive ? ' active' : '')} title="About">ℹ️</NavLink>
        <div className="side-user">👤<br />Adarsh</div>
      </aside>

      <div className="bridge-main">
        <div className="bridge-head">
          <h1>Live Communication Bridge</h1>
          <span className={status === 'connected' ? 'pill-ok' : 'pill-err'}>● {status}</span>
          <select className="select" style={{ width: 'auto' }} value={lang} onChange={e => setLang(e.target.value)}>
            <option value="en-US">🌐 English</option><option value="hi-IN">🌐 Hindi</option>
            <option value="es-ES">🌐 Spanish</option><option value="fr-FR">🌐 French</option>
          </select>
          <button className="btn btn-grad" onClick={() => setConnected(v => !v)}>{connected ? 'Disconnect' : 'Connect'}</button>
        </div>

        <div className="panels">
          <div className="panel">
            <div className="panel-head"><span className="p-num">1</span><h2>Sign Recognition</h2><span className="live">● LIVE</span></div>
            <div className="cam-box">
              <video ref={videoRef} muted playsInline style={{ display: camOn ? 'block' : 'none', filter: bright ? 'brightness(1.5)' : 'none' }} />
              {!camOn && <div style={{ display: 'grid', placeItems: 'center', height: '100%', fontSize: 60 }}>🖐️</div>}
              <span className="corner c-tl" /><span className="corner c-tr" /><span className="corner c-bl" /><span className="corner c-br" />
            </div>
            <div className="detect">{signLabel} ({(signScore * 100).toFixed(0)}%)</div>
            <div className="conf-row"><span>Confidence Score</span><span>{(signScore * 100).toFixed(0)}%</span></div>
            <div className="bar"><i style={{ width: `${signScore * 100}%` }} /></div>
            <div className="cam-tools">
              <button className="tool" title="Camera on/off" onClick={() => setCamOn(v => !v)}>📷</button>
              <button className="tool" title="Turbo mode" onClick={() => setTurbo(v => !v)} style={turbo ? { color: '#fbbf24', borderColor: 'rgba(251,191,36,.5)' } : {}}>⚡</button>
              <button className="tool" title="Brightness" onClick={() => setBright(v => !v)} style={bright ? { color: '#fde047' } : {}}>☀️</button>
              <button className="tool" title="Fullscreen" onClick={goFullscreen}>⛶</button>
            </div>
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>

          <div className="panel">
            <div className="panel-head"><span className="p-num">2</span><h2>Voice & Text</h2></div>
            <div className="label">Microphone</div>
            <div className="row">
              <button className="btn btn-grad" onClick={startSTT} disabled={!sttSupported}>🎤 Start Mic</button>
              <button className="btn btn-dark" onClick={stopSTT}>⏹ Stop Mic</button>
            </div>
            <label className="checkbox"><input type="checkbox" checked={autoConvert} onChange={e => setAutoConvert(e.target.checked)} /> Auto-convert speech to text</label>
            <div className="label">Text Input</div>
            <textarea className="textarea" rows={3} maxLength={500} placeholder="Type your message..." value={input} onChange={e => { setInput(e.target.value); setAnimText(e.target.value); setAnimating(true) }} />
            <div className="count">{input.length}/500</div>
            <div className="label">Text to Speech</div>
            <button className="btn btn-grad" style={{ width: '100%', justifyContent: 'center' }} onClick={() => { speak(input); addHistory('Text', input) }}>🔊 Speak</button>
          </div>

          <div className="panel">
            <div className="panel-head"><span className="p-num">3</span><h2>Communication Log</h2>
              <button className="link-btn" style={{ marginLeft: 'auto' }} onClick={() => setHistory([])}>Clear All 🗑</button>
            </div>
            <div className="log">
              {history.length === 0 && <div className="small">No activity yet…</div>}
              {history.map((h, i) => { const [cls, ic] = logIcon(h.type); return (
                <div key={i} className="log-item">
                  <span className={`log-ico ${cls}`}>{ic}</span>
                  <div>
                    <div className="small" style={{ color: cls === 'li-green' ? 'var(--green)' : cls === 'li-purple' ? '#a78bfa' : '#60a5fa', fontWeight: 700 }}>
                      {h.type === 'Sign' ? 'System (Sign Detected)' : `You (${h.type})`}
                    </div>
                    <div>{h.text}</div>
                  </div>
                  <span className="log-time">{h.time}</span>
                </div>
              )})}
            </div>
            <button className="btn btn-dark" style={{ width: '100%', justifyContent: 'center', marginTop: 12 }} onClick={downloadLog}>⬇ Download Log</button>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head"><span className="p-num">🖐</span><h2>Sign Animation</h2></div>
          <div className="anim-grid">
            <div>
              <div className="small">Current Sign</div>
              <div style={{ color: '#a78bfa', fontWeight: 800, fontSize: 18, margin: '4px 0 10px' }}>{animText || '—'}</div>
              <div className="small" style={{ color: 'var(--green)' }}>● {animating ? 'Playing' : 'Idle'}</div>
            </div>
            <SignLanguageAnimator text={animText} isPlaying={animating} onComplete={() => setAnimating(false)} />
            <div style={{ display: 'grid', gap: 10 }}>
              <select className="select"><option>View: Real Hands</option></select>
              <select className="select"><option>Speed: Normal</option></select>
              <div className="row" style={{ justifyContent: 'center' }}>
                <button className="icon-btn">⏮</button>
                <button className="icon-btn" onClick={() => setAnimating(v => !v)}>{animating ? '⏸' : '▶'}</button>
                <button className="icon-btn">⏭</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}