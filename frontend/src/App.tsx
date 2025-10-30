import React, { useEffect, useRef, useState } from 'react'

function useWebSocket(url: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const [messages, setMessages] = useState<string[]>([])
  const [status, setStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')

  useEffect(() => {
    if (!url) return
    setStatus('connecting')
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => setStatus('connected')
    ws.onmessage = (evt) => setMessages((m) => [...m, String(evt.data)])
    ws.onerror = () => setStatus('disconnected')
    ws.onclose = () => setStatus('disconnected')

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [url])

  const send = (text: string) => wsRef.current?.readyState === WebSocket.OPEN && wsRef.current.send(text)

  return { messages, status, send }
}

export function App() {
  const [input, setInput] = useState('hello')
  const [connected, setConnected] = useState(true) // auto-connect on load
  const [lang, setLang] = useState('en-US')
  const url = connected ? `ws://${location.hostname}:8000/ws` : null
  const { messages, status, send } = useWebSocket(url)

  // Browser TTS
  const speak = (text: string) => {
    const u = new SpeechSynthesisUtterance(text)
    u.lang = lang
    window.speechSynthesis.speak(u)
    // notify backend for logging/ack
    send(JSON.stringify({ type: 'tts_text', payload: { text } }))
  }

  // Browser STT (Chrome: webkitSpeechRecognition)
  const recRef = useRef<any>(null)
  const sttSupported = typeof (window as any).webkitSpeechRecognition !== 'undefined'
  const startSTT = () => {
    if (!sttSupported) return
    const Rec = (window as any).webkitSpeechRecognition
    const rec = new Rec()
    rec.continuous = true
    rec.interimResults = true
    rec.lang = lang
    rec.onresult = (e: any) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i]
        if (res.isFinal) {
          const txt = res[0].transcript.trim()
          send(JSON.stringify({ type: 'stt_result', payload: { text: txt } }))
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

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: 16, lineHeight: 1.5 }}>
      <h1>A11y Bridge</h1>
      <p>Status: <b>{status}</b></p>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
        <button onClick={() => setConnected((v) => !v)}>
          {connected ? 'Disconnect' : 'Connect'}
        </button>
        <label>
          Lang:
          <select value={lang} onChange={(e) => setLang(e.target.value)} style={{ marginLeft: 4 }}>
            <option value="en-US">en-US</option>
            <option value="en-GB">en-GB</option>
          </select>
        </label>
      </div>

      <h2>Text ➜ WS / Voice</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message"
        />
        <button
          onClick={() => send(JSON.stringify({ type: 'text', payload: { text: input } }))}
          disabled={status !== 'connected'}
        >
          Send WS
        </button>
        <button onClick={() => speak(input)}>
          Speak
        </button>
      </div>

      <h2>Voice ➜ Text (browser STT)</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <button onClick={startSTT} disabled={!sttSupported}>Start Mic</button>
        <button onClick={stopSTT}>Stop Mic</button>
        {!sttSupported && <span>STT not supported in this browser</span>}
      </div>

      <h2>Messages</h2>
      <ul>
        {messages.map((m, i) => {
          try {
            const j = JSON.parse(m)
            if (j?.payload?.text) return <li key={i}>{j.type}: {j.payload.text}</li>
            return <li key={i}>{j.type ?? 'msg'}: {JSON.stringify(j.payload ?? {})}</li>
          } catch {
            return <li key={i}>{m}</li>
          }
        })}
      </ul>
    </div>
  )
}
