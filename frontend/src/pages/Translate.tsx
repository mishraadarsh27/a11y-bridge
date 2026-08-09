import React, { useState } from 'react'

const API_BASE = (import.meta as any).env.VITE_API_URL || 'http://localhost:8000'

export function Translate() {
  const [text, setText] = useState('Hello, how are you?')
  const [source, setSource] = useState('auto')
  const [target, setTarget] = useState('es')
  const [out, setOut] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const langs = [
    { code: 'auto', name: 'Auto-detect' }, { code: 'en', name: 'English' }, { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' }, { code: 'de', name: 'German' }, { code: 'hi', name: 'Hindi' },
    { code: 'zh', name: 'Chinese' }, { code: 'ja', name: 'Japanese' },
  ]

  const doTranslate = async () => {
    setLoading(true); setError(null); setOut('')
    try {
      // Pehle backend (Ollama) try karo
      const res = await fetch(`${API_BASE}/translate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source_lang: source, target_lang: target }),
      })
      if (!res.ok) throw new Error('backend')
      const data = await res.json()
      setOut(data?.translated ?? '')
    } catch {
      try {
        // FREE fallback: MyMemory translation API (bina Ollama ke bhi chalega)
        const src = source === 'auto' ? 'en' : source
        const r = await fetch(`https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=${src}|${target}`)
        const j = await r.json()
        const tr = j?.responseData?.translatedText
        if (!tr) throw new Error('failed')
        setOut(tr)
      } catch { setError('Translation failed — internet/backend check karo') }
    } finally { setLoading(false) }
  }
  const copy = () => { navigator.clipboard?.writeText(out); setCopied(true); setTimeout(() => setCopied(false), 1500) }

  return (
    <div className="page-wrap">
      <div className="page-head">
        <h1 className="page-title grad-text">LLM Translate</h1>
        <p className="page-sub">Multilingual translation — Ollama + free cloud fallback.</p>
      </div>
      <div className="tool-card">
        <div className="row" style={{ marginBottom: 14 }}>
          <select className="select" style={{ width: 'auto' }} value={source} onChange={e => setSource(e.target.value)}>{langs.map(l => <option key={l.code} value={l.code}>{l.name}</option>)}</select>
          <span style={{ fontSize: 20 }}>→</span>
          <select className="select" style={{ width: 'auto' }} value={target} onChange={e => setTarget(e.target.value)}>{langs.filter(l => l.code !== 'auto').map(l => <option key={l.code} value={l.code}>{l.name}</option>)}</select>
          <button className="btn btn-grad" style={{ marginLeft: 'auto' }} onClick={doTranslate} disabled={loading}>{loading ? 'Translating…' : '🈺 Translate'}</button>
        </div>
        <textarea className="textarea" rows={6} maxLength={1000} value={text} onChange={e => setText(e.target.value)} />
        <div className="count">{text.length} / 1000</div>
        {error && <div className="small" style={{ color: 'var(--red)', marginTop: 8 }}>Error: {error}</div>}
        <div className="out-box">
          <div className="grad-text" style={{ fontWeight: 800, marginBottom: 8 }}>Output</div>
          <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontSize: 18 }}>{out || '—'}</div>
            <button className="icon-btn" onClick={copy} title="Copy">{copied ? '✅' : '📋'}</button>
          </div>
        </div>
      </div>
    </div>
  )
}