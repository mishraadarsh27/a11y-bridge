import React, { useEffect, useState } from 'react'
import { RealHandDisplay } from './RealHandDisplay'
import { Hand3D } from './Hand3D'

export function SignLanguageAnimator({ text, isPlaying, onComplete }: { text: string; isPlaying: boolean; onComplete?: () => void }) {
  const [seq, setSeq] = useState<string[]>([])
  const [idx, setIdx] = useState(0)
  const [view, setView] = useState<'real' | '3d'>('real')
  const [paused, setPaused] = useState(false)
  const [speed, setSpeed] = useState(2000)

  useEffect(() => {
    const letters = (text || '').toUpperCase().replace(/[^A-Z]/g, '').split('')
    setSeq(letters)
    setIdx(0)
  }, [text])

  useEffect(() => {
    if (isPlaying) { setIdx(0); setPaused(false) }
  }, [isPlaying])

  useEffect(() => {
    if (!isPlaying || paused || seq.length === 0) return
    const t = setInterval(() => {
      setIdx(i => {
        if (i + 1 >= seq.length) { clearInterval(t); onComplete?.(); return i }
        return i + 1
      })
    }, speed)
    return () => clearInterval(t)
  }, [isPlaying, paused, seq, speed])

  const currentLetter = seq[idx] || (text || '').toUpperCase().replace(/[^A-Z]/g, '').charAt(0) || 'A'
  const prev = () => { setPaused(true); setIdx(i => Math.max(0, i - 1)) }
  const next = () => { setPaused(true); setIdx(i => Math.min(seq.length - 1, i + 1)) }

  return (
    <div style={{ position: 'relative' }}>
      {/* Display: Real Hands ya 3D */}
      {view === 'real' ? (
        <RealHandDisplay letter={currentLetter} />
      ) : (
        <div style={{ borderRadius: 16, border: '2px solid rgba(167,139,250,.4)', overflow: 'hidden', boxShadow: '0 0 30px rgba(139,92,246,.25)' }}>
          <Hand3D letter={currentLetter} />
        </div>
      )}

      {/* Controls Bar */}
      <div className="row" style={{ justifyContent: 'center', marginTop: 14 }}>
        <button className="icon-btn" onClick={prev} title="Previous sign">⏮</button>
        <button className="icon-btn" onClick={() => setPaused(p => !p)} title="Play / Pause">{paused ? '▶' : '⏸'}</button>
        <button className="icon-btn" onClick={next} title="Next sign">⏭</button>
        <select className="select" style={{ width: 'auto' }} value={speed} onChange={e => setSpeed(Number(e.target.value))}>
          <option value={3000}>Speed: Slow</option>
          <option value={2000}>Speed: Normal</option>
          <option value={1000}>Speed: Fast</option>
        </select>
        <select className="select" style={{ width: 'auto' }} value={view} onChange={e => setView(e.target.value as 'real' | '3d')}>
          <option value="real">View: Real Hands 📸</option>
          <option value="3d">View: 3D Model 🎮</option>
        </select>
      </div>

      {/* Sequence */}
      {seq.length > 0 && (
        <div className="small" style={{ textAlign: 'center', marginTop: 12, padding: '12px 16px', background: 'rgba(30,41,59,.6)', borderRadius: 8, border: '1px solid rgba(148,163,184,.2)' }}>
          <div style={{ marginBottom: 8, color: '#94a3b8', fontWeight: 600 }}>Sequence ({idx + 1}/{seq.length})</div>
          <div style={{ color: '#60a5fa', fontWeight: 700, fontSize: 16 }}>
            {seq.map((l, i) => (
              <span
                key={i}
                onClick={() => { setPaused(true); setIdx(i) }}
                style={{
                  color: i === idx ? '#22d3ee' : '#60a5fa',
                  fontWeight: i === idx ? 800 : 600,
                  fontSize: i === idx ? 20 : 16,
                  margin: '0 4px',
                  cursor: 'pointer'
                }}
              >{l}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}