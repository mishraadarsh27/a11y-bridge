import React, { useState } from 'react'
import { SignLanguageAnimator } from '../SignLanguageAnimator'

export function Animator() {
  const [text, setText] = useState('hello')
  const [playing, setPlaying] = useState(false)
  return (
    <div className="page-wrap">
      <div className="page-head">
        <h1 className="page-title grad-text">Sign Language Animator</h1>
        <p className="page-sub">Animate and visualize sign language gestures in real-time.</p>
      </div>
      <div className="tool-card">
        <div className="row" style={{ marginBottom: 16 }}>
          <input className="input" value={text} onChange={e => setText(e.target.value)} placeholder="Type text to sign…" />
          <button className="btn btn-grad" onClick={() => setPlaying(true)}>▶ Play</button>
          <button className="btn btn-dark" onClick={() => setPlaying(false)}>⏹ Stop</button>
        </div>
        <div className="glow"><div className="glow-inner">
          <SignLanguageAnimator text={text} isPlaying={playing} onComplete={() => setPlaying(false)} />
        </div></div>
      </div>
    </div>
  )
}