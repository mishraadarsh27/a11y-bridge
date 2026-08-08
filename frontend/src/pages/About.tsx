import React from 'react'
import { Link } from 'react-router-dom'

const gestures = [
  { icon: '🖐️', label: 'Hello' }, { icon: '✊', label: 'Fist' }, { icon: '☝️', label: 'You' },
  { icon: '✌️', label: 'Peace' }, { icon: '👍', label: 'Thumbs Up' },
]
const stack = [
  { icon: '⚛️', label: 'React' }, { icon: '⚡', label: 'FastAPI' }, { icon: '🖐️', label: 'MediaPipe' },
  { icon: '🎤', label: 'Web Speech API' }, { icon: '🦙', label: 'Ollama' },
]

export function About() {
  return (
    <div className="page-wrap" style={{ maxWidth: 1000 }}>
      <div className="page-head">
        <h1 className="page-title">About <span className="grad-text">CommuniBridge</span></h1>
        <p className="page-sub">Empowering conversations. Breaking barriers.</p>
        <div className="sec-underline" style={{ margin: '16px auto 0' }} />
      </div>

      <div style={{ marginTop: 32 }}>
        <div className="a-card">
          <div className="a-ico">🎯</div>
          <div>
            <h2 className="a-title">Mission</h2>
            <div className="a-underline" />
            <p style={{ color: 'var(--muted)', margin: 0, maxWidth: 640 }}>
              CommuniBridge aims to create an accessible, real-time communication platform that connects deaf, mute, and speech-impaired individuals with everyone, everywhere. We believe that communication is a right, not a privilege.
            </p>
          </div>
        </div>

        <div className="a-card">
          <div className="a-ico" style={{ background: 'rgba(59,130,246,.15)', borderColor: 'rgba(59,130,246,.4)' }}>🖐️</div>
          <div>
            <h2 className="a-title">Supported Gestures</h2>
            <div className="a-underline" />
            <div className="gestures">{gestures.map(g => <div key={g.label} className="gesture"><span className="gi">{g.icon}</span>{g.label}</div>)}</div>
          </div>
        </div>

        <div className="a-card">
          <div className="a-ico" style={{ background: 'rgba(34,211,238,.12)', borderColor: 'rgba(34,211,238,.4)' }}>🛠️</div>
          <div>
            <h2 className="a-title">Tech Stack</h2>
            <div className="a-underline" />
            <div className="stack">{stack.map(s => <div key={s.label}><span className="si">{s.icon}</span>{s.label}</div>)}</div>
          </div>
        </div>

        <div className="a-card cta">
          <div className="row">
            <div className="a-ico">👥</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>Together, let's build a world<br /><span className="grad-text">where everyone can be heard.</span></div>
          </div>
          <Link to="/bridge" className="btn btn-grad">Join the Movement →</Link>
        </div>
      </div>
    </div>
  )
}