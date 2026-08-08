import React from 'react'
import { Link } from 'react-router-dom'

const features = [
  { icon: '🤟', title: 'Sign to Text & Speech', desc: 'Real-time sign language detection and conversion to text and speech.' },
  { icon: '🎤', title: 'Speech to Sign & Text', desc: 'Convert spoken words into sign language animations and readable text.' },
  { icon: '🖐️', title: 'Sign Animator', desc: 'AI-powered avatar animations for smooth sign language delivery.' },
  { icon: '🌐', title: 'Multi-language Support', desc: 'Communicate across languages with smart translation and localization.' },
  { icon: '💬', title: 'Conversation History', desc: 'Save, review, and manage your conversations securely.' },
  { icon: '⚡', title: 'Real-time & Offline', desc: 'Lightning-fast responses with offline mode for seamless communication.' },
]
const steps = [
  { n: '1', cls: 'n1', icon: '🎥', title: 'Choose Your Input', desc: 'Use sign language, speech, or text to start the conversation.' },
  { n: '2', cls: 'n2', icon: '🧠', title: 'AI Processes Instantly', desc: 'Our AI understands and converts your input in real-time.' },
  { n: '3', cls: 'n3', icon: '🖥️', title: 'Get Connected', desc: 'Receive output as sign animation, speech, or text — instantly.' },
]

export function Home() {
  return (
    <div>
      <section className="hero">
        <div className="hero-inner">
          <div>
            <div className="hero-badge">👥 Accessibility-First Communication</div>
            <h1>Breaking Barriers Between<br /><span className="grad-text">Sign, Speech & Text</span></h1>
            <p>CommuniBridge is a real-time, multilingual platform that connects deaf, mute and speech-impaired individuals with everyone.</p>
            <div className="row">
              <Link to="/bridge" className="btn btn-grad">🚀 Launch Live Bridge</Link>
              <Link to="/animator" className="btn btn-ghost">▶ Try Sign Animator</Link>
            </div>
          </div>
          <div className="listen-card">
            <div className="wave">{Array.from({ length: 28 }).map((_, i) => <i key={i} style={{ animationDelay: `${i * 0.06}s` }} />)}</div>
            <div style={{ fontWeight: 700 }}>Listening...</div>
            <div className="small" style={{ color: 'var(--green)', marginTop: 6 }}>● Converting to Sign</div>
          </div>
        </div>
      </section>

      <section className="features">
        <div className="feature-grid">
          {features.map(f => <div key={f.title} className="f-card"><span className="f-icon">{f.icon}</span><h3>{f.title}</h3><p>{f.desc}</p><span className="f-arrow">→</span></div>)}
        </div>
      </section>

      <section className="how">
        <h2 className="sec-title">How It Works</h2>
        <div className="sec-underline" />
        <div className="steps">
          {steps.map(s => (
            <div key={s.n} className="step">
              <span className={`step-num ${s.cls}`}>{s.n}</span>
              <div><h3 style={{ margin: '0 0 6px', fontSize: 16 }}>{s.icon} {s.title}</h3><p style={{ margin: 0, color: 'var(--muted)', fontSize: 13 }}>{s.desc}</p></div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}