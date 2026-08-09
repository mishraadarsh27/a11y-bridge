import React from 'react'
import { Link } from 'react-router-dom'
import { useLang } from '../LangContext'

const ICONS = ['🤟', '', '️', '', '💬', '']

export function Home() {
  const { t } = useLang()
  const features = [1, 2, 3, 4, 5, 6].map(n => ({ icon: ICONS[n - 1], title: t(`f${n}t`), desc: t(`f${n}d`) }))
  const steps = [
    { n: '1', cls: 'n1', icon: '🎥', title: t('s1t'), desc: t('s1d') },
    { n: '2', cls: 'n2', icon: '🧠', title: t('s2t'), desc: t('s2d') },
    { n: '3', cls: 'n3', icon: '🖥️', title: t('s3t'), desc: t('s3d') },
  ]

  return (
    <div>
      <section className="hero">
        <div className="hero-inner">
          <div>
            <div className="hero-badge">👥 {t('heroBadge')}</div>
            <h1>{t('heroT1')}<br /><span className="grad-text">{t('heroT2')}</span></h1>
            <p>{t('heroSub')}</p>
            <div className="row">
              <Link to="/bridge" className="btn btn-grad">{t('launch')}</Link>
              <Link to="/animator" className="btn btn-ghost">{t('tryAnim')}</Link>
            </div>
          </div>
          <div className="hero-media">
            <img className="hero-img" src="/hero.png" alt="Person using sign language" onError={e => { e.currentTarget.style.display = 'none' }} />
            <div className="listen-card">
              <div className="wave">{Array.from({ length: 24 }).map((_, i) => <i key={i} style={{ animationDelay: `${i * 0.06}s` }} />)}</div>
              <div style={{ fontWeight: 700 }}>{t('listening')}</div>
              <div className="small" style={{ color: 'var(--green)', marginTop: 6 }}>{t('converting')}</div>
            </div>
          </div>
        </div>
      </section>

      <section className="features">
        <div className="feature-grid">
          {features.map(f => <div key={f.title} className="f-card"><span className="f-icon">{f.icon}</span><h3>{f.title}</h3><p>{f.desc}</p><span className="f-arrow">→</span></div>)}
        </div>
      </section>

      <section className="how">
        <h2 className="sec-title">{t('howTitle')}</h2>
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