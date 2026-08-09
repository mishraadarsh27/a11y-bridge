import React, { useEffect, useState } from 'react'
import { NavLink, Link, useLocation } from 'react-router-dom'
import { Logo } from '../Logo'
import { useLang } from '../LangContext'

type Theme = 'light' | 'dark' | 'extra-dark'

export function Navbar() {
  const loc = useLocation()
  const { lang, setLang, t } = useLang()
  const [theme, setTheme] = useState<Theme>((localStorage.getItem('theme') as Theme) || 'dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const links = [
    { to: '/', label: t('home') }, { to: '/bridge', label: t('bridge') },
    { to: '/animator', label: t('animator') }, { to: '/translate', label: t('translate') },
    { to: '/about', label: t('about') },
  ]

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <Link to="/" className="brand">
          <Logo />
          <span><span className="brand-name">CommuniBridge</span><span className="brand-tag">Bridging Communication. Building Inclusion.</span></span>
        </Link>
        <nav className="nav-links">
          {links.map(l => (
            <NavLink key={l.to} to={l.to} end={l.to === '/'} className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>{l.label}</NavLink>
          ))}
        </nav>
        <div className="nav-actions">
          {loc.pathname === '/' && <Link to="/bridge" className="btn btn-grad" style={{ padding: '10px 18px' }}>{t('getStarted')}</Link>}
          <select className="select" style={{ width: 'auto', padding: '8px 10px' }} value={lang} onChange={e => setLang(e.target.value)} title="Language">
            <option value="en">EN</option>
            <option value="hi">हिंदी</option>
          </select>
          <div className="theme-seg" role="group" aria-label="Theme">
            <button className={theme === 'light' ? 'active' : ''} onClick={() => setTheme('light')} title="Light">☀️</button>
            <button className={theme === 'dark' ? 'active' : ''} onClick={() => setTheme('dark')} title="Dark">🌙</button>
            <button className={theme === 'extra-dark' ? 'active' : ''} onClick={() => setTheme('extra-dark')} title="Extra Dark">🌑</button>
          </div>
          <div className="avatar">👤<span className="dot" /></div>
        </div>
      </div>
    </header>
  )
}