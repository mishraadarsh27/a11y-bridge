import React from 'react'
import { NavLink, Link, useLocation } from 'react-router-dom'
import { Logo } from '../Logo'

const links = [
  { to: '/', label: 'Home' }, { to: '/bridge', label: 'Live Bridge' },
  { to: '/animator', label: 'Animator' }, { to: '/translate', label: 'Translate' },
  { to: '/about', label: 'About' },
]

export function Navbar() {
  const loc = useLocation()
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
          {loc.pathname === '/' && <Link to="/bridge" className="btn btn-grad" style={{ padding: '10px 18px' }}>Get Started</Link>}
          <button className="icon-btn" title="Theme">🌙</button>
          <div className="avatar">👤<span className="dot" /></div>
        </div>
      </div>
    </header>
  )
}