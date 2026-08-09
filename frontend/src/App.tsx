import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { LangProvider } from './LangContext'
import { Navbar } from './components/Navbar'
import { Home } from './pages/Home'
import { Bridge } from './pages/Bridge'
import { Animator } from './pages/Animator'
import { Translate } from './pages/Translate'
import { About } from './pages/About'

export function App() {
  return (
    <LangProvider>
      <div className="app-shell">
        <Navbar />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/bridge" element={<Bridge />} />
            <Route path="/animator" element={<Animator />} />
            <Route path="/translate" element={<Translate />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </LangProvider>
  )
}