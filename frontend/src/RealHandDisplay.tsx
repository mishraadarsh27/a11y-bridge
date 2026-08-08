import React, { useEffect, useState } from 'react'
import { HAND_SIGNS, HandSign } from './handSigns'
import { Hand3D } from './Hand3D'

export function RealHandDisplay({ letter, showDescription = true }: { letter: string; showDescription?: boolean }) {
  const sign: HandSign = HAND_SIGNS[letter.toUpperCase()] || HAND_SIGNS.A
  const [failed, setFailed] = useState(false)

  useEffect(() => { setFailed(false) }, [letter])

  return (
    <div style={{
      width: '100%', minHeight: 420,
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      borderRadius: 16, padding: 24,
      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16,
      border: '2px solid rgba(96,165,250,.3)', position: 'relative'
    }}>
      {/* Letter badge */}
      <div style={{
        position: 'absolute', top: 16, left: 16,
        background: 'linear-gradient(135deg,#3b82f6,#06b6d4)', color: '#fff',
        fontWeight: 800, fontSize: 28, width: 56, height: 56, borderRadius: 14,
        display: 'grid', placeItems: 'center', boxShadow: '0 8px 24px rgba(59,130,246,.4)'
      }}>{sign.letter}</div>

      <div style={{
        position: 'absolute', top: 16, right: 16,
        background: 'rgba(30,41,59,.8)', padding: '8px 14px', borderRadius: 8,
        color: '#94a3b8', fontSize: 12, fontWeight: 600
      }}>{failed ? '🎮 3D Hand (photo missing)' : '📸 Real Human Hand'}</div>

      {/* Hand display */}
      <div style={{
        width: '100%', maxWidth: 420, height: 300, marginTop: 36,
        background: 'rgba(15,23,42,.7)', borderRadius: 14,
        border: '2px solid rgba(96,165,250,.4)', overflow: 'hidden',
        display: 'grid', placeItems: 'center',
        boxShadow: '0 12px 32px rgba(96,165,250,.25)'
      }}>
        {!failed ? (
          <img
            key={sign.letter}
            src={sign.rightHand}
            alt={`ASL sign ${sign.letter}`}
            onError={() => setFailed(true)}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        ) : (
          <Hand3D letter={sign.letter} />
        )}
      </div>

      {failed && (
        <div className="small" style={{ color: '#fbbf24', textAlign: 'center' }}>
          ⚠️ Real photo nahi mili — 3D hand dikh raha hai. Photos add karne ke liye: <b>frontend/public/hand-signs/{sign.letter}.png</b>
        </div>
      )}

      {showDescription && (
        <div style={{
          background: 'rgba(30,41,59,.6)', padding: '14px 22px', borderRadius: 12,
          border: '1px solid rgba(148,163,184,.2)', textAlign: 'center', maxWidth: 500
        }}>
          <div className="small" style={{ marginBottom: 6, fontWeight: 600 }}>How to sign "{sign.letter}"</div>
          <div style={{ color: '#e2e8f0', fontSize: 15, fontWeight: 500 }}>{sign.description}</div>
        </div>
      )}
    </div>
  )
}