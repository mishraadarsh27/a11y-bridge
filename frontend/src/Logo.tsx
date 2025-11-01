import React from 'react'

export function Logo({ className = 'logo' }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      width="28"
      height="28"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path d="M3 15a9 9 0 0 1 18 0" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <path d="M7 15a5 5 0 0 1 10 0" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="6" cy="15" r="1.6" fill="currentColor" />
      <circle cx="18" cy="15" r="1.6" fill="currentColor" />
    </svg>
  )
}
