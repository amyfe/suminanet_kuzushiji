import { useState } from 'react'
import './LoadingIndicator.css'

export default function LoadingIndicator({ label }) {
  return (
    <div className="loading-indicator">
      <span className="loading-brush" aria-hidden="true">
        <span className="loading-brush-dot" />
        <span className="loading-brush-dot" />
        <span className="loading-brush-dot" />
      </span>
      <span className="loading-label">{label}</span>
    </div>
  )
}

// TranscriptionLoader.html is a self-contained bundled page (React + assets)
// that replaces its own document on load. It must run in its own document,
// so we embed it via an iframe pointing at the file served verbatim from
// public/. No sandbox: the bundle needs allow-scripts and same-origin blob
// URLs, and it is our own first-party file.
export function Animation({ label, className, title }) {
  const [failed, setFailed] = useState(false)

  if (failed) {
    return <LoadingIndicator label={label} />
  }

  return (
    <div className={`animation-section ${className || ''}`}>
      <iframe
        className="animation-frame"
        src="/TranscriptionLoader.html"
        title={title || label || 'Loading animation'}
        loading="lazy"
        frameBorder="0"
        onError={() => setFailed(true)}
      />
      {label && <span className="loading-label">{label}</span>}
    </div>
  )
}
