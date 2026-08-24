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

// TranscriptionLoader.html is a plain, self-contained static page (inline
// SVG/CSS/JS, no React, no build step) served verbatim from public/. It
// needs its own document (own <style>/<script> scope) to run its looping
// animation independently of the React tree, so it is embedded via an
// iframe rather than rendered inline. No sandbox: it is our own
// first-party file.
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
