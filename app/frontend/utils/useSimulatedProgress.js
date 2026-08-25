import { useEffect, useState } from 'react'

// Frontend-only estimated progress. No backend instrumentation exists to
// drive this for real (transcribe is one opaque blocking inference call,
// translate is one non-streaming ~30s LLM call) — so this climbs smoothly
// toward a ceiling on an asymptotic curve calibrated by `estimatedMs` (a
// rough typical-duration guess, not a hard deadline), and deliberately
// never claims 100% itself: completion is signaled by the real result
// replacing the loading UI, not by this hook.
const CEILING = 92

export default function useSimulatedProgress(active, estimatedMs) {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    if (!active) {
      setProgress(0)
      return
    }
    const start = Date.now()
    const id = setInterval(() => {
      const elapsed = Date.now() - start
      setProgress(CEILING * (1 - Math.exp(-elapsed / estimatedMs)))
    }, 150)
    return () => clearInterval(id)
  }, [active, estimatedMs])

  return progress
}
