import { useState, useRef } from 'react'
import { scoreTier, formatScore } from '../utils/text'
import i18next from 'i18next'

export default function ImageWithBoxes({ imageUrl, chars, onSelectAlternate, onReplace }) {
  const imgRef = useRef(null)
  const [imgDisplaySize, setImgDisplaySize] = useState(null)
  const [hoveredCharIdx, setHoveredCharIdx] = useState(null)
  const t = i18next.t.bind(i18next)

  function onImageLoad() {
    const img = imgRef.current
    setImgDisplaySize({
      w: img.clientWidth,
      h: img.clientHeight,
      natW: img.naturalWidth,
      natH: img.naturalHeight,
    })
  }

  return (
    <div className="image-overlay-container">
      <img
        ref={imgRef}
        src={imageUrl}
        className={`overlay-image${onReplace ? ' replaceable' : ''}`}
        alt={t('imageWithBoxes.overlayAlt')}
        onLoad={onImageLoad}
        onClick={onReplace}
      />
      {onReplace && <span className="replace-hint">{t('workspace.replaceImageHint')}</span>}
      {imgDisplaySize && chars.map((c, i) => {
        const [x1, y1, x2, y2] = c.box
        const leftPct = (x1 / imgDisplaySize.natW) * 100
        const topPct = (y1 / imgDisplaySize.natH) * 100
        const widthPct = ((x2 - x1) / imgDisplaySize.natW) * 100
        const heightPct = ((y2 - y1) / imgDisplaySize.natH) * 100

        return (
          <div
            key={i}
            className={`char-box char-box--${scoreTier(c.score)}`}
            style={{
              left: `${leftPct}%`,
              top: `${topPct}%`,
              width: `${widthPct}%`,
              height: `${heightPct}%`,
            }}
            onMouseEnter={() => setHoveredCharIdx(i)}
            onMouseLeave={() => setHoveredCharIdx(null)}
          >
            {hoveredCharIdx === i && (
              <div className="char-tooltip">
                {c.alternates?.length > 0 ? (
                  <div className="tooltip-alternates">
                    {c.alternates.slice(0, 3).map((alt, ai) => (
                      <button
                        type="button"
                        // Backend guarantees all candidates in c.alternates are
                        // distinct chars (vocab is a char<->id bijection, and the
                        // chosen id is excluded when collecting runner-ups), so
                        // char equality is a safe way to mark the active pick.
                        className={`tooltip-alt-char${alt.char === c.char ? ' tooltip-alt-char--active' : ''}`}
                        key={ai}
                        onClick={(e) => {
                          e.stopPropagation()
                          onSelectAlternate?.(i, alt.char)
                        }}
                        aria-label={t('imageWithBoxes.useAlternate', { char: alt.char })}
                      >
                        {alt.char} <span className="tooltip-alt-score">{formatScore(alt.prob)}</span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="tooltip-main">
                    <span className="tooltip-char">{c.char}</span>
                    <span className="tooltip-score">{formatScore(c.score)}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
