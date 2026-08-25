import { useState, useRef } from 'react'
import { scoreTier } from '../utils/text'
import i18next from 'i18next'
import CharTooltipContent from './CharTooltipContent'

export default function ImageWithBoxes({ imageUrl, chars, onSelectAlternate, onReplace, boxesVisible = true }) {
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
            className={`char-box char-box--${scoreTier(c.score)}${boxesVisible ? '' : ' char-box--hidden'}`}
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
                <CharTooltipContent
                  char={c.char}
                  score={c.score}
                  alternates={c.alternates}
                  onSelect={(altChar) => onSelectAlternate?.(i, altChar)}
                />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
