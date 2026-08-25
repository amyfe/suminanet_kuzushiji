import { useState, useRef } from 'react'
import { scoreTier } from '../utils/text'
import i18next from 'i18next'
import CharTooltipContent from './CharTooltipContent'

export default function ImageWithBoxes({ imageUrl, chars, onSelectAlternate, onToggleDeleteChar, onReplace, boxesVisible = true, onFile, zoomed = false }) {
  const imgRef = useRef(null)
  const [imgDisplaySize, setImgDisplaySize] = useState(null)
  const [hoveredCharIdx, setHoveredCharIdx] = useState(null)
  const [dragOver, setDragOver] = useState(false)
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

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) onFile?.(file)
  }

  return (
    <div
      className={`image-overlay-container${dragOver ? ' drag-over' : ''}${zoomed ? ' zoomed' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
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
            className={`char-box char-box--${scoreTier(c.score)}${boxesVisible ? '' : ' char-box--hidden'}${c.deleted ? ' char-box--deleted' : ''}`}
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
                  deleted={c.deleted}
                  onSelect={(altChar) => onSelectAlternate?.(i, altChar)}
                  onToggleDelete={onToggleDeleteChar ? () => onToggleDeleteChar(i) : undefined}
                />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
