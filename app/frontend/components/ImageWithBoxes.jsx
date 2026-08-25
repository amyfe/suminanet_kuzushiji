import { useState, useRef } from 'react'
import { scoreTier } from '../utils/text'
import i18next from 'i18next'
import CharTooltipContent from './CharTooltipContent'

export default function ImageWithBoxes({ imageUrl, chars, onSelectAlternate, onToggleDeleteChar, onReplace, boxesVisible = true, onFile, zoomed = false, hoveredCharIdx, onHoverChar }) {
  const imgRef = useRef(null)
  const [imgDisplaySize, setImgDisplaySize] = useState(null)
  // Local, not the shared hoveredCharIdx prop: this panel's own tooltip
  // should only appear when the mouse is actually over one of its own
  // boxes, not just because the OTHER panel reported a hover. The shared
  // prop still drives this panel's highlight styling below, so hovering
  // either panel lights up both -- only the tooltip itself stays local.
  const [localHoveredIdx, setLocalHoveredIdx] = useState(null)
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
            className={`char-box char-box--${scoreTier(c.score)}${boxesVisible ? '' : ' char-box--hidden'}${c.deleted ? ' char-box--deleted' : ''}${hoveredCharIdx === i ? ' char-box--highlighted' : ''}`}
            style={{
              left: `${leftPct}%`,
              top: `${topPct}%`,
              width: `${widthPct}%`,
              height: `${heightPct}%`,
            }}
            onMouseEnter={() => { setLocalHoveredIdx(i); onHoverChar?.(i) }}
            onMouseLeave={() => { setLocalHoveredIdx(null); onHoverChar?.(null) }}
          >
            {localHoveredIdx === i && (
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
