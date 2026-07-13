import { useState, useRef } from 'react'

export default function ImageWithBoxes({ imageUrl, chars }) {
  const imgRef = useRef(null)
  const [imgDisplaySize, setImgDisplaySize] = useState(null)
  const [hoveredCharIdx, setHoveredCharIdx] = useState(null)

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
        className="overlay-image"
        alt="Manuscript with annotations"
        onLoad={onImageLoad}
      />
      {imgDisplaySize && chars.map((c, i) => {
        const scaleX = imgDisplaySize.w / imgDisplaySize.natW
        const scaleY = imgDisplaySize.h / imgDisplaySize.natH
        const [x1, y1, x2, y2] = c.box
        return (
          <div
            key={i}
            className="char-box"
            style={{
              left: x1 * scaleX,
              top: y1 * scaleY,
              width: (x2 - x1) * scaleX,
              height: (y2 - y1) * scaleY,
            }}
            onMouseEnter={() => setHoveredCharIdx(i)}
            onMouseLeave={() => setHoveredCharIdx(null)}
          >
            {hoveredCharIdx === i && (
              <div className="char-tooltip">
                <span className="tooltip-char">{c.char}</span>
                <span className="tooltip-score">{(c.score * 100).toFixed(0)}%</span>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
