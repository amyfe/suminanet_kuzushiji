import i18next from 'i18next'
import { formatScore } from '../utils/text'

export default function CharTooltipContent({ char, score, alternates, onSelect }) {
  const t = i18next.t.bind(i18next)

  if (alternates?.length > 0) {
    return (
      <div className="tooltip-alternates">
        {alternates.slice(0, 3).map((alt, ai) => (
          <button
            type="button"
            // Backend guarantees all candidates in alternates are distinct
            // chars (vocab is a char<->id bijection, and the chosen id is
            // excluded when collecting runner-ups), so char equality is a
            // safe way to mark the active pick.
            className={`tooltip-alt-char${alt.char === char ? ' tooltip-alt-char--active' : ''}`}
            key={ai}
            onClick={(e) => {
              e.stopPropagation()
              onSelect?.(alt.char)
            }}
            aria-label={t('imageWithBoxes.useAlternate', { char: alt.char })}
          >
            {alt.char} <span className="tooltip-alt-score">{formatScore(alt.prob)}</span>
          </button>
        ))}
      </div>
    )
  }

  return (
    <div className="tooltip-main">
      <span className="tooltip-char">{char}</span>
      <span className="tooltip-score">{formatScore(score)}</span>
    </div>
  )
}
