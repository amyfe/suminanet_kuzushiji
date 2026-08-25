import { MdOutlineDeleteOutline, MdOutlineRestoreFromTrash } from 'react-icons/md'
import i18next from 'i18next'
import { formatScore } from '../utils/text'

export default function CharTooltipContent({ char, score, alternates, onSelect, onToggleDelete, deleted }) {
  const t = i18next.t.bind(i18next)

  // Reversible toggle, not a removal (a misclick shouldn't be destructive):
  // marked chars keep showing here so the same button can undo the mark.
  const deleteButton = onToggleDelete && (
    <button
      type="button"
      className={`tooltip-delete-btn${deleted ? ' tooltip-delete-btn--active' : ''}`}
      onClick={(e) => {
        e.stopPropagation()
        onToggleDelete()
      }}
      aria-label={deleted ? t('imageWithBoxes.restoreCharLabel') : t('imageWithBoxes.deleteCharLabel')}
    >
      {deleted ? <MdOutlineRestoreFromTrash /> : <MdOutlineDeleteOutline />}
    </button>
  )

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
        {deleteButton}
      </div>
    )
  }

  return (
    <div className="tooltip-main">
      <span className="tooltip-char">{char}</span>
      <span className="tooltip-score">{formatScore(score)}</span>
      {deleteButton}
    </div>
  )
}
