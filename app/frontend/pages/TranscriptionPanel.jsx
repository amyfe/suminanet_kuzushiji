import { useEffect, useRef, useState } from 'react'
import { columnsOf, columnsFromChars } from '../utils/text'
import { Animation } from '../components/LoadingIndicator'
import CharTooltipContent from '../components/CharTooltipContent'
import useSimulatedProgress from '../utils/useSimulatedProgress'
import i18next from 'i18next'

const VERTICAL_TARGET_CHARS_PER_COL = 14

// flag-icons uses ISO country codes, not language codes (English -> GB).
const TARGET_LANGUAGES = [
  { code: 'en', country: 'gb', name: 'English' },
  { code: 'de', country: 'de', name: 'Deutsch' },
]

// Which normalization methods are degraded fallbacks (warn-styled) — not
// translatable content, so kept out of the JSON, keyed the same way as
// transcriptionPanel.normalization.
const DEGRADED_METHODS = new Set(['mecab+unidic', 'mecab+unidic-lite', 'heuristic'])

export default function TranscriptionPanel({
  transcription,
  visibleTranscription,
  chars,
  onTranscriptionChange,
  onReorderColumns,
  onSelectAlternate,
  onToggleDeleteChar,
  hoveredCharIdx,
  onHoverChar,
  onTranslate,
  translating,
  translation,
  normalizedJapanese,
  modernJapanese,
  conversionNotes,
  translationNotes,
  normalizationMethod,
  onCopy,
  onDownload,
  copyLabel,
  targetLang,
  onTargetLangChange,
  includeNotes,
  onIncludeNotesChange,
}) {
  const t = i18next.t.bind(i18next)
  const normalization = t('transcriptionPanel.normalization', { returnObjects: true })
  const resultLabels = {
    en: t('transcriptionPanel.resultLabel'),
    de: t('transcriptionPanel.resultLabelDe'),
  }
  const [dragOverIdx, setDragOverIdx] = useState(null)
  const [draggedDomIdx, setDraggedDomIdx] = useState(null)
  const [showIntermediate, setShowIntermediate] = useState(false)
  // Local, not the shared hoveredCharIdx prop: this panel's own tooltip
  // should only appear when the mouse is actually over one of its own
  // characters, not just because the OTHER panel reported a hover. The
  // shared prop still drives this panel's highlight styling below, so
  // hovering either panel lights up both -- only the tooltip stays local.
  const [localHoveredIdx, setLocalHoveredIdx] = useState(null)
  const translatingRef = useRef(null)
  const translateProgress = useSimulatedProgress(translating, 12000)

  useEffect(() => {
    if (!includeNotes) setShowIntermediate(false)
  }, [includeNotes])

  useEffect(() => {
    if (translating) {
      translatingRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [translating])

  // Prefer true column boundaries from the detected char boxes (matches the
  // manuscript's actual layout). Only valid while chars still matches the
  // displayed transcription exactly — once the user edits the text, boxes
  // and characters are no longer in correspondence, so fall back to a naive
  // fixed-length chunking of the string instead.
  const charsInSync = chars && chars.length > 0 && chars.map((c) => c.char).join('') === transcription
  // Object-reference map from each char back to its position in the flat
  // chars/transcription arrays (columnsFromChars groups the same object
  // references, never clones), so a click deep inside a column can still
  // call onSelectAlternate with the right index.
  const charIndex = charsInSync ? new Map(chars.map((c, i) => [c, i])) : null
  let cols = []
  if (charsInSync) {
    cols = columnsFromChars(chars).reverse()
  } else if (transcription) {
    cols = columnsOf(transcription, Math.max(1, Math.ceil(Array.from(transcription).length / VERTICAL_TARGET_CHARS_PER_COL)))
  }
  // Reordering only makes sense against real box-derived columns (not the
  // naive fallback chunking), and only when there's more than one column.
  const reorderable = charsInSync && cols.length > 1
  // Same real-columns requirement as reorderable, but no column-count floor
  // -- a single-column page still benefits from per-character correction.
  // Only worth advertising when some char actually has a runner-up beyond
  // its own chosen pick.
  const hasAlternates = charsInSync && chars.some((c) => c.alternates?.length > 1)

  // Ticket 6.8: mirror the vertical-column view's structure in the flat
  // horizontal textarea below it by breaking at the same column boundaries
  // (in canonical reading order, matching how `transcription` itself was
  // assembled -- unlike `cols` above, this is NOT DOM-reversed). Gated on
  // charsInSync like every other column-aware feature here: the moment the
  // user edits, chars/transcription fall out of sync and this reverts to a
  // single unbroken line, same as `reorderable` already does.
  const readingOrderColStrings = charsInSync
    ? columnsFromChars(chars).map((col) => col.filter((c) => !c.deleted).map((c) => c.char).join(''))
    : null
  const textareaValue = readingOrderColStrings && readingOrderColStrings.length > 1
    ? readingOrderColStrings.join('\n')
    : visibleTranscription

  // cols is already reversed relative to columnsFromChars' canonical reading
  // order (rightmost/first-read column rendered last in DOM, since a plain
  // LTR flex-row lays out left-to-right) — so a DOM index must be flipped
  // back to a canonical index before reaching onReorderColumns.
  function domToCanonical(domIdx) {
    return cols.length - 1 - domIdx
  }

  function handleColDragStart(e, domIdx) {
    e.dataTransfer.setData('text/plain', String(domIdx))
    e.dataTransfer.effectAllowed = 'move'
    setDraggedDomIdx(domIdx)
  }

  function handleColDragOver(e, domIdx) {
    e.preventDefault()
    if (dragOverIdx !== domIdx) setDragOverIdx(domIdx)
  }

  function handleColDrop(e, domIdx) {
    e.preventDefault()
    setDragOverIdx(null)
    setDraggedDomIdx(null)
    const fromDomIdx = Number(e.dataTransfer.getData('text/plain'))
    if (Number.isNaN(fromDomIdx) || fromDomIdx === domIdx) return
    onReorderColumns?.(domToCanonical(fromDomIdx), domToCanonical(domIdx))
  }

  const dragOverSide =
    dragOverIdx !== null && draggedDomIdx !== null && dragOverIdx !== draggedDomIdx
      ? (dragOverIdx > draggedDomIdx ? 'right' : 'left')
      : null

  const normText = normalizationMethod ? normalization[normalizationMethod] : null
  const normDegraded = DEGRADED_METHODS.has(normalizationMethod)

  return (
    <>
      <div className="panel-frame">
        <div className="panel-bar">{t('transcriptionPanel.panelHeader')}</div>
        <div className="panel-body">
          {cols.length > 0 && (
            <>
              <div className="transcription-vertical" aria-hidden="true">
                {cols.map((col, ci) => (
                  <div
                    className={`transcription-vertical-col${reorderable ? ' reorderable' : ''}${dragOverIdx === ci ? ' drag-over' : ''}${dragOverIdx === ci && dragOverSide ? ` drag-over--${dragOverSide}` : ''}`}
                    key={ci}
                    draggable={reorderable}
                    onDragStart={reorderable ? (e) => handleColDragStart(e, ci) : undefined}
                    onDragOver={reorderable ? (e) => handleColDragOver(e, ci) : undefined}
                    onDrop={reorderable ? (e) => handleColDrop(e, ci) : undefined}
                    onDragEnd={reorderable ? () => { setDragOverIdx(null); setDraggedDomIdx(null) } : undefined}
                  >
                    {charsInSync
                      ? col.map((c) => {
                          const idx = charIndex.get(c)
                          return (
                            <span
                              key={idx}
                              className={`transcription-char${c.deleted ? ' transcription-char--deleted' : ''}${hoveredCharIdx === idx ? ' transcription-char--highlighted' : ''}`}
                              onMouseEnter={() => { setLocalHoveredIdx(idx); onHoverChar?.(idx) }}
                              onMouseLeave={() => { setLocalHoveredIdx(null); onHoverChar?.(null) }}
                            >
                              {c.char}
                              {localHoveredIdx === idx && (
                                <div className="char-tooltip char-tooltip--vertical" draggable={false}>
                                  <CharTooltipContent
                                    char={c.char}
                                    score={c.score}
                                    alternates={c.alternates}
                                    deleted={c.deleted}
                                    onSelect={(altChar) => onSelectAlternate?.(idx, altChar)}
                                    onToggleDelete={onToggleDeleteChar ? () => onToggleDeleteChar(idx) : undefined}
                                  />
                                </div>
                              )}
                            </span>
                          )
                        })
                      : col.join('')}
                  </div>
                ))}
              </div>
              {reorderable && (
                <p className="transcription-hint">{t('transcriptionPanel.reorderHint')}</p>
              )}
              {hasAlternates && (
                <p className="transcription-hint">{t('transcriptionPanel.alternatesHint')}</p>
              )}
            </>
          )}
          <textarea
            className="transcription-textarea"
            value={textareaValue}
            onChange={(e) => onTranscriptionChange(e.target.value.replace(/\n/g, ''))}
            placeholder={t('transcriptionPanel.placeholder')}
            rows={4}
          />
          <p className="transcription-hint">
            {t('transcriptionPanel.hint')}
          </p>
        </div>
      </div>

      {visibleTranscription && !translation && (
        <div className="export-bar">
          <span className="export-label">{t('transcriptionPanel.exportLabel')}</span>
          <button className="btn-export" onClick={onCopy}>{copyLabel}</button>
          <button className="btn-export" onClick={onDownload}>{t('transcriptionPanel.downloadBtn')}</button>
        </div>
      )}

      <div className="translate-row">
        {!translating && (
          <button
            className="btn btn-primary"
            onClick={onTranslate}
            disabled={!visibleTranscription}
          >
            {t('transcriptionPanel.translateBtn')}
          </button>
        )}
        <div className="lang-select-wrap">
          <span className={`fi fi-${TARGET_LANGUAGES.find((l) => l.code === targetLang)?.country} lang-select-flag`} aria-hidden="true" />
          <select
            className="lang-select"
            value={targetLang}
            onChange={(e) => onTargetLangChange(e.target.value)}
            disabled={translating}
            aria-label={t('transcriptionPanel.targetLangAriaLabel')}
          >
            {TARGET_LANGUAGES.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <label className="notes-toggle">
        <input
          type="checkbox"
          checked={includeNotes}
          onChange={(e) => onIncludeNotesChange(e.target.checked)}
          disabled={translating}
        />
        {t('transcriptionPanel.includeNotesLabel')}
      </label>

      {translating && (
        <div ref={translatingRef}>
          <Animation label={t('transcriptionPanel.translatingLabel')} progress={translateProgress} />
        </div>
      )}

      {translation && (
        <div className="result-box translation">
          <h2 className="result-label">{resultLabels[targetLang] || resultLabels.en}</h2>
          <p className="result-text">{translation}</p>

          {normDegraded && (
            <p className="result-caption result-caption--warn">⚠ {normText}</p>
          )}

          {includeNotes && (normalizedJapanese || modernJapanese || conversionNotes || translationNotes || (normText && !normDegraded)) && (
            <>
              <label className="intermediate-toggle">
                <input
                  type="checkbox"
                  checked={showIntermediate}
                  onChange={(e) => setShowIntermediate(e.target.checked)}
                />
                {t('transcriptionPanel.showIntermediateLabel')}
              </label>

              {showIntermediate && (
                <div className="intermediate-steps">
                  {normalizedJapanese && (
                    <div className="intermediate-step">
                      <span className="intermediate-step-label">{t('transcriptionPanel.normalizedLabel')}</span>
                      <p className="result-modern">{normalizedJapanese}</p>
                    </div>
                  )}
                  {modernJapanese && (
                    <div className="intermediate-step">
                      <span className="intermediate-step-label">{t('transcriptionPanel.modernLabel')}</span>
                      <p className="result-modern">{modernJapanese}</p>
                    </div>
                  )}
                  {conversionNotes && (
                    <div className="intermediate-step">
                      <span className="intermediate-step-label">{t('transcriptionPanel.conversionNotesLabel')}</span>
                      <p className="result-caption">{conversionNotes}</p>
                    </div>
                  )}
                  {translationNotes && (
                    <div className="intermediate-step">
                      <span className="intermediate-step-label">{t('transcriptionPanel.translationNotesLabel')}</span>
                      <p className="result-caption">{translationNotes}</p>
                    </div>
                  )}
                  {normText && !normDegraded && (
                    <p className="result-caption">{normText}</p>
                  )}
                </div>
              )}
            </>
          )}

          <div className="export-bar" style={{ marginTop: '1.1rem' }}>
            <span className="export-label">{t('transcriptionPanel.exportLabel')}</span>
            <button className="btn-export" onClick={onCopy}>{copyLabel}</button>
            <button className="btn-export" onClick={onDownload}>{t('transcriptionPanel.downloadBtn')}</button>
          </div>
        </div>
      )}
    </>
  )
}
