import { useState } from 'react'
import { columnsOf, columnsFromChars } from '../utils/text'
import { Animation } from '../components/LoadingIndicator'
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
  chars,
  onTranscriptionChange,
  onReorderColumns,
  onTranslate,
  translating,
  translation,
  modernJapanese,
  translationNotes,
  normalizationMethod,
  onCopy,
  onDownload,
  copyLabel,
  targetLang,
  onTargetLangChange,
}) {
  const t = i18next.t.bind(i18next)
  const normalization = t('transcriptionPanel.normalization', { returnObjects: true })
  const resultLabels = {
    en: t('transcriptionPanel.resultLabel'),
    de: t('transcriptionPanel.resultLabelDe'),
  }
  const [dragOverIdx, setDragOverIdx] = useState(null)

  // Prefer true column boundaries from the detected char boxes (matches the
  // manuscript's actual layout). Only valid while chars still matches the
  // displayed transcription exactly — once the user edits the text, boxes
  // and characters are no longer in correspondence, so fall back to a naive
  // fixed-length chunking of the string instead.
  const charsInSync = chars && chars.length > 0 && chars.map((c) => c.char).join('') === transcription
  let cols = []
  if (charsInSync) {
    cols = columnsFromChars(chars).reverse().map((col) => col.map((c) => c.char))
  } else if (transcription) {
    cols = columnsOf(transcription, Math.max(1, Math.ceil(Array.from(transcription).length / VERTICAL_TARGET_CHARS_PER_COL)))
  }
  // Reordering only makes sense against real box-derived columns (not the
  // naive fallback chunking), and only when there's more than one column.
  const reorderable = charsInSync && cols.length > 1

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
  }

  function handleColDragOver(e, domIdx) {
    e.preventDefault()
    if (dragOverIdx !== domIdx) setDragOverIdx(domIdx)
  }

  function handleColDrop(e, domIdx) {
    e.preventDefault()
    setDragOverIdx(null)
    const fromDomIdx = Number(e.dataTransfer.getData('text/plain'))
    if (Number.isNaN(fromDomIdx) || fromDomIdx === domIdx) return
    onReorderColumns?.(domToCanonical(fromDomIdx), domToCanonical(domIdx))
  }

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
                    className={`transcription-vertical-col${reorderable ? ' reorderable' : ''}${dragOverIdx === ci ? ' drag-over' : ''}`}
                    key={ci}
                    draggable={reorderable}
                    onDragStart={reorderable ? (e) => handleColDragStart(e, ci) : undefined}
                    onDragOver={reorderable ? (e) => handleColDragOver(e, ci) : undefined}
                    onDrop={reorderable ? (e) => handleColDrop(e, ci) : undefined}
                    onDragEnd={reorderable ? () => setDragOverIdx(null) : undefined}
                  >
                    {col.join('')}
                  </div>
                ))}
              </div>
              {reorderable && (
                <p className="transcription-hint">{t('transcriptionPanel.reorderHint')}</p>
              )}
            </>
          )}
          <textarea
            className="transcription-textarea"
            value={transcription}
            onChange={(e) => onTranscriptionChange(e.target.value)}
            placeholder={t('transcriptionPanel.placeholder')}
            rows={4}
          />
          <p className="transcription-hint">
            {t('transcriptionPanel.hint')}
          </p>
        </div>
      </div>

      <div className="translate-row">
        <button
          className="btn btn-primary"
          onClick={onTranslate}
          disabled={!transcription || translating}
        >
          {translating ? t('transcriptionPanel.translatingBtn') : t('transcriptionPanel.translateBtn')}
        </button>
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

      {translating && <Animation label={t('transcriptionPanel.translatingLabel')} />}

      {translation && (
        <div className="result-box translation">
          <h2 className="result-label">{resultLabels[targetLang] || resultLabels.en}</h2>
          <p className="result-text">{translation}</p>
          {modernJapanese && (
            <p className="result-modern">{modernJapanese}</p>
          )}
          {translationNotes && (
            <p className="result-caption">{translationNotes}</p>
          )}
          {normText && (
            <p className={`result-caption${normDegraded ? ' result-caption--warn' : ''}`}>
              {normDegraded ? '⚠ ' : ''}
              {normText}
            </p>
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
