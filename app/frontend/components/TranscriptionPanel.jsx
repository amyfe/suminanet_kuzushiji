import { columnsOf, columnsFromChars } from '../utils/text'

const VERTICAL_TARGET_CHARS_PER_COL = 14

export default function TranscriptionPanel({
  transcription,
  chars,
  onTranscriptionChange,
  onTranslate,
  translating,
  translation,
  modernJapanese,
  translationNotes,
  onCopy,
  onDownload,
  copyLabel,
}) {
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

  return (
    <>
      <div className="panel-frame">
        <div className="panel-bar">TRANSCRIPTION · 翻刻</div>
        <div className="panel-body">
          {cols.length > 0 && (
            <div className="transcription-vertical" aria-hidden="true">
              {cols.map((col, ci) => (
                <div className="transcription-vertical-col" key={ci}>
                  {col.join('')}
                </div>
              ))}
            </div>
          )}
          <textarea
            className="transcription-textarea"
            value={transcription}
            onChange={(e) => onTranscriptionChange(e.target.value)}
            placeholder="Type or correct the reading…"
            rows={4}
          />
          <p className="transcription-hint">
            Run Transcribe on a scan, or type/paste classical Japanese directly, then Translate.
          </p>
        </div>
      </div>

      <button
        className="btn btn-primary"
        onClick={onTranslate}
        disabled={!transcription || translating}
      >
        {translating ? 'Translating...' : 'Translate'}
      </button>

      {translation && (
        <div className="result-box translation">
          <h2 className="result-label">English · 英訳</h2>
          <p className="result-text">{translation}</p>
          {modernJapanese && (
            <p className="result-modern">{modernJapanese}</p>
          )}
          {translationNotes && (
            <p className="result-caption">{translationNotes}</p>
          )}

          <div className="export-bar" style={{ marginTop: '1.1rem' }}>
            <span className="export-label">EXPORT</span>
            <button className="btn-export" onClick={onCopy}>{copyLabel}</button>
            <button className="btn-export" onClick={onDownload}>Download .txt</button>
          </div>
        </div>
      )}
    </>
  )
}
