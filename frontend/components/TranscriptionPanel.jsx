export default function TranscriptionPanel({
  transcription,
  onTranscriptionChange,
  onTranslate,
  translating,
  translation,
}) {
  return (
    <>
      <div className="result-box">
        <h2 className="result-label">Transcription</h2>
        <textarea
          className="transcription-textarea"
          value={transcription}
          onChange={(e) => onTranscriptionChange(e.target.value)}
        />
      </div>

      <button
        className="btn btn-secondary"
        onClick={onTranslate}
        disabled={!transcription || translating}
      >
        {translating ? 'Translating...' : 'Translate'}
      </button>

      {translation && (
        <div className="result-box">
          <h2 className="result-label">Translation</h2>
          <p className="result-text">{translation}</p>
        </div>
      )}
    </>
  )
}
