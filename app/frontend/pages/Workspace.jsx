import UploadArea from '../components/UploadArea'
import ImageWithBoxes from '../components/ImageWithBoxes'
import { Animation } from '../components/LoadingIndicator'
import TranscriptionPanel from './TranscriptionPanel'
import { useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { useLocation } from 'react-router'
import { FaEye, FaEyeSlash } from 'react-icons/fa'
import { MdOutlineZoomIn, MdOutlineZoomOut } from 'react-icons/md'
import i18next from 'i18next'
import { columnsFromChars } from '../utils/text'
import { apiFetch } from '../utils/api'
import useSimulatedProgress from '../utils/useSimulatedProgress'

const Workspace = () => {
  const location = useLocation()
  const example = location.state?.example
  const t = i18next.t.bind(i18next)

  const [image, setImage] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [fileName, setFileName] = useState(example ? example.id + '.jpg' : '')
  const [chars, setChars] = useState([])
  const [transcription, setTranscription] = useState(example ? example.text : '')
  const [translation, setTranslation] = useState('')
  const [normalizedJapanese, setNormalizedJapanese] = useState('')
  const [modernJapanese, setModernJapanese] = useState('')
  const [conversionNotes, setConversionNotes] = useState('')
  const [translationNotes, setTranslationNotes] = useState('')
  const [normalizationMethod, setNormalizationMethod] = useState('')
  const [targetLang, setTargetLang] = useState('en')
  const [includeNotes, setIncludeNotes] = useState(true)
  const [transcribing, setTranscribing] = useState(false)
  const [translating, setTranslating] = useState(false)
  const [copied, setCopied] = useState(false)
  const [boxesVisible, setBoxesVisible] = useState(true)
  const [zoomed, setZoomed] = useState(false)
  const transcribeProgress = useSimulatedProgress(transcribing, 10000)

  useEffect(() => {
    if (!zoomed) return
    function onKeyDown(e) {
      if (e.key === 'Escape') setZoomed(false)
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [zoomed])

  function handleTranscriptionChange(value) {
    setTranscription(value)
    setChars([])
  }

  function resetResults() {
    setChars([])
    setTranscription('')
    setTranslation('')
    setNormalizedJapanese('')
    setModernJapanese('')
    setConversionNotes('')
    setTranslationNotes('')
    setNormalizationMethod('')
  }

  function handleFile(file) {
    setImage(file)
    setImageUrl(URL.createObjectURL(file))
    setFileName(file.name)
    resetResults()
    setZoomed(false)
  }

  function replaceImage() {
    setImage(null)
    setImageUrl(null)
    setFileName('')
    resetResults()
    setZoomed(false)
  }

  // chars[i].char is positionally 1:1 with transcription[i] (both built
  // together by run_inference()); swap both so the box overlay and the
  // text stay in sync.
  function handleAlternateSelect(index, altChar) {
    setChars((prev) => {
      const next = [...prev]
      next[index] = { ...next[index], char: altChar }
      return next
    })
    setTranscription((prev) => {
      const codepoints = Array.from(prev)
      codepoints[index] = altChar
      return codepoints.join('')
    })
  }

  // Marks a detected box as a background element rather than real text --
  // a reversible flag, not a removal, so a misclick is never destructive.
  // chars/transcription stay 1:1 in length (nothing spliced out), which
  // is what every other index-based feature here (alternates, reorder,
  // hover) relies on; the character is only excluded from what actually
  // reaches translation/export via `visibleTranscription` below.
  function handleToggleDeleteChar(index) {
    setChars((prev) => {
      const next = [...prev]
      next[index] = { ...next[index], deleted: !next[index].deleted }
      return next
    })
  }

  // The text actually sent for translation/export excludes chars marked as
  // background -- this is where that exclusion is applied, on top of the
  // full chars/transcription that every other index-based feature relies
  // on. Falls back to the raw transcription once a manual edit has broken
  // the chars/transcription correspondence (mirrors TranscriptionPanel's
  // own charsInSync gate).
  const charsInSync = chars.length > 0 && chars.map((c) => c.char).join('') === transcription
  const visibleTranscription = charsInSync
    ? chars.filter((c) => !c.deleted).map((c) => c.char).join('')
    : transcription

  // Manual fix for reading-order mistakes (ticket 6.19): moves a whole
  // column from fromIdx to toIdx. Indices are canonical reading-order
  // indices (0 = first-read/rightmost column), matching columnsFromChars'
  // own ordering — TranscriptionPanel converts its reversed DOM order back
  // to this canonical form before calling in. Functional updates (not
  // closing over `chars`/`transcription`) match handleAlternateSelect's
  // discipline of keeping the two arrays 1:1 positionally synced.
  function handleColumnReorder(fromIdx, toIdx) {
    if (fromIdx === toIdx) return
    setChars((prevChars) => {
      const cols = columnsFromChars(prevChars)
      if (fromIdx < 0 || toIdx < 0 || fromIdx >= cols.length || toIdx >= cols.length) {
        return prevChars
      }
      const reordered = [...cols]
      const [moved] = reordered.splice(fromIdx, 1)
      reordered.splice(toIdx, 0, moved)
      const flat = reordered.flat()
      setTranscription(flat.map((c) => c.char).join(''))
      return flat
    })
  }

  async function transcribe() {
    if (!image) return
    setTranscribing(true)
    setZoomed(false)
    setChars([])
    setTranscription('')
    try {
      const form = new FormData()
      form.append('image', image)
      const res = await apiFetch('/api/transcribe', { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || 'transcribe failed')
      }
      setTranscription(data.transcription)
      setChars(data.chars || [])
    } catch {
      setTranscription(t('workspace.transcribeError'))
    } finally {
      setTranscribing(false)
    }
  }

  async function translate() {
    if (!visibleTranscription) return
    setTranslating(true)
    setTranslation('')
    setNormalizedJapanese('')
    setModernJapanese('')
    setConversionNotes('')
    setTranslationNotes('')
    setNormalizationMethod('')
    try {
      const res = await apiFetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: visibleTranscription,
          chars: charsInSync ? chars.filter((c) => !c.deleted) : chars,
          lang: targetLang,
          include_notes: includeNotes,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || 'translate failed')
      }
      setTranslation(data.english_translation)
      setNormalizedJapanese(data.normalized_japanese || '')
      setModernJapanese(data.modern_japanese || '')
      setConversionNotes(data.conversion_notes || '')
      setTranslationNotes(data.translation_notes || '')
      setNormalizationMethod(data.normalization_method || '')
    } catch {
      setTranslation(t('workspace.translateError'))
    } finally {
      setTranslating(false)
    }
  }

  function buildResultText() {
    const lines = [
      t('workspace.resultHeaderPrefix'),
      fileName ? `${t('workspace.resultSourceLabel')}: ${fileName}` : null,
      '',
      t('workspace.resultTranscriptionLabel'),
      visibleTranscription || t('workspace.resultNone'),
    ]
    if (translation) {
      const translationLabel = targetLang === 'de' ? t('workspace.resultGermanLabel') : t('workspace.resultEnglishLabel')
      lines.push('', t('workspace.resultModernLabel'), modernJapanese || t('workspace.resultNone'), '', translationLabel, translation)
    }
    return lines.filter((l) => l !== null).join('\n')
  }

  function copyResult() {
    const done = () => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    }
    const text = buildResultText()
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, done)
    } else {
      done()
    }
  }

  function downloadTxt() {
    const blob = new Blob([buildResultText()], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = (fileName.replace(/\.[^.]+$/, '') || 'sumina') + '-sumina.txt'
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const step = translation ? 2 : transcription ? 1 : 0
  const hasBoxes = chars.length > 0

  return (
    <div className="container wide">
      <header className="header" style={{ textAlign: 'left', display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
        <h1 className="title">{t('workspace.title')}</h1>
        {fileName && <span className="result-caption" style={{ fontStyle: 'italic' }}>{fileName}</span>}
      </header>

      <div className="step-indicator">
        <span className="step active">{t('workspace.stepUpload')}</span>
        <span className="sep">──</span>
        <span className={`step ${step >= 1 ? 'active' : ''}`}>{t('workspace.stepTranscribe')}</span>
        <span className="sep">──</span>
        <span className={`step ${step >= 2 ? 'active' : ''}`}>{t('workspace.stepTranslate')}</span>
      </div>

      <div className="results-layout">
        <div className="image-panel">
          <div className="panel-frame">
            <div className="panel-bar">
              {t('workspace.panelOriginal')}
              {hasBoxes && (
                <div className="panel-bar-actions">
                  <button
                    type="button"
                    className="panel-zoom-toggle"
                    onClick={() => setZoomed(true)}
                    aria-label={t('workspace.zoomInLabel')}
                  >
                    <MdOutlineZoomIn />
                  </button>
                  <button
                    type="button"
                    className="panel-eye-toggle"
                    onClick={() => setBoxesVisible((v) => !v)}
                    aria-pressed={boxesVisible}
                    aria-label={boxesVisible ? t('workspace.hideBoxesLabel') : t('workspace.showBoxesLabel')}
                  >
                    {boxesVisible ? <FaEye /> : <FaEyeSlash />}
                  </button>
                </div>
              )}
            </div>
            <div className="panel-body">
              {transcribing ? (
                <Animation label={t('workspace.transcribingLabel')} progress={transcribeProgress} />
              ) : hasBoxes ? (
                zoomed ? (
                  // Portaled straight to document.body: .panel-frame/.panel-body
                  // (here and in the sibling text-panel) each set an explicit
                  // z-index, which creates a stacking context even at z-index:0.
                  // A fixed-position overlay nested inside one is trapped
                  // there for stacking purposes -- it would lose a z-index tie
                  // against the text-panel's own equally-trapped z-index:0
                  // context (whichever comes later in the DOM wins ties), so
                  // rendering it in-place could put the transcription panel on
                  // top of the "zoomed" image instead of under it. Portaling
                  // escapes every ancestor's stacking context entirely.
                  createPortal(
                    <>
                      <div className="zoom-backdrop" onClick={() => setZoomed(false)} />
                      <ImageWithBoxes
                        imageUrl={imageUrl}
                        chars={chars}
                        onSelectAlternate={handleAlternateSelect}
                        onToggleDeleteChar={handleToggleDeleteChar}
                        boxesVisible={boxesVisible}
                        zoomed
                      />
                      <button
                        type="button"
                        className="zoom-out-btn"
                        onClick={() => setZoomed(false)}
                        aria-label={t('workspace.zoomOutLabel')}
                      >
                        <MdOutlineZoomOut />
                      </button>
                    </>,
                    document.body
                  )
                ) : (
                  <ImageWithBoxes
                    imageUrl={imageUrl}
                    chars={chars}
                    onSelectAlternate={handleAlternateSelect}
                    onToggleDeleteChar={handleToggleDeleteChar}
                    onReplace={replaceImage}
                    boxesVisible={boxesVisible}
                    onFile={handleFile}
                  />
                )
              ) : (
                <UploadArea imageUrl={imageUrl} onFile={handleFile} />
              )}
            </div>
          </div>
          <div className="actions">
            {!transcribing && (
              <button
                className="btn btn-primary"
                onClick={transcribe}
                disabled={!image}
              >
                {t('workspace.transcribeBtn')}
              </button>
            )}
            {imageUrl && (
              <button className="btn btn-secondary" onClick={replaceImage} disabled={transcribing}>
                {t('workspace.replaceImageBtn')}
              </button>
            )}
          </div>
        </div>

        <div className="text-panel">
          <TranscriptionPanel
            transcription={transcription}
            visibleTranscription={visibleTranscription}
            chars={chars}
            onTranscriptionChange={handleTranscriptionChange}
            onReorderColumns={handleColumnReorder}
            onSelectAlternate={handleAlternateSelect}
            onToggleDeleteChar={handleToggleDeleteChar}
            onTranslate={translate}
            translating={translating}
            translation={translation}
            normalizedJapanese={normalizedJapanese}
            modernJapanese={modernJapanese}
            conversionNotes={conversionNotes}
            translationNotes={translationNotes}
            normalizationMethod={normalizationMethod}
            targetLang={targetLang}
            onTargetLangChange={setTargetLang}
            includeNotes={includeNotes}
            onIncludeNotesChange={setIncludeNotes}
            onCopy={copyResult}
            onDownload={downloadTxt}
            copyLabel={copied ? t('workspace.copiedText') : t('workspace.copyText')}
          />
        </div>
      </div>
    </div>
  )
}

export default Workspace
