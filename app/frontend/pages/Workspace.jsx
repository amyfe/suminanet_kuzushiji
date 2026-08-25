import UploadArea from '../components/UploadArea'
import ImageWithBoxes from '../components/ImageWithBoxes'
import { Animation } from '../components/LoadingIndicator'
import TranscriptionPanel from './TranscriptionPanel'
import { useState } from 'react'
import { useLocation } from 'react-router'
import { FaEye, FaEyeSlash } from 'react-icons/fa'
import i18next from 'i18next'
import { columnsFromChars } from '../utils/text'
import { apiFetch } from '../utils/api'

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
  }

  function replaceImage() {
    setImage(null)
    setImageUrl(null)
    setFileName('')
    resetResults()
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
    if (!transcription) return
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
        body: JSON.stringify({ text: transcription, chars, lang: targetLang, include_notes: includeNotes }),
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
      transcription || t('workspace.resultNone'),
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
                <button
                  type="button"
                  className="panel-eye-toggle"
                  onClick={() => setBoxesVisible((v) => !v)}
                  aria-pressed={boxesVisible}
                  aria-label={boxesVisible ? t('workspace.hideBoxesLabel') : t('workspace.showBoxesLabel')}
                >
                  {boxesVisible ? <FaEye /> : <FaEyeSlash />}
                </button>
              )}
            </div>
            <div className="panel-body">
              {transcribing ? (
                <Animation label={t('workspace.transcribingLabel')} />
              ) : hasBoxes ? (
                <ImageWithBoxes
                  imageUrl={imageUrl}
                  chars={chars}
                  onSelectAlternate={handleAlternateSelect}
                  onReplace={replaceImage}
                  boxesVisible={boxesVisible}
                />
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
            chars={chars}
            onTranscriptionChange={handleTranscriptionChange}
            onReorderColumns={handleColumnReorder}
            onSelectAlternate={handleAlternateSelect}
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
