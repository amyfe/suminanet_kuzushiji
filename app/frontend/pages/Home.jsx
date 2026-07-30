import UploadArea from '../components/UploadArea'
import ImageWithBoxes from '../components/ImageWithBoxes'
import TranscriptionPanel from '../components/TranscriptionPanel'
import { columnsOf } from '../utils/text'
import { useState, useEffect } from 'react'

const HOW_STEPS = [
  { kanji: '一', title: 'Upload a scan', body: 'Drop in a photograph or scan of any woodblock print or handwritten manuscript — modern or Edo-era.' },
  { kanji: '二', title: 'Transcribe & correct', body: 'KuroNet detects each cursive character and proposes a reading. You review and fix anything it misjudged.' },
  { kanji: '三', title: 'Translate', body: 'Get fluent English alongside the original, drawn straight from the corrected transcription.' },
]

const EXAMPLES = [
  { id: 'furuike', title: 'The Old Pond', author: '松尾芭蕉 · Bashō', year: 'c.1686', text: '古池や蛙飛び込む水の音' },
  { id: 'natsukusa', title: 'Summer Grasses', author: '松尾芭蕉 · Bashō', year: '1689', text: '夏草や兵どもが夢の跡' },
  { id: 'shizukasa', title: 'Cicada in Stone', author: '松尾芭蕉 · Bashō', year: '1689', text: '閑さや岩にしみ入る蝉の声' },
  { id: 'asagao', title: 'Morning Glory', author: '加賀千代女 · Chiyo-jo', year: 'c.1740', text: '朝顔に釣瓶とられてもらひ水' },
  { id: 'suzume', title: 'Little Sparrow', author: '小林一茶 · Issa', year: 'c.1819', text: '雀の子そこのけそこのけ御馬が通る' },
  { id: 'nanakorobi', title: 'Fall Seven, Rise Eight', author: '江戸の諺 · Edo proverb', year: 'Edo period', text: '七転び八起き' },
]

function scrollToId(id) {
  const el = document.getElementById(id)
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

const Home = ({ intent, onIntentHandled }) => {
  const [view, setView] = useState('landing')

  const [image, setImage] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [fileName, setFileName] = useState('')
  const [chars, setChars] = useState([])
  const [transcription, setTranscription] = useState('')
  const [translation, setTranslation] = useState('')
  const [modernJapanese, setModernJapanese] = useState('')
  const [translationNotes, setTranslationNotes] = useState('')
  const [normalizationMethod, setNormalizationMethod] = useState('')
  const [transcribing, setTranscribing] = useState(false)
  const [translating, setTranslating] = useState(false)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!intent) return
    if (intent === 'try') {
      setView('workspace')
    } else {
      setView('landing')
      if (intent === 'home') {
        window.scrollTo({ top: 0 })
      } else {
        requestAnimationFrame(() => requestAnimationFrame(() => scrollToId(intent)))
      }
    }
    onIntentHandled()
  }, [intent])

  function handleTranscriptionChange(value) {
    setTranscription(value)
    setChars([])
  }

  function resetResults() {
    setChars([])
    setTranscription('')
    setTranslation('')
    setModernJapanese('')
    setTranslationNotes('')
    setNormalizationMethod('')
  }

  function handleFile(file) {
    setImage(file)
    setImageUrl(URL.createObjectURL(file))
    setFileName(file.name)
    resetResults()
  }

  function loadExample(example) {
    setImage(null)
    setImageUrl(null)
    setFileName(example.id + '.jpg')
    setChars([])
    setTranscription(example.text)
    setTranslation('')
    setModernJapanese('')
    setTranslationNotes('')
    setNormalizationMethod('')
    setView('workspace')
    window.scrollTo({ top: 0 })
  }

  async function transcribe() {
    if (!image) return
    setTranscribing(true)
    setChars([])
    setTranscription('')
    try {
      const form = new FormData()
      form.append('image', image)
      const res = await fetch('/api/transcribe', { method: 'POST', body: form })
      const data = await res.json()
      setTranscription(data.transcription)
      setChars(data.chars || [])
    } catch {
      setTranscription('Error: could not reach transcription API.')
    } finally {
      setTranscribing(false)
    }
  }

  async function translate() {
    if (!transcription) return
    setTranslating(true)
    setTranslation('')
    setModernJapanese('')
    setTranslationNotes('')
    setNormalizationMethod('')
    try {
      const res = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: transcription, chars }),
      })
      const data = await res.json()
      setTranslation(data.english_translation)
      setModernJapanese(data.modern_japanese || '')
      setTranslationNotes(data.translation_notes || '')
      setNormalizationMethod(data.normalization_method || '')
    } catch {
      setTranslation('Error: could not reach translation API.')
    } finally {
      setTranslating(false)
    }
  }

  function buildResultText() {
    const lines = [
      '墨奈 Sumina — transcription & translation',
      fileName ? `Source: ${fileName}` : null,
      '',
      'TRANSCRIPTION (翻刻):',
      transcription || '(none)',
    ]
    if (translation) {
      lines.push('', 'MODERN JAPANESE:', modernJapanese || '(none)', '', 'ENGLISH:', translation)
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

  if (view === 'landing') {
    return (
      <div className="landing">
        <section className="hero">
          <div className="hero-eyebrow">墨 SUMI · BUILT ON KURONET 黒</div>
          <h1 className="hero-title">Read the hand of Edo.</h1>
          <p className="hero-sub">
            Sumina reads the cursive <em>kuzushiji</em> in your scans of woodblock prints and
            manuscripts, lets you correct the draft, and renders centuries of Japanese into
            English you can actually read.
          </p>
          <div className="hero-cta">
            <button className="btn btn-primary" onClick={() => setView('workspace')}>
              Try it now →
            </button>
          </div>
        </section>

        <section id="how">
          <h2 className="section-title">How it works</h2>
          <div className="how-grid">
            {HOW_STEPS.map((s) => (
              <div className="how-step" key={s.kanji}>
                <div className="how-kanji">{s.kanji}</div>
                <div className="how-step-title">{s.title}</div>
                <p className="how-step-body">{s.body}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="examples">
          <div className="section-eyebrow">SELECT A TEXT · 用例</div>
          <h2 className="section-title">Examples to try</h2>
          <p className="section-sub">
            Public-domain Edo-period verse. Pick one to drop its text straight into the
            workspace, ready to translate.
          </p>
          <div className="examples-grid">
            {EXAMPLES.map((ex) => (
              <button
                type="button"
                className="example-card"
                key={ex.id}
                onClick={() => loadExample(ex)}
              >
                <div className="example-thumb">
                  <div className="example-cols">
                    {columnsOf(ex.text, 3).map((col, ci) => (
                      <div className="example-col" key={ci}>
                        {col.map((ch, ri) => (
                          <span className="example-char" key={ri}>{ch}</span>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="example-meta">
                  <div className="example-title">{ex.title}</div>
                  <div className="example-sub">{ex.author} · {ex.year}</div>
                  <div className="example-cta">Open in workspace →</div>
                </div>
              </button>
            ))}
          </div>
        </section>

        <div className="site-footer">
          <span>墨奈 Sumina — 幸 by Amina Felic</span>
          <span>Recognition powered by KuroNet (Clanuwat et al.)</span>
        </div>
      </div>
    )
  }

  return (
    <div className="container wide">
      <header className="header" style={{ textAlign: 'left', display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
        <h1 className="title">Workspace</h1>
        {fileName && <span className="result-caption" style={{ fontStyle: 'italic' }}>{fileName}</span>}
      </header>

      <div className="step-indicator">
        <span className="step active">① UPLOAD</span>
        <span className="sep">──</span>
        <span className={`step ${step >= 1 ? 'active' : ''}`}>② TRANSCRIBE &amp; CORRECT</span>
        <span className="sep">──</span>
        <span className={`step ${step >= 2 ? 'active' : ''}`}>③ TRANSLATE</span>
      </div>

      <div className="results-layout">
        <div className="image-panel">
          <div className="panel-frame">
            <div className="panel-bar">ORIGINAL · 原文</div>
            <div className="panel-body">
              {hasBoxes ? (
                <ImageWithBoxes imageUrl={imageUrl} chars={chars} />
              ) : (
                <UploadArea imageUrl={imageUrl} onFile={handleFile} />
              )}
            </div>
          </div>
          <div className="actions">
            <button
              className="btn btn-primary"
              onClick={transcribe}
              disabled={!image || transcribing}
            >
              {transcribing ? 'Transcribing...' : 'Transcribe'}
            </button>
          </div>
        </div>

        <div className="text-panel">
          <TranscriptionPanel
            transcription={transcription}
            chars={chars}
            onTranscriptionChange={handleTranscriptionChange}
            onTranslate={translate}
            translating={translating}
            translation={translation}
            modernJapanese={modernJapanese}
            translationNotes={translationNotes}
            normalizationMethod={normalizationMethod}
            onCopy={copyResult}
            onDownload={downloadTxt}
            copyLabel={copied ? 'Copied ✓' : 'Copy text'}
          />
        </div>
      </div>
    </div>
  )
}

export default Home
