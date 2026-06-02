import { useState, useRef } from 'react'
import './style.css'

export default function App() {
  const [image, setImage] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [transcription, setTranscription] = useState('')
  const [translation, setTranslation] = useState('')
  const [transcribing, setTranscribing] = useState(false)
  const [translating, setTranslating] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return
    setImage(file)
    setImageUrl(URL.createObjectURL(file))
    setTranscription('')
    setTranslation('')
  }

  function onFileChange(e) {
    handleFile(e.target.files[0])
  }

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  async function transcribe() {
    if (!image) return
    setTranscribing(true)
    setTranscription('')
    try {
      const form = new FormData()
      form.append('image', image)
      const res = await fetch('/api/transcribe', { method: 'POST', body: form })
      const data = await res.json()
      setTranscription(data.transcription)
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
    try {
      const res = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: transcription }),
      })
      const data = await res.json()
      setTranslation(data.translation)
    } catch {
      setTranslation('Error: could not reach translation API.')
    } finally {
      setTranslating(false)
    }
  }

  return (
    <div className="container">
      <header className="header">
        <h1 className="title">Kuzushiji Reader</h1>
        <p className="subtitle">Upload a kuzushiji manuscript image to transcribe and translate</p>
      </header>

      <div
        className={`upload-area ${dragOver ? 'drag-over' : ''} ${imageUrl ? 'has-image' : ''}`}
        onClick={() => fileInputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={onFileChange}
          style={{ display: 'none' }}
        />
        {imageUrl ? (
          <img src={imageUrl} className="preview-image" alt="Uploaded manuscript" />
        ) : (
          <div className="upload-placeholder">
            <div className="upload-icon">+</div>
            <p>Click or drag an image here</p>
          </div>
        )}
      </div>

      <div className="actions">
        <button
          className="btn btn-primary"
          onClick={transcribe}
          disabled={!image || transcribing}
        >
          {transcribing ? 'Transcribing...' : 'Transcribe'}
        </button>
        <button
          className="btn btn-secondary"
          onClick={translate}
          disabled={!transcription || translating}
        >
          {translating ? 'Translating...' : 'Translate'}
        </button>
      </div>

      {transcription && (
        <div className="result-box">
          <h2 className="result-label">Transcription</h2>
          <p className="result-text">{transcription}</p>
        </div>
      )}

      {translation && (
        <div className="result-box">
          <h2 className="result-label">Translation</h2>
          <p className="result-text">{translation}</p>
        </div>
      )}
    </div>
  )
}
