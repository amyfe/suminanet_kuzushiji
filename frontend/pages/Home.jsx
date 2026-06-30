import UploadArea from '../components/UploadArea'
import ImageWithBoxes from '../components/ImageWithBoxes'
import TranscriptionPanel from '../components/TranscriptionPanel'
import { useState } from 'react'

const Home = () => {
      const [image, setImage] = useState(null)
      const [imageUrl, setImageUrl] = useState(null)
      const [chars, setChars] = useState([])
      const [transcription, setTranscription] = useState('')
      const [translation, setTranslation] = useState('')
      const [transcribing, setTranscribing] = useState(false)
      const [translating, setTranslating] = useState(false)
    
      function handleFile(file) {
        setImage(file)
        setImageUrl(URL.createObjectURL(file))
        setChars([])
        setTranscription('')
        setTranslation('')
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
        try {
          const res = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: transcription }),
          })
          const data = await res.json()
          setTranslation(data.english_translation)
        } catch {
          setTranslation('Error: could not reach translation API.')
        } finally {
          setTranslating(false)
        }
      }
    
      const hasResults = chars.length > 0
    
  return (
    <div className={`container ${hasResults ? 'wide' : ''}`}>
        <header className="header">
        <h1 className="title">Kuzushiji Reader</h1>
        <p className="subtitle">Upload a kuzushiji manuscript image to transcribe and translate</p>
      </header>

      <UploadArea imageUrl={imageUrl} onFile={handleFile} />

      <div className="actions">
        <button
          className="btn btn-primary"
          onClick={transcribe}
          disabled={!image || transcribing}
        >
          {transcribing ? 'Transcribing...' : 'Transcribe'}
        </button>
      </div>

      {hasResults && (
        <div className="results-layout">
          <div className="image-panel">
            <ImageWithBoxes imageUrl={imageUrl} chars={chars} />
          </div>
          <div className="text-panel">
            <TranscriptionPanel
              transcription={transcription}
              onTranscriptionChange={setTranscription}
              onTranslate={translate}
              translating={translating}
              translation={translation}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default Home