import { useState, useRef } from 'react'

export default function UploadArea({ imageUrl, onFile }) {
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return
    onFile(file)
  }

  return (
    <div
      className={`upload-area ${dragOver ? 'drag-over' : ''} ${imageUrl ? 'has-image' : ''}`}
      onClick={() => fileInputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) }}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => handleFile(e.target.files[0])}
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
  )
}
