import React, { useState } from 'react'
import { uploadImage } from './api'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [resultUrl, setResultUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResultUrl(null)
      setError(null)
    }
  }

  const handleSegment = async () => {
    if (!selectedFile) {
      setError('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const resultBlob = await uploadImage(selectedFile)
      const resultUrl = URL.createObjectURL(resultBlob)
      setResultUrl(resultUrl)
    } catch (err) {
      setError(err.message || 'Segmentation failed')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setResultUrl(null)
    setError(null)
    document.getElementById('file-input').value = ''
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Lung Cancer Segmentation</h1>
        <p>Multi-class segmentation using nnU-Net 2D</p>
      </header>

      <main className="main">
        <div className="upload-section">
          <div className="file-input-container">
            <input
              id="file-input"
              type="file"
              accept="image/jpeg,image/jpg"
              onChange={handleFileSelect}
              className="file-input"
            />
            <label htmlFor="file-input" className="file-input-label">
              Choose CT Image (JPG)
            </label>
          </div>

          <div className="button-group">
            <button
              onClick={handleSegment}
              disabled={!selectedFile || loading}
              className="segment-button"
            >
              {loading ? 'Segmenting...' : 'Segment Tumor'}
            </button>
            
            <button
              onClick={handleReset}
              className="reset-button"
            >
              Reset
            </button>
          </div>
        </div>

        {error && (
          <div className="error-message">
            <p>Error: {error}</p>
          </div>
        )}

        <div className="results-section">
          {previewUrl && (
            <div className="image-container">
              <h3>Original Image</h3>
              <img src={previewUrl} alt="Original CT scan" className="result-image" />
            </div>
          )}

          {resultUrl && (
            <div className="image-container">
              <h3>Segmentation Result</h3>
              <img src={resultUrl} alt="Segmented image" className="result-image" />
              <div className="legend">
                <h4>Color Legend:</h4>
                <div className="legend-item">
                  <span className="color-box red"></span>
                  <span>Adenocarcinoma (ADC)</span>
                </div>
                <div className="legend-item">
                  <span className="color-box green"></span>
                  <span>Large Cell Carcinoma (LCC)</span>
                </div>
                <div className="legend-item">
                  <span className="color-box blue"></span>
                  <span>Squamous Cell Carcinoma (SCC)</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {loading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Processing image...</p>
          </div>
        )}
      </main>
    </div>
  )
}

export default App