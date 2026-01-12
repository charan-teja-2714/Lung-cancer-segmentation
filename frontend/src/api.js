import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds for image processing
})

export const uploadImage = async (file) => {
  try {
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      responseType: 'blob', // Important for image response
    })

    return response.data
  } catch (error) {
    if (error.response) {
      // Server responded with error status
      const errorText = await error.response.data.text()
      throw new Error(`Server error: ${errorText}`)
    } else if (error.request) {
      // Request made but no response
      throw new Error('No response from server. Please check if the backend is running.')
    } else {
      // Something else happened
      throw new Error(`Request failed: ${error.message}`)
    }
  }
}

export const checkServerHealth = async () => {
  try {
    const response = await api.get('/')
    return response.data
  } catch (error) {
    throw new Error('Backend server is not responding')
  }
}