const API_BASE = import.meta.env.VITE_API_BASE || ''
const API_KEY = import.meta.env.VITE_API_KEY || ''

export async function apiFetch(path, options = {}) {
  const headers = { ...options.headers }
  if (API_KEY) headers['X-API-Key'] = API_KEY
  return fetch(`${API_BASE}${path}`, { ...options, headers })
}
