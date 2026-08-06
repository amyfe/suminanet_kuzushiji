import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import i18next from 'i18next';
import enTranslation from './locales/en/translation.json';
import deTranslation from './locales/de/translation.json';

const storedLng = localStorage.getItem('i18nextLng');
const initialLng = ['de', 'en'].includes(storedLng) ? storedLng : 'en';

i18next.init({
  lng: initialLng,
  fallbackLng: 'en',
  resources: {
    en: { translation: enTranslation },
    de: { translation: deTranslation },
  },
});

createRoot(document.getElementById('app')).render(
  <StrictMode>
    <App />
  </StrictMode>
)
