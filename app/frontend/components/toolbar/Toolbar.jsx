import { useState, useEffect } from 'react'
import { NavLink, useNavigate } from 'react-router'
import i18next from 'i18next'
import './Toolbar.css'

const LANGUAGES = [
  { code: 'en', label: 'EN' },
  { code: 'de', label: 'DE' },
]

// Plain i18next (no react-i18next), so nothing re-renders on language change
// on its own — persist the choice and reload to re-render the whole app
// with the newly active language.
function changeLanguage(code) {
  localStorage.setItem('i18nextLng', code)
  window.location.reload()
}

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false)
  const navigate = useNavigate()
  const t = i18next.t.bind(i18next)

  useEffect(() => {
    function onResize() {
      if (window.innerWidth > 768) setMenuOpen(false)
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  function goHomeSection(section) {
    return (e) => {
      e.preventDefault()
      setMenuOpen(false)
      navigate(`/#${section}`)
    }
  }

  return (
    <nav className="navbar">
      <a href="/" className="brand" onClick={goHomeSection('top')}>
        <span className="brand-mark">墨奈</span>
        <span className="brand-word">
          <span className="brand-name">{t('toolbar.brandName')}</span>
          <span className="brand-tagline">{t('toolbar.brandTagline')}</span>
        </span>
      </a>

      <div className={`navbar-container ${menuOpen ? 'show-menu' : ''}`}>
        <ul className="navbar-links">
          <li><a href="/#how" onClick={goHomeSection('how')}>{t('toolbar.howItWorks')}</a></li>
          <li><a href="/#examples" onClick={goHomeSection('examples')}>{t('toolbar.examples')}</a></li>
          <li>
            <NavLink to="/about" className={({ isActive }) => (isActive ? 'active' : undefined)} onClick={() => setMenuOpen(false)}>
              {t('toolbar.about')}
            </NavLink>
          </li>
          <li>
            <NavLink to="/policy" className={({ isActive }) => (isActive ? 'active' : undefined)} onClick={() => setMenuOpen(false)}>
              {t('toolbar.policy')}
            </NavLink>
          </li>
          <li className="navbar-lang">
            {LANGUAGES.map((lang) => (
              <button
                key={lang.code}
                type="button"
                className={`lang-btn ${i18next.language === lang.code ? 'active' : ''}`}
                onClick={() => changeLanguage(lang.code)}
                aria-label={`Switch language to ${lang.label}`}
              >
                {lang.label}
              </button>
            ))}
          </li>
        </ul>
        <NavLink to="/workspace" className="navbar-cta" onClick={() => setMenuOpen(false)}>
          {t('toolbar.tryIt')}
        </NavLink>
      </div>

      <button
        type="button"
        className="menu-button"
        onClick={() => setMenuOpen((o) => !o)}
        aria-label="Toggle menu"
      >
        ☰
      </button>
    </nav>
  )
}
