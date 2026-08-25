import { useState, useEffect } from 'react'
import { NavLink, useNavigate } from 'react-router'
import i18next from 'i18next'
import './Toolbar.css'

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
          {/* <li>
            <NavLink to="/policy" className={({ isActive }) => (isActive ? 'active' : undefined)} onClick={() => setMenuOpen(false)}>
              {t('toolbar.policy')}
            </NavLink>
          </li> */}
          <li>
            <NavLink to="/impressum" className={({ isActive }) => (isActive ? 'active' : undefined)} onClick={() => setMenuOpen(false)}>
              {t('toolbar.impressum')}
            </NavLink>
          </li>
          <li className="navbar-lang">
            <label className="lang-switch">
              <input
                type="checkbox"
                role="switch"
                checked={i18next.language === 'de'}
                onChange={(e) => changeLanguage(e.target.checked ? 'de' : 'en')}
                aria-label={i18next.language === 'de' ? 'Switch language to English' : 'Switch language to German'}
              />
              <span className="lang-switch-track" aria-hidden="true">
                <span className="lang-switch-option">EN</span>
                <span className="lang-switch-option">DE</span>
                <span className="lang-switch-thumb" />
              </span>
            </label>
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
