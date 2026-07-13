import { useState, useEffect } from 'react'
import { Link, useRouter } from './Router'
import './Toolbar.css'

export default function Navbar({ onNavigateHome }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const { navigate } = useRouter()

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
      navigate('/')
      onNavigateHome(section)
    }
  }

  return (
    <nav className="navbar">
      <a href="/" className="brand" onClick={goHomeSection('home')}>
        <span className="brand-mark">墨奈</span>
        <span className="brand-word">
          <span className="brand-name">Sumina</span>
          <span className="brand-tagline">TRANSCRIBE · TRANSLATE EDO</span>
        </span>
      </a>

      <div className={`navbar-container ${menuOpen ? 'show-menu' : ''}`}>
        <ul className="navbar-links">
          <li><a href="/" onClick={goHomeSection('how')}>How it works</a></li>
          <li><a href="/" onClick={goHomeSection('examples')}>Examples</a></li>
          <li>
            <Link href="/about" activeClass="active" onClick={() => setMenuOpen(false)}>
              About
            </Link>
          </li>
          <li>
            <Link href="/policy" activeClass="active" onClick={() => setMenuOpen(false)}>
              Policy
            </Link>
          </li>
        </ul>
        <a href="/" className="navbar-cta" onClick={goHomeSection('try')}>
          Try it →
        </a>
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
