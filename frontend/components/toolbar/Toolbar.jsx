import { useState, useEffect } from 'react'
import { Link } from './Router'
import './Toolbar.css'
import logo from '../../assets/japanese-at-edo-period.png'

const NAV_LINKS = [
  { href: '/',       label: 'Homepage' },
  { href: '/about',  label: 'Über die Webseite' },
  { href: '/policy', label: 'Richtlinien' },
]

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    function onResize() {
      if (window.innerWidth > 768) setMenuOpen(false)
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  return (
    <nav className="navbar">
      <div className="logo-container">
        <Link href="/">
          <img src={logo} alt="Kuzushiji Reader Logo" className="company-logo" />
        </Link>
      </div>

      <div className={`navbar-container ${menuOpen ? 'show-menu' : ''}`}>
        <ul className="navbar-links">
          {NAV_LINKS.map(({ href, label }) => (
            <li key={href}>
              <Link href={href} activeClass="active" onClick={() => setMenuOpen(false)}>
                {label}
              </Link>
            </li>
          ))}
        </ul>
      </div>

      <button
        type="button"
        className="menu-button"
        onClick={() => setMenuOpen((o) => !o)}
        aria-label="Menü öffnen"
      >
        ☰
      </button>
    </nav>
  )
}
