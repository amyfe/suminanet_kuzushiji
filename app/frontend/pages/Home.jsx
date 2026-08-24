import { columnsOf } from '../utils/text'
import { useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router'
import i18next from 'i18next'

// Decorative step markers — fixed regardless of UI language, keyed by the
// numeric "step" field in the translated how_steps array.
const KANJI_BY_STEP = { 1: '一', 2: '二', 3: '三' }

function scrollToId(id) {
  const el = document.getElementById(id)
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

const Home = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const t = i18next.t.bind(i18next)
  const howSteps = t('home.how_steps', { returnObjects: true })
  const examples = t('home.examples', { returnObjects: true })

  // Navbar links to "/#how" / "/#examples" (or "/#top" for the brand) from
  // anywhere in the app; once here, scroll to the target section and clear
  // the hash so the same link can be clicked again later.
  useEffect(() => {
    if (!location.hash) return
    const id = location.hash.slice(1)
    if (id === 'top') {
      window.scrollTo({ top: 0 })
    } else {
      requestAnimationFrame(() => requestAnimationFrame(() => scrollToId(id)))
    }
    navigate(location.pathname, { replace: true })
  }, [location.hash])

  function loadExample(example) {
    navigate('/workspace', { state: { example } })
  }

  return (
    <div className="landing">
      <section className="hero">
        <div className="hero-eyebrow">{t('home.heroEyebrow')}</div>
        <h1 className="hero-title">{t('home.heroTitle')}</h1>
        <p className="hero-sub">{t('home.heroSub')}</p>
        <div className="hero-cta">
          <button className="btn btn-primary" onClick={() => navigate('/workspace')}>
            {t('home.heroCta')}
          </button>
        </div>
      </section>

      <section id="how">
        <h2 className="section-title">{t('home.howTitle')}</h2>
        <div className="how-grid">
          {howSteps.map((s) => (
            <div className="how-step" key={s.step}>
              <div className="how-kanji">{KANJI_BY_STEP[s.step]}</div>
              <div className="how-step-title">{s.title}</div>
              <p className="how-step-body">{s.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="examples">
        <div className="section-eyebrow">{t('home.examplesEyebrow')}</div>
        <h2 className="section-title">{t('home.examplesTitle')}</h2>
        <p className="section-sub">{t('home.examplesSub')}</p>
        <div className="examples-grid">
          {examples.map((ex) => (
            <button
              type="button"
              className="example-card"
              key={ex.id}
              onClick={() => loadExample(ex)}
            >
              <div className="example-thumb">
                <div className="example-cols">
                  {columnsOf(ex.text, 3).map((col, ci) => (
                    <div className="example-col" key={ci}>
                      {col.map((ch, ri) => (
                        <span className="example-char" key={ri}>{ch}</span>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
              <div className="example-meta">
                <div className="example-title">{ex.title}</div>
                <div className="example-sub">{ex.author} · {ex.year}</div>
                <div className="example-cta">{t('home.exampleCta')}</div>
              </div>
            </button>
          ))}
        </div>
        <p className="section-sub">
          {t('home.furtherExamplesPre')}{' '}
          <a
            href="https://codh.rois.ac.jp/char-shape/"
            target="_blank"
            rel="noopener noreferrer"
          >
            {t('home.furtherExamplesLink')}
          </a>
          .
        </p>
      </section>

      <div className="site-footer">
        <span>{t('home.footerCredit')}</span>
        <span>{t('home.footerPowered')}</span>
      </div>
    </div>
  )
}

export default Home
