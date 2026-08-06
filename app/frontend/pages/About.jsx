import edoImg from '../assets/japanese-at-edo-period.png'
import i18next from 'i18next'

// Renders a translation segment array (plain text plus { em } / { b }
// inline-emphasis markers) so bold/italic styling survives translation
// without needing dangerouslySetInnerHTML.
function renderSegments(segments) {
  return segments.map((seg, i) => {
    if (seg.em) return <em key={i}>{seg.em}</em>
    if (seg.b) return <b key={i}>{seg.b}</b>
    return <span key={i}>{seg.text}</span>
  })
}

const About = () => {
  const t = i18next.t.bind(i18next)
  const p1 = t('about.p1', { returnObjects: true })
  const p2 = t('about.p2', { returnObjects: true })
  const etymology = t('about.etymology', { returnObjects: true })

  return (
    <div className="landing" style={{ paddingTop: '3rem' }}>
      <section className="about-block" style={{ borderTop: 'none', padding: '0 2rem' }}>
        <div className="about-seal">
          <div className="about-seal-inner">
            古きを読み、
            <br />
            今に渡す
          </div>
        </div>
        <div className="about-copy">
          <div className="section-eyebrow" style={{ textAlign: 'left' }}>{t('about.eyebrow')}</div>
          <h2>{t('about.title')}</h2>
          <p>{renderSegments(p1)}</p>
          <p>{renderSegments(p2)}</p>
          <div className="about-etymology">
            {t('about.creditLine')}
            <br />
            {renderSegments(etymology)}
          </div>
          <img src={edoImg} alt={t('about.photoAlt')} className="about-photo" />
        </div>
      </section>
    </div>
  )
}

export default About
