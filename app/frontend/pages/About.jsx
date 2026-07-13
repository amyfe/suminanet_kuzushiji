import edoImg from '../assets/japanese-at-edo-period.png'

const About = () => {
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
          <div className="section-eyebrow" style={{ textAlign: 'left' }}>ABOUT · 沿革</div>
          <h2>Why Sumina exists</h2>
          <p>
            Most people — even fluent readers of modern Japanese — cannot read{' '}
            <em>kuzushiji</em>, the flowing cursive in which nearly every Edo-period book,
            letter, and ledger was written. Centuries of literature, science, and ordinary life
            sit locked behind a hand almost no one is trained to decipher.
          </p>
          <p>
            Sumina pairs <b>KuroNet</b> — the character-recognition model built by Tarin
            Clanuwat and collaborators — with a human correction step and translation, so a
            scan becomes a reading, and a reading becomes English. Made for researchers,
            students, and anyone curious enough to look closely.
          </p>
          <div className="about-etymology">
            墨奈 Sumina — 幸 by Amina Felic.
            <br />
            The name fuses 墨 <b>sumi</b> ("ink", after KuroNet's 黒 <b>kuro</b>, "black")
            with <b>Amina</b>; 幸 <b>sachi</b> ("fortune") for Felic.
          </div>
          <img src={edoImg} alt="Illustration of daily life in the Edo period" className="about-photo" />
        </div>
      </section>
    </div>
  )
}

export default About
