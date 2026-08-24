import { Link } from 'react-router'
import i18next from 'i18next'

const Impressum = () => {
  const t = i18next.t.bind(i18next)
  const addressLines = t('impressum.addressLines', { returnObjects: true }) || []

  return (
    <div className="page-content">
      <h1>{t('impressum.title')}</h1>
      <h2>{t('impressum.heading')}</h2>
      <p>
        {t('impressum.nameLabel')}
        <br />
        {addressLines.map((line, i) => (
          <span key={i}>
            {line}
            <br />
          </span>
        ))}
      </p>

      <h2>{t('impressum.contactHeading')}</h2>
      <p>{t('impressum.emailLabel')}</p>

      <h2>{t('impressum.natureHeading')}</h2>
      <p>{t('impressum.natureText')}</p>

      <h2>{t('impressum.institutionHeading')}</h2>
      <p>{t('impressum.institutionText')}</p>

      <h2>{t('impressum.responsibleHeading')}</h2>
      <p>{t('impressum.responsibleText')}</p>

      <h2>{t('impressum.liabilityContentHeading')}</h2>
      <p>{t('impressum.liabilityContentText')}</p>

      <h2>{t('impressum.liabilityLinksHeading')}</h2>
      <p>{t('impressum.liabilityLinksText')}</p>

      <p>
        {t('impressum.policyLinkText')} <Link to="/policy">{t('toolbar.policy')}</Link>
      </p>
    </div>
  )
}

export default Impressum
