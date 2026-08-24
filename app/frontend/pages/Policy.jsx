import i18next from 'i18next'

const Policy = () => {
  const t = i18next.t.bind(i18next)
  const collectItems = t('policy.collectItems', { returnObjects: true })
  const useItems = t('policy.useItems', { returnObjects: true })
  const detailedSections = t('policy.detailedSections', { returnObjects: true }) || []

  return (
    <div className="page-content">
      <h1>{t('policy.title')}</h1>
      <p>{t('policy.intro')}</p>
      <h2>{t('policy.collectHeading')}</h2>
      <p>{t('policy.collectIntro')}</p>
      <ul>
        {collectItems.map((item, i) => <li key={i}>{item}</li>)}
      </ul>
      <h2>{t('policy.useHeading')}</h2>
      <p>{t('policy.useIntro')}</p>
      <ul>
        {useItems.map((item, i) => <li key={i}>{item}</li>)}
      </ul>
      <h2>{t('policy.thirdPartyHeading')}</h2>
      <p>{t('policy.thirdPartyText')}</p>
      <h2>{t('policy.protectionHeading')}</h2>
      <p>{t('policy.protectionText')}</p>

      {detailedSections.length > 0 && (
        <div>
          {detailedSections.map((section) => (
            <div key={section.heading}>
              <h2>{section.heading}</h2>
              {section.paragraphs.map((paragraph, index) => (
                <p key={`${section.heading}-${index}`}>{paragraph}</p>
              ))}
            </div>
          ))}
        </div>
      )}

      <p className="policy-meta">{t('policy.lastUpdated')}</p>
      <p className="policy-credit">{t('policy.credit')}</p>
    </div>
  )
}

export default Policy
