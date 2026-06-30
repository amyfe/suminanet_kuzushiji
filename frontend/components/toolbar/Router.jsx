import { createContext, useContext, useState, useEffect } from 'react'

const RouterContext = createContext()

export function useRouter() {
  return useContext(RouterContext)
}

export function Router({ children }) {
  const normalize = (p) => (p.length > 1 && p.endsWith('/') ? p.slice(0, -1) : p) || '/'

  const [location, setLocation] = useState(() => normalize(window.location.pathname))

  function navigate(to) {
    const target = normalize(to)
    if (normalize(window.location.pathname) !== target) {
      window.history.pushState(null, '', target)
    }
    setLocation(target)
    window.scrollTo(0, 0)
  }

  useEffect(() => {
    function onPopState() {
      setLocation(normalize(window.location.pathname))
    }
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  return (
    <RouterContext.Provider value={{ location, navigate }}>
      {children}
    </RouterContext.Provider>
  )
}

export function Route({ path, children }) {
  const { location } = useRouter()
  return location === path ? children : null
}

export function Link({ href, children, activeClass, className, onClick }) {
  const { location, navigate } = useRouter()
  const isActive = location === href

  function handleClick(e) {
    e.preventDefault()
    if (onClick) onClick(e)
    navigate(href)
  }

  const cls = [className, isActive && activeClass].filter(Boolean).join(' ')
  return (
    <a href={href} onClick={handleClick} className={cls || undefined}>
      {children}
    </a>
  )
}
