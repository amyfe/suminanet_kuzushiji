import './style.css'
import { useState } from 'react'
import { Router, Route } from './components/toolbar/Router'
import Navbar from './components/toolbar/Toolbar'
import Home from './pages/Home'
import About from './pages/About'
import Policy from './pages/Policy'

export default function App() {
  const [homeIntent, setHomeIntent] = useState(null)

  return (
    <Router>
    <div className="app">
      <Navbar onNavigateHome={setHomeIntent} />
      <div style={{ paddingTop: '56px' }}>
        <Route path="/">
          <Home intent={homeIntent} onIntentHandled={() => setHomeIntent(null)} />
        </Route>
        <Route path="/about">
          <About />
        </Route>
        <Route path="/policy">
          <Policy />
        </Route>
      </div>
    </div>
    </Router>
  )
}
