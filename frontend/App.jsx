import './style.css'
import { Router, Route } from './components/toolbar/Router'
import Navbar from './components/toolbar/Toolbar'
import Home from './pages/Home'
import About from './pages/About'
import Policy from './pages/Policy'

export default function App() {
  return (
    <Router>
    <div className="app">
      <Navbar />
      <div style={{ paddingTop: '56px' }}>
        <Route path="/">
          <Home />
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
