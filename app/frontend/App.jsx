import './style.css'
import './components/lang-flags.css'
import { BrowserRouter } from 'react-router'
import Navbar from './components/toolbar/Toolbar'
import AppRoutes from './components/routes/Routes'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <Navbar />
        <div style={{ paddingTop: '56px' }}>
          <AppRoutes />
        </div>
      </div>
    </BrowserRouter>
  )
}
