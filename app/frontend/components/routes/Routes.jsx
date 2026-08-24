import { Routes, Route, Navigate } from 'react-router'
import Home from '../../pages/Home'
import Workspace from '../../pages/Workspace'
import About from '../../pages/About'
import Policy from '../../pages/Policy'
import Impressum from '../../pages/Impressum'

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/workspace" element={<Workspace />} />
      <Route path="/about" element={<About />} />
      <Route path="/policy" element={<Policy />} />
      <Route path="/impressum" element={<Impressum />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
