import './App.css'
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import Login from './Login';
import Register from './Register';
import SwipeImageUI from './screening';
import Recommendations from './Recommendations';

function App() {
  const { user } = useAuth();
  return (
    <Routes>
      <Route path="/login" element={!user ? <Login /> : <Navigate to="/" />} />
      <Route path="/register" element={!user ? <Register /> : <Navigate to="/" />} />
      <Route path="/" element={user ? <SwipeImageUI /> : <Navigate to="/login" />} />
      <Route path="/recommendations" element={user ? <Recommendations /> : <Navigate to="/login" />} />
    </Routes>
  );
}

export default App
