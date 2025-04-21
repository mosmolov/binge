import './App.css'
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import Login from './Login';
import Register from './Register';
import SwipeImageUI from './screening';
import Recommendations from './Recommendations';
import Profile from './Profile';
import { useState } from 'react';
import { Fab, Dialog, DialogContent } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import AddRestaurant from './AddRestaurant';

function App() {
  const { user } = useAuth();
  const [openModal, setOpenModal] = useState(false);

  const handleOpenModal = () => {
    setOpenModal(true);
  };

  const handleCloseModal = () => {
    setOpenModal(false);
  };

  return (
    <>
      <Routes>
        <Route path="/profile" element={user ? <Profile /> : <Navigate to="/login" />} />
        <Route path="/login" element={!user ? <Login /> : <Navigate to="/" />} />
        <Route path="/register" element={!user ? <Register /> : <Navigate to="/" />} />
        <Route path="/" element={user ? <SwipeImageUI /> : <Navigate to="/login" />} />
        <Route path="/recommendations" element={user ? <Recommendations /> : <Navigate to="/login" />} />
      </Routes>

      {user && (
        <>
          <Fab 
            color="primary" 
            aria-label="add restaurant" 
            onClick={handleOpenModal}
            sx={{ 
              position: 'fixed', 
              bottom: 16, 
              right: 16
            }}
          >
            <AddIcon />
          </Fab>

          <Dialog 
            open={openModal} 
            onClose={handleCloseModal} 
            maxWidth="sm" 
            fullWidth
          >
            <DialogContent>
              <AddRestaurant onClose={handleCloseModal} />
            </DialogContent>
          </Dialog>
        </>
      )}
    </>
  );
}

export default App
