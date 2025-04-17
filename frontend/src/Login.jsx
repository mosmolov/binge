import { useState } from 'react';
import { Container, Card, CardContent, CardHeader, TextField, Button } from '@mui/material';
import { keyframes } from '@emotion/react';
import { useAuth } from './context/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

// Gradient background animation
const gradientAnimation = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

// Fade-in animation for the card
const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
`;

// Wave animation keyframes
const waveAnimation = keyframes`
  0% {
    transform: translateX(0);
  }
  100% {
    transform: translateX(-50%);
  }
`;

// Wave component for animated background
const Wave = () => (
  <div style={{
    position: 'absolute',
    bottom: 0,
    left: 0,
    width: '100%', // Container stays full width
    height: '150px',
    overflow: 'hidden',
    zIndex: 1,
  }}>
    <svg viewBox="0 0 1200 120" preserveAspectRatio="none"
         style={{
           width: '200%', // Increased width for smooth animation
           height: '100%',
           animation: `${waveAnimation} 10s linear infinite`,
         }}>
      <path d="M0,0 C300,60 900,-60 1200,0 L1200,120 L0,120 Z" fill="rgba(255,255,255,0.7)" />
    </svg>
  </div>
);


export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await login(username, password);
      navigate('/');
    } catch (err) {
      console.error('Login failed', err);
      // handle error UI
    }
  };

  return (
    <Container
      sx={{
        position: 'relative', // positioning context for absolute elements
        minHeight: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        background: 'linear-gradient(45deg, #e0f7fa, #80deea, #b2dfdb, #26a69a)',
        backgroundSize: '400% 400%',
        animation: `${gradientAnimation} 15s ease infinite`,
        padding: 2,
        overflow: 'hidden',
      }}
    >
      {/* Animated wave background */}
      <Wave />

      {/* Login Card */}
      <Card
        sx={{
          position: 'relative',
          zIndex: 2, // ensures the card appears above the wave
          width: { xs: '90%', sm: 400 },
          padding: 3,
          borderRadius: 4,
          boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
          animation: `${fadeIn} 1s ease-out`,
        }}
      >
        <CardHeader
          title="Login"
          sx={{
            textAlign: 'center',
            '& .MuiCardHeader-title': {
              fontSize: '1.8rem',
              fontWeight: 600,
              color: '#333',
            },
          }}
        />
        <CardContent>
          <form 
            onSubmit={handleSubmit} 
            style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}
          >
            <TextField
              label="Username"
              variant="outlined"
              fullWidth
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '8px',
                },
              }}
            />
            <TextField
              label="Password"
              type="password"
              variant="outlined"
              fullWidth
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '8px',
                },
              }}
            />
            <Button
              type="submit"
              variant="contained"
              fullWidth
              sx={{
                padding: '12px',
                borderRadius: '8px',
                fontWeight: 600,
                backgroundColor: '#1976d2',
                '&:hover': {
                  backgroundColor: '#115293',
                },
              }}
            >
              Login
            </Button>
            <p style={{ marginTop: '1rem', textAlign: 'center' }}>
              Don&apos;t have an account? <Link to="/register">Register</Link>
            </p>
          </form>
        </CardContent>
      </Card>
    </Container>
  );
}
