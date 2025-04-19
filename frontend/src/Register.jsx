import { useState } from 'react';
import { Container, Card, CardContent, CardHeader, TextField, Button, Typography, Alert } from '@mui/material';
import { keyframes } from '@emotion/react';
import { useAuth } from './context/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

// Re-use animations from Login.jsx for consistency
const gradientAnimation = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(15px) scale(0.98); }
  to { opacity: 1; transform: translateY(0) scale(1); }
`;

const waveAnimation = keyframes`
  0% { transform: translateX(0); }
  100% { transform: translateX(-50%); }
`;

const Wave = () => (
  <div style={{
    position: 'absolute',
    bottom: 0,
    left: 0,
    width: '100%',
    height: '150px',
    overflow: 'hidden',
    zIndex: 1,
  }}>
    <svg viewBox="0 0 1200 120" preserveAspectRatio="none"
         style={{
           position: 'absolute',
           bottom: 0,
           left: 0,
           width: '200%',
           height: '100%',
           animation: `${waveAnimation} 12s linear infinite`,
         }}>
      <path d="M0,50 C300,100 900,0 1200,50 L1200,120 L0,120 Z" fill="rgba(255, 255, 255, 0.5)" />
      <path d="M0,70 C400,120 800,20 1200,70 L1200,120 L0,120 Z" fill="rgba(255, 255, 255, 0.3)" />
    </svg>
  </div>
);

export default function Register() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const { register } = useAuth();
  const navigate = useNavigate();
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (password !== confirmPassword) {
      setError("Passwords don't match.");
      return;
    }

    try {
      await register(username, password);
      setSuccess('Registration successful! Redirecting to login...');
      setTimeout(() => navigate('/login'), 2000); // Redirect after 2 seconds
    } catch (err) {
      console.error('Registration failed:', err);
      // Attempt to parse backend error message
      const errorMessage = err.response?.data?.detail || 'Registration failed. Please try again.';
      setError(errorMessage);
    }
  };

  return (
    <Container
      maxWidth={false}
      disableGutters
      sx={{
        position: 'relative',
        minHeight: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        background: 'linear-gradient(-45deg, #6a11cb, #2575fc, #ec008c, #fc6767)',
        backgroundSize: '400% 400%',
        animation: `${gradientAnimation} 18s ease infinite`,
        overflow: 'hidden',
      }}
    >
      <Wave />
      <Card
        sx={{
          position: 'relative',
          zIndex: 2,
          width: { xs: '90%', sm: 450 }, // Slightly wider for more fields
          padding: { xs: 2, sm: 4 },
          borderRadius: '16px',
          backgroundColor: 'rgba(255, 255, 255, 0.85)',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
          border: '1px solid rgba(255, 255, 255, 0.18)',
          animation: `${fadeIn} 0.8s cubic-bezier(0.25, 0.8, 0.25, 1) forwards`,
          opacity: 0,
        }}
      >
        <CardHeader
          title="Create Account"
          titleTypographyProps={{ variant: 'h4', fontWeight: 'bold', color: 'primary.main' }}
          sx={{ textAlign: 'center', pb: 1 }}
        />
        <CardContent>
          <form
            onSubmit={handleSubmit}
            style={{ display: 'flex', flexDirection: 'column', gap: '20px' }} // Adjusted gap
          >
            <TextField
              label="Username"
              variant="outlined"
              fullWidth
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '12px',
                  transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
                  '&.Mui-focused fieldset': {
                    boxShadow: (theme) => `0 0 0 2px ${theme.palette.primary.light}`,
                  },
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
              required
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '12px',
                  transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
                  '&.Mui-focused fieldset': {
                    boxShadow: (theme) => `0 0 0 2px ${theme.palette.primary.light}`,
                  },
                },
              }}
            />
            <TextField
              label="Confirm Password"
              type="password"
              variant="outlined"
              fullWidth
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              error={!!error && error.includes('Passwords don\'t match')} // Highlight if passwords mismatch
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '12px',
                  transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
                  '&.Mui-focused fieldset': {
                    boxShadow: (theme) => `0 0 0 2px ${theme.palette.primary.light}`,
                  },
                },
              }}
            />
            {/* Display error or success messages */}
            {error && (
              <Alert severity="error" sx={{ borderRadius: '8px' }}>{error}</Alert>
            )}
            {success && (
              <Alert severity="success" sx={{ borderRadius: '8px' }}>{success}</Alert>
            )}
            <Button
              type="submit"
              variant="contained"
              fullWidth
              sx={{
                padding: '14px',
                borderRadius: '12px',
                fontWeight: 'bold',
                fontSize: '1rem',
                textTransform: 'none',
                background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)', // Different gradient for Register
                boxShadow: '0 3px 5px 2px rgba(255, 105, 135, .3)',
                transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: '0 6px 10px 4px rgba(255, 105, 135, .4)',
                },
              }}
            >
              Register
            </Button>
            <Typography variant="body2" sx={{ marginTop: '1rem', textAlign: 'center' }}>
              Already have an account?{' '}
              <Link to="/login" style={{ color: '#2575fc', textDecoration: 'none', fontWeight: 'bold' }}>
                Login Here
              </Link>
            </Typography>
          </form>
        </CardContent>
      </Card>
    </Container>
  );
}
