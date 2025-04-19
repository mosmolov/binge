import { useState } from 'react';
import { Container, Card, CardContent, CardHeader, TextField, Button, Typography } from '@mui/material';
import { keyframes } from '@emotion/react';
import { useAuth } from './context/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

// Refined Gradient background animation
const gradientAnimation = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

// Subtle Fade-in animation for the card
const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(15px) scale(0.98); }
  to { opacity: 1; transform: translateY(0) scale(1); }
`;

// Smoother Wave animation keyframes
const waveAnimation = keyframes`
  0% { transform: translateX(0); }
  100% { transform: translateX(-50%); }
`;

// Wave component - adjusted opacity and color
const Wave = () => (
  <div style={{
    position: 'absolute',
    bottom: 0,
    left: 0,
    width: '100%',
    height: '150px', // Adjust height as needed
    overflow: 'hidden',
    zIndex: 1,
  }}>
    <svg viewBox="0 0 1200 120" preserveAspectRatio="none"
         style={{
           position: 'absolute',
           bottom: 0,
           left: 0,
           width: '200%', // Double width for seamless looping
           height: '100%',
           animation: `${waveAnimation} 12s linear infinite`, // Slightly slower animation
         }}>
      {/* Multiple wave layers for depth */}
      <path d="M0,50 C300,100 900,0 1200,50 L1200,120 L0,120 Z" fill="rgba(255, 255, 255, 0.5)" />
      <path d="M0,70 C400,120 800,20 1200,70 L1200,120 L0,120 Z" fill="rgba(255, 255, 255, 0.3)" />
    </svg>
  </div>
);


export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();
  const [error, setError] = useState(''); // State for login errors

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(''); // Clear previous errors
    try {
      await login(username, password);
      navigate('/'); // Navigate to dashboard or home on success
    } catch (err) {
      console.error('Login failed:', err);
      setError('Login failed. Please check your credentials.'); // Set user-friendly error message
    }
  };

  return (
    <Container
      maxWidth={false} // Ensure container takes full width
      disableGutters // Remove default padding
      sx={{
        position: 'relative',
        minHeight: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        // Updated, more modern gradient
        background: 'linear-gradient(-45deg, #6a11cb, #2575fc, #ec008c, #fc6767)',
        backgroundSize: '400% 400%',
        animation: `${gradientAnimation} 18s ease infinite`,
        overflow: 'hidden', // Hide overflow from wave/animations
      }}
    >
      {/* Animated wave background */}
      <Wave />

      {/* Login Card - Enhanced Styling */}
      <Card
        sx={{
          position: 'relative',
          zIndex: 2,
          width: { xs: '90%', sm: 420 }, // Slightly wider card
          padding: { xs: 2, sm: 4 }, // Responsive padding
          borderRadius: '16px', // More rounded corners
          // Frosted glass effect
          backgroundColor: 'rgba(255, 255, 255, 0.85)',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)', // Softer, more modern shadow
          border: '1px solid rgba(255, 255, 255, 0.18)',
          animation: `${fadeIn} 0.8s cubic-bezier(0.25, 0.8, 0.25, 1) forwards`, // Smoother fade-in
          opacity: 0, // Start hidden for animation
        }}
      >
        <CardHeader
          title="Welcome Back!"
          titleTypographyProps={{ variant: 'h4', fontWeight: 'bold', color: 'primary.main' }} // Use theme color
          sx={{
            textAlign: 'center',
            pb: 1, // Adjust padding
            '& .MuiCardHeader-title': {
              // Custom title styling if needed
            },
          }}
        />
        <CardContent>
          <form
            onSubmit={handleSubmit}
            style={{ display: 'flex', flexDirection: 'column', gap: '24px' }} // Increased gap
          >
            <TextField
              label="Username"
              variant="outlined"
              fullWidth
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '12px', // Match card rounding
                  transition: 'border-color 0.3s ease, box-shadow 0.3s ease', // Smooth transition
                  '&.Mui-focused fieldset': {
                    boxShadow: (theme) => `0 0 0 2px ${theme.palette.primary.light}`, // Subtle focus glow
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
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '12px', // Match card rounding
                  transition: 'border-color 0.3s ease, box-shadow 0.3s ease', // Smooth transition
                  '&.Mui-focused fieldset': {
                    boxShadow: (theme) => `0 0 0 2px ${theme.palette.primary.light}`, // Subtle focus glow
                  },
                },
              }}
            />
            {/* Display error message if login fails */}
            {error && (
              <Typography color="error" variant="body2" sx={{ textAlign: 'center', mt: -1, mb: 1 }}>
                {error}
              </Typography>
            )}
            <Button
              type="submit"
              variant="contained"
              fullWidth
              sx={{
                padding: '14px', // Slightly larger button
                borderRadius: '12px', // Match card rounding
                fontWeight: 'bold', // Bolder text
                fontSize: '1rem',
                textTransform: 'none', // Keep casing as is
                // Button gradient and hover effect
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
                transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-2px)', // Subtle lift effect
                  boxShadow: '0 6px 10px 4px rgba(33, 203, 243, .4)',
                },
              }}
            >
              Login
            </Button>
            <Typography variant="body2" sx={{ marginTop: '1rem', textAlign: 'center' }}>
              Don&apos;t have an account?{' '}
              <Link to="/register" style={{ color: '#2575fc', textDecoration: 'none', fontWeight: 'bold' }}>
                Register Now
              </Link>
            </Typography>
          </form>
        </CardContent>
      </Card>
    </Container>
  );
}
