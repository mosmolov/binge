import { useState } from 'react';
import { Container, Card, CardContent, CardHeader, TextField, Button } from '@mui/material';
import { keyframes } from '@emotion/react';
import { useAuth } from './context/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

// Use same animations as Login
const gradientAnimation = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;
const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
`;

export default function Register() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await register(username, email, password);
      navigate('/login');
    } catch (err) {
      console.error('Registration failed', err);
      // handle error UI
    }
  };

  return (
    <Container
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        background: 'linear-gradient(45deg, #ffe0b2, #ffcc80, #ffe0b2, #ffb74d)',
        backgroundSize: '400% 400%',
        animation: `${gradientAnimation} 15s ease infinite`,
        padding: 2,
      }}
    >
      <Card
        sx={{
          width: { xs: '90%', sm: 400 },
          padding: 3,
          borderRadius: 4,
          boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
          animation: `${fadeIn} 1s ease-out`,
        }}
      >
        <CardHeader
          title="Register"
          sx={{ textAlign: 'center', '& .MuiCardHeader-title': { fontSize: '1.8rem', fontWeight: 600 } }}
        />
        <CardContent>
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <TextField
              label="Username"
              variant="outlined"
              fullWidth
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              sx={{ '& .MuiOutlinedInput-root': { borderRadius: '8px' } }}
            />
            <TextField
              label="Email"
              type="email"
              variant="outlined"
              fullWidth
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              sx={{ '& .MuiOutlinedInput-root': { borderRadius: '8px' } }}
            />
            <TextField
              label="Password"
              type="password"
              variant="outlined"
              fullWidth
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              sx={{ '& .MuiOutlinedInput-root': { borderRadius: '8px' } }}
            />
            <Button type="submit" variant="contained" fullWidth sx={{ padding: '12px', borderRadius: '8px', fontWeight: 600 }}>
              Register
            </Button>
            <p style={{ marginTop: '1rem', textAlign: 'center' }}>
              Already have an account? <Link to="/login">Login</Link>
            </p>
          </form>
        </CardContent>
      </Card>
    </Container>
  );
}
