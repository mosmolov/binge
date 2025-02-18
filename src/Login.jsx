import { useState } from 'react';
import { Input, Card, Button, Grid, Typography, Container } from '@mui/material';
import { CardContent, CardHeader } from '@mui/material';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Login attempt with:', { email, password });
    // Add authentication logic here
  };

  return (
    <Container
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        backgroundColor: 'grey.100',
      }}
    >
      <Card
        sx={{
          maxWidth: 450,
          width: '100%',
          padding: 3,
          borderRadius: 3,
          boxShadow: 24,
          backgroundColor: 'background.paper',
        }}
      >
        <CardHeader
          title="Login"
          sx={{
            textAlign: 'center',
            marginBottom: 3,
            fontSize: '1.5rem',
            fontWeight: 500,
            color: 'text.primary',
          }}
        />
        <CardContent>
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <label
                htmlFor="email"
                style={{
                  marginBottom: '8px',
                  fontSize: '1rem',
                  fontWeight: 'bold',
                  color: 'text.secondary',
                }}
              >
                Email
              </label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email"
                required
                fullWidth
                sx={{
                  padding: 1.5,
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: 'grey.300',
                  transition: 'border-color 0.3s ease-in-out',
                  '&:focus': {
                    borderColor: 'primary.main',
                  },
                }}
              />
            </div>
            <div>
              <label
                htmlFor="password"
                style={{
                  marginBottom: '8px',
                  fontSize: '1rem',
                  fontWeight: 'bold',
                  color: 'text.secondary',
                }}
              >
                Password
              </label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                required
                fullWidth
                sx={{
                  padding: 1.5,
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: 'grey.300',
                  transition: 'border-color 0.3s ease-in-out',
                  '&:focus': {
                    borderColor: 'primary.main',
                  },
                }}
              />
            </div>
            <Button
              type="submit"
              fullWidth
              sx={{
                marginTop: 2,
                padding: 1.5,
                backgroundColor: 'primary.main',
                color: '#fff',
                borderRadius: 2,
                '&:hover': {
                  backgroundColor: 'primary.dark',
                  transform: 'scale(1.05)',
                  transition: 'transform 0.3s ease-in-out',
                },
              }}
            >
              Login
            </Button>
          </form>
        </CardContent>
      </Card>
    </Container>
  );
}