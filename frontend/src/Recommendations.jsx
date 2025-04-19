import { Container, Card, CardContent, CardHeader, Button, Avatar, Typography, Box } from '@mui/material';
import { keyframes } from '@emotion/react';
import { useLocation, useNavigate } from 'react-router-dom';

// Gradient background animation
const gradientAnimation = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

export default function Recommendations() {
  const location = useLocation();
  const navigate = useNavigate();
  const recommendations = location.state?.recommendations || [];

  return (
    <Container
      sx={{
        position: 'relative',
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
      <Card
        sx={{
          position: 'relative',
          zIndex: 2,
          width: { xs: '90%', sm: 500 },
          padding: 3,
          borderRadius: 4,
          boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
        }}
      >
        <CardHeader
          title="Your Top Foodie Picks"
          sx={{ textAlign: 'center', '& .MuiCardHeader-title': { fontSize: '1.8rem', fontWeight: 600, color: '#333' } }}
        />
        <CardContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {recommendations.length === 0 && (
              <Typography variant="body1" color="text.secondary" align="center">
                No recommendations found.
              </Typography>
            )}
            {recommendations.map((rec, index) => (
              <Card key={index} sx={{ display: 'flex', alignItems: 'center', p: 2, borderRadius: 4, boxShadow: 2, mb: 2 }}>
                <Avatar
                  variant="rounded"
                  src={rec.restaurant.photo_url || '/foodtest.jpeg'}
                  alt={rec.restaurant.name}
                  sx={{ width: 80, height: 80, mr: 3 }}
                />
                <Box>
                  <Typography variant="h6" fontWeight={700}>{rec.restaurant.name}</Typography>
                  <Typography variant="body2" color="text.secondary">{rec.restaurant.address}</Typography>
                  <Typography variant="body2" color="#f8b500">⭐ {rec.restaurant.stars}</Typography>
                </Box>
              </Card>
            ))}
          </Box>
          <Button
            fullWidth
            variant="contained"
            sx={{ mt: 3, borderRadius: 3, fontWeight: 600, backgroundColor: '#1976d2', '&:hover': { backgroundColor: '#115293' } }}
            onClick={() => navigate('/')}
          >
            Start New Session
          </Button>
        </CardContent>
      </Card>
    </Container>
  );
}
