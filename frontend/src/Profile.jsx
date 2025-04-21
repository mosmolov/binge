import React, { useState } from 'react';
import { Container, Typography, Button, List, ListItem, ListItemAvatar, Avatar, ListItemText, CircularProgress, Alert, Box, Rating, Paper } from '@mui/material';
import { keyframes } from '@emotion/react';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import StarIcon from '@mui/icons-material/Star';
import { useAuth } from './context/AuthContext';
import { useNavigate } from 'react-router-dom';

const gradientAnimation = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

export default function Profile() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [recs, setRecs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchRecommendations = async () => {
    setLoading(true);
    setError('');
    try {
      // get user location
      const pos = await new Promise((res, rej) => navigator.geolocation.getCurrentPosition(res, rej));
      const { latitude, longitude } = pos.coords;
      const response = await fetch(`http://localhost:8000/recommendations?user_id=${user.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_location: [latitude, longitude], radius_miles: 10, top_n: 25, liked_ids: [], disliked_ids: [] })
      });
      if (!response.ok) throw new Error(`Status ${response.status}`);
      const data = await response.json();
      setRecs(data.recommendations);
    } catch (e) {
      setError(e.message || 'Failed to load recommendations');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        width: '100vw',
        background: 'linear-gradient(-45deg, #c1dfc4 0%, #deecdd 50%, #c1dfc4 100%)',
        backgroundSize: '400% 400%',
        animation: `${gradientAnimation} 20s ease infinite`,
        py: { xs: 4, sm: 6 }
      }}
    >
      <Container maxWidth="sm" disableGutters>
        <Paper elevation={4} sx={{ p: 3, mt: 4, borderRadius: 2 }}>
          <Typography variant="h4" gutterBottom>Profile</Typography>
          {user && (
            <Box sx={{ mb: 2 }}>
              <Typography><strong>Username:</strong> {user.username}</Typography>
              <Typography><strong>Email:</strong> {user.email}</Typography>
            </Box>
          )}
          <Button variant="contained" onClick={fetchRecommendations} sx={{ mb: 2 }}>Load Recommendations</Button>
          {loading && <CircularProgress />}
          {error && <Alert severity="error">{error}</Alert>}
          {!loading && recs.length > 0 && (
            <List>
              {recs.map((rec, idx) => (
                <React.Fragment key={idx}>
                  <ListItem alignItems="flex-start">
                    <ListItemAvatar>
                      <Avatar alt={rec.restaurant.name} src={rec.restaurant.photo_url} />
                    </ListItemAvatar>
                    <ListItemText
                      primary={rec.restaurant.name}
                      secondary={<>
                        <Typography variant="body2" color="text.secondary">{rec.restaurant.address}</Typography>
                        <Rating value={rec.restaurant.stars || 0} readOnly precision={0.5} emptyIcon={<StarIcon fontSize="inherit" />} sx={{ color: '#ffb400' }} />
                      </>}
                    />
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          )}
          <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/')} sx={{ mt: 2 }}>Back</Button>
          <Button onClick={logout} sx={{ ml: 2, mt: 2 }}>Logout</Button>
        </Paper>
      </Container>
    </Box>
  );
}
