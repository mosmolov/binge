import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Button, 
  List, 
  ListItem, 
  ListItemAvatar, 
  Avatar, 
  ListItemText, 
  CircularProgress, 
  Alert, 
  Box, 
  Rating, 
  Paper, 
  Divider, 
  Card, 
  CardContent, 
  Fade,
  IconButton,
  Grid
} from '@mui/material';
import { keyframes } from '@emotion/react';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import StarIcon from '@mui/icons-material/Star';
import LogoutIcon from '@mui/icons-material/Logout';
import RestaurantMenuIcon from '@mui/icons-material/RestaurantMenu';
import { useAuth } from './context/AuthContext';
import { useNavigate } from 'react-router-dom';

// Consistent gradient animation with other pages
const gradientAnimation = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

// Fade-in animation for profile content
const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
`;

export default function Profile() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [recs, setRecs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [userStats, setUserStats] = useState({
    totalLikes: 0,
    totalDislikes: 0,
  });

  // Load user statistics on component mount
  useEffect(() => {
    const fetchUserStats = async () => {
      if (!user || !user.id) return;
      
      try {
        const response = await fetch(`http://localhost:8000/users/${user.id}`);
        if (response.ok) {
          const userData = await response.json();
          setUserStats({
            totalLikes: userData.liked_business_ids?.length || 0,
            totalDislikes: userData.disliked_business_ids?.length || 0,
          });
        }
      } catch (err) {
        console.error("Error fetching user stats:", err);
      }
    };
    
    fetchUserStats();
  }, [user]);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError('');
    try {
      // Get user location
      const pos = await new Promise((res, rej) => {
        navigator.geolocation.getCurrentPosition(res, rej);
      });
      
      const { latitude, longitude } = pos.coords;
      const response = await fetch(`http://localhost:8000/recommendations?user_id=${user.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          user_location: [latitude, longitude], 
          radius_miles: 10, 
          top_n: 25, 
          liked_ids: [], 
          disliked_ids: [] 
        })
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
      <Container maxWidth="md" sx={{ px: { xs: 2, sm: 3 } }}>
        <Fade in={true} timeout={800}>
          <Paper 
            elevation={4} 
            sx={{ 
              borderRadius: '16px', 
              overflow: 'hidden',
              backgroundColor: 'rgba(255, 255, 255, 0.9)',
              backdropFilter: 'blur(8px)',
              boxShadow: '0 8px 32px rgba(31, 38, 135, 0.15)',
              animation: `${fadeIn} 0.5s ease-out forwards`,
              mb: 3
            }}
          >
            <Box sx={{ 
              p: 3, 
              display: 'flex', 
              alignItems: 'center', 
              borderBottom: '1px solid rgba(0,0,0,0.08)',
              background: 'linear-gradient(to right, #f5f7fa, #c3cfe2)',
            }}>
              <RestaurantMenuIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
              <Typography variant="h4" component="h1" fontWeight="600" color="primary.dark">
                My Profile
              </Typography>
              <Box sx={{ ml: 'auto', display: 'flex' }}>
                <IconButton 
                  onClick={() => navigate('/')}
                  color="primary"
                  sx={{ mr: 1 }}
                  aria-label="back to swipes"
                >
                  <ArrowBackIcon />
                </IconButton>
                <IconButton 
                  onClick={logout} 
                  color="error"
                  aria-label="logout"
                >
                  <LogoutIcon />
                </IconButton>
              </Box>
            </Box>
            
            <CardContent sx={{ p: 4 }}>
              {user && (
                <Grid container spacing={3}>
                  <Grid item xs={12} sm={6}>
                    <Card elevation={2} sx={{ 
                      borderRadius: '12px', 
                      height: '100%',
                      transition: 'transform 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-4px)',
                      },
                    }}>
                      <CardContent>
                        <Typography variant="h6" color="primary.main" gutterBottom fontWeight={600}>
                          User Information
                        </Typography>
                        <Divider sx={{ mb: 2 }} />
                        <Typography variant="body1" sx={{ mb: 1 }}>
                          <strong>Username:</strong> {user.username}
                        </Typography>
                        <Typography variant="body1">
                          <strong>Email:</strong> {user.email}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Card elevation={2} sx={{ 
                      borderRadius: '12px', 
                      height: '100%',
                      transition: 'transform 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-4px)',
                      },
                    }}>
                      <CardContent>
                        <Typography variant="h6" color="primary.main" gutterBottom fontWeight={600}>
                          Swipe Activity
                        </Typography>
                        <Divider sx={{ mb: 2 }} />
                        <Typography variant="body1" sx={{ mb: 1 }}>
                          <strong>Likes:</strong> {userStats.totalLikes}
                        </Typography>
                        <Typography variant="body1">
                          <strong>Dislikes:</strong> {userStats.totalDislikes}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              )}

              <Box sx={{ mt: 4 }}>
                <Typography variant="h5" color="primary.dark" gutterBottom fontWeight={600}>
                  Personalized Recommendations
                </Typography>
                <Button 
                  variant="contained" 
                  onClick={fetchRecommendations} 
                  sx={{ 
                    mb: 3,
                    borderRadius: '12px',
                    padding: '10px 24px',
                    textTransform: 'none',
                    fontWeight: 600,
                    background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                    boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
                    transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 6px 10px 4px rgba(33, 203, 243, .4)',
                    },
                  }}
                >
                  Load Recommendations
                </Button>
                
                {loading && <CircularProgress sx={{ display: 'block', mx: 'auto', my: 4 }} />}
                {error && <Alert severity="error" sx={{ mb: 3, borderRadius: '8px' }}>{error}</Alert>}
                
                {!loading && recs.length > 0 && (
                  <Paper elevation={2} sx={{ borderRadius: '12px', overflow: 'hidden' }}>
                    <List sx={{ p: 0 }}>
                      {recs.map((rec, idx) => (
                        <React.Fragment key={idx}>
                          <ListItem 
                            alignItems="flex-start" 
                            sx={{ 
                              py: 2.5,
                              transition: 'background-color 0.2s ease',
                              '&:hover': {
                                backgroundColor: 'rgba(0, 0, 0, 0.03)',
                              }
                            }}
                          >
                            <ListItemAvatar sx={{ mr: 2 }}>
                              <Avatar 
                                alt={rec.restaurant.name} 
                                src={rec.restaurant.photo_url || '/foodtest.jpeg'} 
                                variant="rounded"
                                sx={{ 
                                  width: 80, 
                                  height: 80,
                                  borderRadius: '8px',
                                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
                                }}
                              />
                            </ListItemAvatar>
                            <ListItemText
                              primary={
                                <Typography variant="h6" fontWeight={600}>
                                  {rec.restaurant.name}
                                </Typography>
                              }
                              secondary={
                                <Box sx={{ mt: 0.5 }}>
                                  <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                                    {rec.restaurant.address}
                                  </Typography>
                                  <Rating 
                                    value={rec.restaurant.stars || 0} 
                                    readOnly 
                                    precision={0.5} 
                                    emptyIcon={<StarIcon fontSize="inherit" style={{ opacity: 0.55 }} />}
                                    sx={{ color: '#ffb400' }} 
                                  />
                                </Box>
                              }
                            />
                          </ListItem>
                          {idx < recs.length - 1 && <Divider variant="inset" component="li" />}
                        </React.Fragment>
                      ))}
                    </List>
                  </Paper>
                )}
              </Box>
            </CardContent>
          </Paper>
        </Fade>
      </Container>
    </Box>
  );
}
