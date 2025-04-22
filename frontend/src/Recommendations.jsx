import React, { useState, useEffect } from 'react';
import {
  Container,
  CardHeader,
  Button,
  Avatar,
  Typography,
  Box,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Divider,
  Paper,
  Fade,
  Rating,
  IconButton,
  CircularProgress
} from '@mui/material';
import { keyframes } from '@emotion/react';
import { useLocation, useNavigate } from 'react-router-dom';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import StarIcon from '@mui/icons-material/Star';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';
import { useAuth } from './context/AuthContext';

// Consistent gradient animation
const gradientAnimation = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

// Fade-in animation for list items
const itemFadeIn = keyframes`
  from { opacity: 0; transform: translateX(-10px); }
  to { opacity: 1; transform: translateX(0); }
`;

// Pulse animation for load more button
const pulseAnimation = keyframes`
  0% { transform: scale(1); box-shadow: 0 3px 5px 2px rgba(33, 150, 243, 0.3); }
  50% { transform: scale(1.03); box-shadow: 0 5px 15px 4px rgba(33, 150, 243, 0.4); }
  100% { transform: scale(1); box-shadow: 0 3px 5px 2px rgba(33, 150, 243, 0.3); }
`;

export default function Recommendations() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();
  const initialRecommendations = location.state?.recommendations || [];
  const [recommendations, setRecommendations] = useState(initialRecommendations);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [userLocation, setUserLocation] = useState(null);
  const fallbackImageUrl = '/foodtest.jpeg';

  // Get user location on component mount
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation([position.coords.latitude, position.coords.longitude]);
        },
        (error) => {
          console.error("Error getting location:", error);
        }
      );
    }
  }, []);

  const loadMoreRecommendations = async () => {
    if (!user || !user.id || !userLocation || isLoadingMore) return;
    
    setIsLoadingMore(true);
    
    try {
      // Fetch user profile to get likes/dislikes
      const userProfileResponse = await fetch(`http://localhost:8000/users/${user.id}`);
      if (!userProfileResponse.ok) {
        throw new Error(`Failed to fetch user profile: ${userProfileResponse.status}`);
      }
      
      const userProfile = await userProfileResponse.json();
      const liked_ids = userProfile.liked_business_ids || [];
      const disliked_ids = userProfile.disliked_business_ids || [];
      
      // Define request body for more recommendations
      const requestBody = {
        liked_ids: liked_ids,
        disliked_ids: disliked_ids,
        user_location: userLocation,
        radius_miles: 25, // Expanded radius for more options
        top_n: 5 * (page + 1), // Increase the number of recommendations each time
        offset: recommendations.length // Skip already loaded recommendations
      };
      
      const response = await fetch("http://localhost:8000/recommendations/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // If we got fewer new recommendations than expected, we've reached the end
      if (data.recommendations.length === 0) {
        setHasMore(false);
      } else {
        // Filter out any duplicates based on business_id
        const existingIds = new Set(recommendations.map(rec => rec.restaurant.business_id));
        const newRecs = data.recommendations.filter(rec => !existingIds.has(rec.restaurant.business_id));
        
        setRecommendations(prevRecs => [...prevRecs, ...newRecs]);
        setPage(page + 1);
      }
    } catch (error) {
      console.error("Error loading more recommendations:", error);
    } finally {
      setIsLoadingMore(false);
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
        alignItems: 'flex-start',
        py: { xs: 4, sm: 6 },
        background:
          'linear-gradient(-45deg, #c1dfc4 0%, #deecdd 50%, #c1dfc4 100%)',
        backgroundSize: '400% 400%',
        animation: `${gradientAnimation} 20s ease infinite`,
        overflowY: 'auto',
      }}
    >
      <Fade in={true} timeout={800}>
        <Paper
          elevation={6}
          sx={{
            position: 'relative', // enable absolute positioning for profile button
            width: { xs: '95%', sm: '80%', md: 650 },
            maxWidth: 650,
            padding: { xs: 2, sm: 4 },
            borderRadius: '16px',
            backgroundColor: 'rgba(255, 255, 255, 0.88)',
            backdropFilter: 'blur(10px)',
            boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.25)',
            border: '1px solid rgba(255, 255, 255, 0.18)',
          }}
        >
          <IconButton
            onClick={() => navigate('/profile')}
            aria-label="profile"
            sx={{ position: 'absolute', top: 8, right: 8, color: 'primary.main' }}
          >
            <AccountCircleIcon />
          </IconButton>
          <CardHeader
            title="Your Top Recommendations"
            titleTypographyProps={{
              variant: 'h4',
              fontWeight: 'bold',
              color: 'primary.dark',
              textAlign: 'center',
            }}
            sx={{ pb: 2 }}
          />
          <Box sx={{ p: 0 }}>
            {recommendations.length === 0 ? (
              <Typography variant="body1" color="text.secondary" align="center" sx={{ py: 4 }}>
                No recommendations found based on your swipes.
              </Typography>
            ) : (
              <List disablePadding>
                {recommendations.map((rec, index) => (
                  <React.Fragment key={rec.restaurant.business_id || index}>
                    <ListItem
                      sx={{
                        py: 2.5,
                        px: { xs: 1, sm: 2 },
                        opacity: 0,
                        animation: `${itemFadeIn} 0.5s ease-out ${Math.min(index, 10) * 0.1}s forwards`,
                        display: 'flex',
                        alignItems: 'flex-start',
                      }}
                    >
                      <ListItemAvatar sx={{ mr: 2.5, mt: 0.5 }}>
                        <Avatar
                          variant="rounded"
                          src={rec.restaurant.photo_url || fallbackImageUrl}
                          alt={rec.restaurant.name}
                          sx={{
                            width: { xs: 70, sm: 90 },
                            height: { xs: 70, sm: 90 },
                            borderRadius: '12px',
                            boxShadow: 3,
                          }}
                          imgProps={{ loading: 'lazy' }}
                          onError={(e) => { e.target.src = fallbackImageUrl; }}
                        />
                      </ListItemAvatar>
                      <ListItemText
                        primary={
                          <Typography variant="h6" fontWeight={600}>
                            {rec.restaurant.name}
                          </Typography>
                        }
                        secondary={
                          <Box component="span" sx={{ display: 'flex', flexDirection: 'column', mt: 0.5 }}>
                            <Typography component="span" variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                              {rec.restaurant.address}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Rating
                                name={`rating-${index}`}
                                value={rec.restaurant.stars || 0}
                                readOnly
                                precision={0.5}
                                size="small"
                                emptyIcon={<StarIcon style={{ opacity: 0.55 }} fontSize="inherit" />}
                                sx={{ color: '#ffb400', mr: 1 }}
                              />
                              <Typography variant="body2" color="text.secondary">
                                ({rec.restaurant.stars?.toFixed(1) || 'N/A'})
                              </Typography>
                            </Box>
                            {rec.restaurant.categories && (
                              <Typography 
                                variant="body2" 
                                color="text.secondary" 
                                sx={{ 
                                  mt: 0.5, 
                                  fontStyle: 'italic',
                                  maxWidth: '100%',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap'
                                }}
                              >
                                {rec.restaurant.categories}
                              </Typography>
                            )}
                          </Box>
                        }
                        primaryTypographyProps={{ variant: 'h6', fontWeight: 600, mb: 0.5 }}
                      />
                    </ListItem>
                    {index < recommendations.length - 1 && <Divider variant="inset" component="li" />}
                  </React.Fragment>
                ))}
              </List>
            )}
            
            {hasMore && recommendations.length > 0 && (
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3, mb: 2 }}>
                <Button
                  variant="outlined"
                  color="primary"
                  onClick={loadMoreRecommendations}
                  disabled={isLoadingMore}
                  startIcon={isLoadingMore ? <CircularProgress size={20} color="inherit" /> : <MoreHorizIcon />}
                  sx={{
                    borderRadius: '12px',
                    padding: '10px 16px',
                    textTransform: 'none',
                    fontWeight: 600,
                    animation: hasMore ? `${pulseAnimation} 2s infinite` : 'none',
                  }}
                >
                  {isLoadingMore ? 'Loading...' : 'Load More Recommendations'}
                </Button>
              </Box>
            )}
            
            <Button
              fullWidth
              variant="contained"
              startIcon={<ArrowBackIcon />}
              sx={{
                mt: 4,
                padding: '12px',
                borderRadius: '12px',
                fontWeight: 'bold',
                fontSize: '1rem',
                textTransform: 'none',
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
                transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: '0 6px 10px 4px rgba(33, 203, 243, .4)',
                },
              }}
              onClick={() => navigate('/')}
            >
              Swipe Again
            </Button>
          </Box>
        </Paper>
      </Fade>
    </Container>
  );
}
