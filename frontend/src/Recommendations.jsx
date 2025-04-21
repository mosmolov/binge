import React from 'react';
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
} from '@mui/material';
import { keyframes } from '@emotion/react';
import { useLocation, useNavigate } from 'react-router-dom';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import StarIcon from '@mui/icons-material/Star';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';

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

export default function Recommendations() {
  const location = useLocation();
  const navigate = useNavigate();
  const recommendations = location.state?.recommendations || [];
  const fallbackImageUrl = '/foodtest.jpeg';

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
                        animation: `${itemFadeIn} 0.5s ease-out ${index * 0.1}s forwards`,
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
                        primary={rec.restaurant.name}
                        secondary={
                          <Box component="span" sx={{ display: 'flex', flexDirection: 'column' }}>
                            <Typography component="span" variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                              {rec.restaurant.address}
                            </Typography>
                            <Rating
                              name={`rating-${index}`}
                              value={rec.restaurant.stars || 0}
                              readOnly
                              precision={0.5}
                              emptyIcon={<StarIcon style={{ opacity: 0.55 }} fontSize="inherit" />}
                              sx={{ color: '#ffb400' }}
                            />
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
