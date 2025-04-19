import { useState, useEffect } from "react";
import TinderCard from "react-tinder-card";
import { Button, AppBar, Toolbar, Typography, IconButton, Box, Fade } from "@mui/material";
import { keyframes } from '@emotion/react';
import ThumbUpAltIcon from '@mui/icons-material/ThumbUpAlt';
import ThumbDownAltIcon from '@mui/icons-material/ThumbDownAlt';
import RestaurantMenuIcon from '@mui/icons-material/RestaurantMenu';
import { useAuth } from './context/AuthContext';
import { useNavigate } from 'react-router-dom';

// Gradient and fade animations
const fullScreenGradient = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

export default function SwipeImageUI() {
  const [images, setImages] = useState([]);
  const [currIndex, setCurrIndex] = useState(0);
  const [likedIds, setLikedIds] = useState([]);
  const [dislikedIds, setDislikedIds] = useState([]);
  const [imageError, setImageError] = useState(false);
  const [userLatitude, setUserLatitude] = useState(37.7749); // Default to San Francisco
  const [userLongitude, setUserLongitude] = useState(-122.4194);
  const [locationStatus, setLocationStatus] = useState(null);

  const { logout, user } = useAuth();
  const navigate = useNavigate();

  // Retrieve images from the backend
  const retrieveRandomImages = async () => {
    const imagesArray = [];
    await fetch("http://localhost:8000/photos/random", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((response) => response.json())
      .then((data) => {
        for (let j = 0; j < data.length; j++) {
          imagesArray.push({
            id: j,
            url: data[j].photo_id + ".jpg",
            photo_id: data[j].photo_id,
          });
        }
      })
      .catch((error) => console.error("Error fetching images:", error));
    console.log(imagesArray);
    return imagesArray;
  };

  // Get user's current location
  const getCurrentLocation = () => {
    setLocationStatus("Fetching your location...");
    
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLatitude(position.coords.latitude);
          setUserLongitude(position.coords.longitude);
          setLocationStatus(`Location updated! (${position.coords.latitude.toFixed(4)}, ${position.coords.longitude.toFixed(4)})`);
          setTimeout(() => setLocationStatus(null), 3000);
        },
        (error) => {
          console.error("Error getting location:", error);
          setLocationStatus("Unable to get location. Please check permissions.");
        }
      );
    } else {
      setLocationStatus("Geolocation is not supported by your browser.");
    }
  };

  // Send the recommendations request
  const fetchRecommendations = async () => {
    const requestBody = {
      liked_ids: likedIds,
      disliked_ids: dislikedIds,
      user_latitude: parseFloat(userLatitude),
      user_longitude: parseFloat(userLongitude),
      radius_miles: 10.0,
      top_n: 5,
    };

    try {
      const response = await fetch("http://localhost:8000/recommendations/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      if (!response.ok) {
        throw new Error("Failed to fetch recommendations");
      }
      const data = await response.json();
      console.log("Recommendations data:", data);
      // Instead of setting state, navigate to recommendations page
      navigate('/recommendations', { state: { recommendations: data.recommendations } });
    } catch (error) {
      console.error("Error fetching recommendations:", error);
    }
  };
  
  // Handle swipe event (also used by the extra buttons)
  const handleSwipe = async (direction) => {
    if (!user || !user.id) {
      console.error("User not logged in, cannot record swipe.");
      return; 
    }
    
    if (currIndex >= images.length) {
        console.log("No more images to swipe.");
        return;
    }

    console.log(`Swiped ${direction} on image ${currIndex}`);
    setImageError(false); // Reset error state for the next image

    const currentImage = images[currIndex];
    if (!currentImage || !currentImage.photo_id) {
        console.error("Current image data is missing.");
        return;
    }

    try {
        // Fetch the business ID for the current image
        const businessIdResponse = await fetch(`http://localhost:8000/photos/business/${currentImage.photo_id.toString()}`);
        if (!businessIdResponse.ok) {
            throw new Error(`Failed to fetch business ID: ${businessIdResponse.statusText}`);
        }
        const businessId = await businessIdResponse.json();

        if (!businessId) {
            throw new Error("Business ID not found for the photo.");
        }

        // Determine the API endpoint based on swipe direction
        const endpoint = direction === "right" ? "likes" : "dislikes";
        const apiUrl = `http://localhost:8000/users/${user.id}/${endpoint}?business_id=${encodeURIComponent(businessId)}`;

        // Persist the swipe to the backend
        const updateResponse = await fetch(apiUrl, {
            method: "PATCH",
            headers: {
                "Content-Type": "application/json",
            },
        });

        if (!updateResponse.ok) {
            const errorData = await updateResponse.json();
            throw new Error(`Failed to update user ${endpoint}: ${errorData.detail || updateResponse.statusText}`);
        }

        console.log(`Successfully updated user ${endpoint} for business ${businessId}`);

        // Update local state
        if (direction === "right") {
            setLikedIds((prev) => [...prev, businessId]);
        } else if (direction === "left") {
            setDislikedIds((prev) => [...prev, businessId]);
        }

        // Move to the next card or fetch recommendations
        if (currIndex === images.length - 1) {
            fetchRecommendations();
        } else {
            setCurrIndex((prevIndex) => prevIndex + 1);
        }

    } catch (error) {
        console.error("Error handling swipe:", error);
    }
  };

  useEffect(() => {
    retrieveRandomImages().then((data) => {
      setImages(data);
    });
  }, []);

  // Reset image error state whenever the current index changes
  useEffect(() => {
    setImageError(false);
  }, [currIndex]);

  return (
    <Box
      sx={{
        minHeight: '100vh',
        width: '100vw',
        background: 'linear-gradient(120deg, #f8b500, #fceabb, #f8b500, #f76d1a)',
        backgroundSize: '400% 400%',
        animation: `${fullScreenGradient} 15s ease infinite`,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <AppBar position="static" color="transparent" elevation={0} sx={{ background: 'rgba(255,255,255,0.85)' }}>
        <Toolbar>
          <RestaurantMenuIcon sx={{ mr: 1, color: '#f76d1a' }} />
          <Typography variant="h5" sx={{ flexGrow: 1, fontWeight: 700, color: '#333' }}>Binge</Typography>
          <IconButton edge="end" color="inherit" onClick={() => { logout && logout(); navigate('/login'); }}>
          </IconButton>
        </Toolbar>
      </AppBar>
      <Fade in={true} timeout={800}>
        <Box sx={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
          <Box sx={{ width: { xs: '95vw', sm: 420, md: 500 }, maxWidth: 500, height: { xs: 500, sm: 600 }, position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            <Box sx={{ mb: 2, textAlign: 'center' }}>
              <Button
                variant="outlined"
                onClick={getCurrentLocation}
                startIcon={<span role="img" aria-label="location">📍</span>}
                size="small"
                sx={{ borderRadius: 3, fontWeight: 600 }}
              >
                Use My Location
              </Button>
              {locationStatus && (
                <Typography variant="body2" color="#1976d2" sx={{ mt: 1 }}>{locationStatus}</Typography>
              )}
            </Box>
            <Box sx={{ position: 'relative', width: '100%', height: { xs: 400, sm: 500 }, mb: 2 }}>
              {images.length > 0 && currIndex < images.length && (
                <TinderCard
                  key={currIndex}
                  onSwipe={(dir) => handleSwipe(dir)}
                  preventSwipe={["up", "down"]}
                  className="absolute w-full h-full"
                >
                  <Box
                    sx={{
                      backgroundImage: !imageError ? `url(/photos/${images[currIndex]?.url})` : "none",
                      backgroundColor: imageError ? "#f0f0f0" : "rgba(0,0,0,0.2)",
                      backgroundSize: "cover",
                      backgroundPosition: "center",
                      borderRadius: 6,
                      boxShadow: 8,
                      width: '100%',
                      height: { xs: 400, sm: 500 },
                      display: 'flex',
                      alignItems: 'flex-end',
                      justifyContent: 'center',
                      color: imageError ? "#333" : "#fff",
                      position: 'relative',
                      overflow: 'hidden',
                    }}
                  >
                    {/* Overlay info bar */}
                    {!imageError && (
                      <Box sx={{
                        width: '100%',
                        background: 'rgba(0,0,0,0.45)',
                        p: 2,
                        borderBottomLeftRadius: 24,
                        borderBottomRightRadius: 24,
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                      }}>
                        <Typography variant="h6" fontWeight={700} sx={{ color: '#fff' }}>
                          Food #{images[currIndex]?.photo_id}
                        </Typography>
                        {/* Optionally add more info here */}
                      </Box>
                    )}
                    {imageError && (
                      <Box sx={{ textAlign: 'center', p: 4 }}>
                        <Typography variant="h6">Image could not be loaded</Typography>
                        <Typography variant="body2" sx={{ mt: 2 }}>Please try swiping to the next image</Typography>
                      </Box>
                    )}
                    {/* Hidden image element to catch loading errors */}
                    {images[currIndex]?.url && (
                      <img
                        src={`/photos/${images[currIndex]?.url}`}
                        alt="food"
                        style={{ display: "none" }}
                        onError={() => setImageError(true)}
                      />
                    )}
                  </Box>
                </TinderCard>
              )}
            </Box>
            <Box sx={{ display: 'flex', gap: 3, width: '100%' }}>
              <Button
                variant="contained"
                color="error"
                fullWidth
                onClick={() => handleSwipe('left')}
                sx={{ borderRadius: 3, fontWeight: 700, fontSize: 18, py: 1.5, background: 'linear-gradient(90deg,#ff5858,#f09819)' }}
                startIcon={<ThumbDownAltIcon />}
              >
                Dislike
              </Button>
              <Button
                variant="contained"
                color="success"
                fullWidth
                onClick={() => handleSwipe('right')}
                sx={{ borderRadius: 3, fontWeight: 700, fontSize: 18, py: 1.5, background: 'linear-gradient(90deg,#43cea2,#185a9d)' }}
                startIcon={<ThumbUpAltIcon />}
              >
                Like
              </Button>
            </Box>
          </Box>
        </Box>
      </Fade>
    </Box>
  );
}
