import { useState, useEffect, useMemo, createRef } from "react"; // Remove unused React and useRef
import TinderCard from "react-tinder-card";
import {
  Button,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Fade,
  CircularProgress,
  Alert,
  Chip,
  Paper,
  Container,
  Stack,
  Slider,
} from "@mui/material";
import { keyframes } from "@emotion/react";
import ThumbUpAltIcon from "@mui/icons-material/ThumbUpAlt";
import ThumbDownAltIcon from "@mui/icons-material/ThumbDownAlt";
import RestaurantMenuIcon from "@mui/icons-material/RestaurantMenu";
import LogoutIcon from "@mui/icons-material/Logout";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import { useAuth } from "./context/AuthContext";
import { useNavigate } from "react-router-dom";

// Consistent gradient animation
const fullScreenGradient = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

export default function SwipeImageUI() {
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false); // Start as false, loading triggered by actions
  const [error, setError] = useState(null);
  const [imageLoadError, setImageLoadError] = useState(false);
  const [userLatitude, setUserLatitude] = useState(null);
  const [userLongitude, setUserLongitude] = useState(null);
  const [locationStatus, setLocationStatus] = useState("Click button to get location");
  const [isFetchingLocation, setIsFetchingLocation] = useState(false);
  const [locationAcquired, setLocationAcquired] = useState(false); // Track if location is acquired
  const [radius, setRadius] = useState(10); // Recommendation radius in miles

  const { logout, user } = useAuth();
  const navigate = useNavigate();

  // Refs for controlling TinderCard swipes programmatically
  const childRefs = useMemo(() =>
    Array(images.length)
      .fill(0)
      .map(() => createRef()), // Remove unused index `_`
    [images.length]
  );

  // Fetch images only after location is acquired
  useEffect(() => {
    const retrieveRandomImages = async () => {
      if (!locationAcquired) return; // Don't fetch if location isn't ready

      setLoading(true);
      setError(null);
      setImages([]); // Clear previous images if any
      try {
        const response = await fetch("http://localhost:8000/photos/random", {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.length === 0) {
          // Handle case where no images are returned
          setError("No images found for swiping.");
          setImages([]);
          setCurrentIndex(-1); // No cards to swipe
        } else {
          const formattedImages = data.map((img, index) => ({
            id: index,
            url: `/photos/${img.photo_id}.jpg`,
            photo_id: img.photo_id,
          }));
          setImages(formattedImages);
          setCurrentIndex(0);
        }
      } catch (e) {
        console.error("Error fetching images:", e);
        setError("Failed to load images. Please try refreshing.");
      } finally {
        setLoading(false);
      }
    };
    retrieveRandomImages();
  }, [locationAcquired]); // Depend on locationAcquired

  // Get user's current location
  const getCurrentLocation = () => {
    if (!navigator.geolocation) {
      setLocationStatus("Geolocation is not supported.");
      return;
    }
    setIsFetchingLocation(true);
    setLocationStatus("Fetching location...");
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setUserLatitude(position.coords.latitude);
        setUserLongitude(position.coords.longitude);
        setLocationStatus(`Location Acquired`);
        setLocationAcquired(true); // Set location acquired flag
        setIsFetchingLocation(false);
        setTimeout(() => setLocationStatus(null), 3000);
      },
      (err) => {
        console.error("Error getting location:", err);
        setLocationStatus("Location access denied.");
        setLocationAcquired(false); // Ensure flag is false on error
        setIsFetchingLocation(false);
      }
    );
  };

  // Send swipe data (like/dislike) to the backend for the specific photo
  const handleSwipeAction = async (direction, photoId) => {
    if (!user || !user.id) {
      console.error("User not logged in.");
      setError("Authentication error. Please log in again.");
      return; // Stop if user is not logged in
    }

    try {
      // Fetch the business ID for the swiped image
      const businessIdResponse = await fetch(`http://localhost:8000/photos/business/${photoId}`);
      if (!businessIdResponse.ok) throw new Error('Failed to get business ID');
      const businessId = await businessIdResponse.json();
      if (!businessId) throw new Error('Business ID not found');

      // Update user likes/dislikes
      const endpoint = direction === "right" ? "likes" : "dislikes";
      const updateUrl = `http://localhost:8000/users/${user.id}/${endpoint}?business_id=${encodeURIComponent(businessId)}`;
      const updateResponse = await fetch(updateUrl, { method: "PATCH" });
      if (!updateResponse.ok) throw new Error(`Failed to update ${endpoint}`);

      console.log(`User ${user.id} ${direction === 'right' ? 'liked' : 'disliked'} business ${businessId}`);
    } catch (e) {
      console.error("Error during swipe action:", e);
      setError(`An error occurred while recording your preference: ${e.message}`);
    }
  };

  // Fetch recommendations after all swipes are done
  const fetchRecommendations = async () => {
    console.log("Fetching recommendations...");
    setLoading(true); // Show loading indicator for recommendation fetch
    setError(null);

    if (!user || !user.id || !userLatitude || !userLongitude) {
      setError("Cannot fetch recommendations: Missing user data or location.");
      setLoading(false);
      return;
    }

    try {
      // Fetch user's profile data (likes/dislikes)
      const userProfileResponse = await fetch(`http://localhost:8000/users/${user.id}`);
      if (!userProfileResponse.ok) {
        const errorText = await userProfileResponse.text();
        console.error("Failed to fetch user profile:", errorText);
        throw new Error(`Failed to fetch user profile: ${userProfileResponse.status}`);
      }
      const userProfile = await userProfileResponse.json();

      const liked_ids = userProfile.liked_businesses || [];
      const disliked_ids = userProfile.disliked_businesses || [];

      // Prepare request body for recommendation endpoint
      const requestBody = {
        liked_ids: liked_ids,
        disliked_ids: disliked_ids,
        user_latitude: parseFloat(userLatitude),
        user_longitude: parseFloat(userLongitude),
        radius_miles: radius, // Use selected recommendation radius
        top_n: 5, // Example count
      };

      // Call recommendation endpoint
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
      navigate('/recommendations', { state: { recommendations: data.recommendations } });
    } catch (e) {
      console.error("Error fetching recommendations:", e);
      setError(`Failed to get recommendations: ${e.message}`);
      setLoading(false);
    }
  };

  // Called when a card leaves the screen
  const swiped = (direction, photoId, index) => {
    console.log(`Swiped ${direction} on photo ${photoId} at index ${index}`);
    setImageLoadError(false); // Reset error for the next card
    handleSwipeAction(direction, photoId); // Record the swipe action immediately

    setCurrentIndex(currentIndex + 1); // Update the index to the next card

    // Check if it was the last card *after* updating the index
    if (currentIndex === images.length - 1) {
      console.log("Last card swiped, fetching recommendations...");
      fetchRecommendations(); // Fetch recommendations now
    }
  };

  // Programmatic swipe trigger
  const triggerSwipe = async (direction) => {
    if (currentIndex >= 0 && childRefs[currentIndex].current) {
      await childRefs[currentIndex].current.swipe(direction); // Swipe the current card
    }
  };

  // Handle image loading error
  const handleImageError = () => {
    console.warn(`Failed to load image: ${images[currentIndex]?.url}`);
    setImageLoadError(true);
  };

  // Reset image error when index changes
  useEffect(() => {
    setImageLoadError(false);
  }, [currentIndex]);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        width: "100vw",
        background:
          "linear-gradient(-45deg, #ff9a9e, #fad0c4, #fad0c4, #ff9a9e)",
        backgroundSize: "400% 400%",
        animation: `${fullScreenGradient} 20s ease infinite`,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      <AppBar
        position="static"
        color="transparent"
        elevation={1}
        sx={{
          background: "rgba(255, 255, 255, 0.7)",
          backdropFilter: "blur(8px)",
        }}
      >
        <Toolbar>
          <RestaurantMenuIcon sx={{ mr: 1.5, color: "primary.main" }} />
          <Typography
            variant="h5"
            sx={{ flexGrow: 1, fontWeight: 700, color: "text.primary" }}
          >
            Binge
          </Typography>
          <IconButton
            edge="end"
            color="inherit"
            onClick={handleLogout}
            aria-label="logout"
            sx={{ color: "text.secondary" }}
          >
            <LogoutIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Container maxWidth="sm" sx={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', py: 3 }}>
        {!locationAcquired && (
          <Paper elevation={2} sx={{ p: 3, width: '100%', maxWidth: 400, borderRadius: '12px', textAlign: 'center', background: 'rgba(255, 255, 255, 0.85)', backdropFilter: 'blur(5px)' }}>
            <Typography variant="h6" gutterBottom>Share Your Location</Typography>
            <Typography variant="body1" sx={{ mb: 2, color: 'text.secondary' }}>
              We need your location to find nearby restaurants.
            </Typography>
            <Stack direction="column" spacing={1.5} justifyContent="center" alignItems="center">
              <Button
                variant="contained"
                onClick={getCurrentLocation}
                startIcon={<MyLocationIcon />}
                disabled={isFetchingLocation}
                sx={{ borderRadius: '8px', textTransform: 'none', px: 3, py: 1 }}
              >
                {isFetchingLocation ? "Fetching..." : "Use My Location"}
              </Button>
              {locationStatus && !isFetchingLocation && (
                <Chip
                  label={locationStatus}
                  size="small"
                  variant="outlined"
                  color={locationStatus === "Location Acquired" ? "success" : (locationStatus === "Location access denied." ? "error" : "info")}
                  sx={{ mt: 1 }}
                />
              )}
              {error && <Alert severity="error" sx={{ width: '100%', mt: 1 }}>{error}</Alert>}
            </Stack>
          </Paper>
        )}

        {locationAcquired && (
          <>
            {/* Slider for selecting recommendation radius */}
            <Box sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
              <Typography gutterBottom>Recommendation Radius: {radius} miles</Typography>
              <Slider
                value={radius}
                onChange={(e, val) => setRadius(val)}
                min={1}
                max={100}
                step={1}
                valueLabelDisplay="auto"
              />
            </Box>

            {loading && <CircularProgress sx={{ my: 4 }} />}

            {error && !loading && <Alert severity="error" sx={{ width: '100%', mb: 2 }}>{error}</Alert>}

            {!loading && !error && images.length > 0 && (
              <Box sx={{ width: '100%', maxWidth: 400, height: '100%', position: 'relative', mb: 3 }}>
                {images[currentIndex] && (
                  <TinderCard
                    ref={childRefs[currentIndex]}
                    key={images[currentIndex].photo_id}
                    onSwipe={(dir) => swiped(dir, images[currentIndex].photo_id, currentIndex)}
                    preventSwipe={["up", "down"]}
                    style={{ position: 'absolute', width: '100%', height: '100%' }}
                  >
                    <Fade in={true} timeout={500}>
                      <Paper
                        
                        sx={{
                          position: 'relative',
                          backgroundImage: !imageLoadError ? `url(${images[currentIndex].url})` : 'none',
                          backgroundColor: imageLoadError ? "grey.200" : "grey.400",
                          backgroundSize: "cover",
                          backgroundPosition: "center",
                          width: '100%',
                          height: '50vh',
                          borderRadius: '16px',
                          overflow: 'hidden',
                          display: 'flex',
                          flexDirection: 'column',
                          justifyContent: 'space-between',
                        }}
                      >
                        {imageLoadError && (
                          <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center', p: 2, background: 'rgba(0,0,0,0.1)' }}>
                            <ErrorOutlineIcon color="error" sx={{ fontSize: 40, mb: 1 }} />
                            <Typography variant="h6" color="text.secondary">Image Error</Typography>
                            <Typography variant="body2" color="text.secondary">Could not load image.</Typography>
                          </Box>
                        )}

                        <img
                          src={images[currentIndex].url}
                          alt=""
                          style={{ display: "none" }}
                          onError={handleImageError}
                        />
                      </Paper>
                    </Fade>
                  </TinderCard>
                )}
              </Box>
            )}

            {!loading && !error && images.length === 0 && currentIndex < 0 && (
              <Typography sx={{ textAlign: 'center', mt: 5, color: 'text.secondary' }}>
                No more images to show! Fetching recommendations...
              </Typography>
            )}

            {!loading && !error && images.length > 0 && currentIndex >= 0 && (
              <Stack direction="row" spacing={3} justifyContent="center" sx={{ width: '100%', maxWidth: 400 }}>
                <Button
                  variant="contained"
                  onClick={() => triggerSwipe('left')}
                  sx={{
                    borderRadius: '50%',
                    width: 70, height: 70,
                    minWidth: 0,
                    background: 'linear-gradient(135deg, #ff758c 0%, #ff7eb3 100%)',
                    boxShadow: '0 4px 12px rgba(255, 117, 140, 0.4)',
                    '&:hover': {
                      transform: 'scale(1.05)',
                      boxShadow: '0 6px 16px rgba(255, 117, 140, 0.5)',
                    },
                    transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                  }}
                  aria-label="Dislike"
                >
                  <ThumbDownAltIcon sx={{ fontSize: 30, color: 'white' }} />
                </Button>
                <Button
                  variant="contained"
                  onClick={() => triggerSwipe('right')}
                  sx={{
                    borderRadius: '50%',
                    width: 70, height: 70,
                    minWidth: 0,
                    background: 'linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)',
                    boxShadow: '0 4px 12px rgba(132, 250, 176, 0.4)',
                    '&:hover': {
                      transform: 'scale(1.05)',
                      boxShadow: '0 6px 16px rgba(132, 250, 176, 0.5)',
                    },
                    transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                  }}
                  aria-label="Like"
                >
                  <ThumbUpAltIcon sx={{ fontSize: 30, color: 'white' }} />
                </Button>
              </Stack>
            )}
          </>
        )}
      </Container>
    </Box>
  );
}
