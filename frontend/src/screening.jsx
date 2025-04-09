import { useState, useEffect } from "react";
import TinderCard from "react-tinder-card";
import { Button } from "@mui/material";

export default function SwipeImageUI() {
  const [images, setImages] = useState([]);
  const [currIndex, setCurrIndex] = useState(0);
  const [likedIds, setLikedIds] = useState([]);
  const [dislikedIds, setDislikedIds] = useState([]);
  const [imageError, setImageError] = useState(false);
  const [recommendations, setRecommendations] = useState(null);
  const [userLatitude, setUserLatitude] = useState(37.7749); // Default to San Francisco
  const [userLongitude, setUserLongitude] = useState(-122.4194);
  const [locationStatus, setLocationStatus] = useState(null);

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
      setRecommendations(data.recommendations); // Access the recommendations array
    } catch (error) {
      console.error("Error fetching recommendations:", error);
    }
  };

  // Handle swipe event (also used by the extra buttons)
  const handleSwipe = async (direction) => {
    console.log(`Swiped ${direction} on image ${currIndex}`);
    setImageError(false); // Reset error state for the next image

    // Fetch the business ID for the current image
    const businessId = await fetch(`http://localhost:8000/photos/business/${images[currIndex].photo_id.toString()}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to fetch business ID");
        }
        return response.json();
      })
      .then((data) => data)
      .catch((error) => console.error("Error fetching business ID:", error));

    // Register the swipe action
    if (direction === "right") {
      setLikedIds((prev) => [...prev, businessId]);
    } else if (direction === "left") {
      setDislikedIds((prev) => [...prev, businessId]);
    }

    // If swiping on the last image, fetch recommendations automatically
    if (currIndex === images.length - 1) {
      fetchRecommendations();
    } else {
      setCurrIndex((prevIndex) => prevIndex + 1);
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

  // If recommendations data is available, display the recommendations view automatically
  if (recommendations) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen p-4">
        <h1 className="text-3xl font-bold mb-6">Restaurant Recommendations</h1>
        {recommendations.map((rec, index) => (
          <div key={index} className="w-full max-w-md p-4 mb-4 border rounded-lg shadow-md">
            <h2 className="text-xl font-bold mb-2">{rec.restaurant.name}</h2>
            <p>Stars: {rec.restaurant.stars}</p>
            <p className="mt-2 text-sm text-gray-600">Location: {rec.restaurant.address}</p>
          </div>
        ))}
        <Button
          className="mt-6 px-6 py-2 bg-blue-500 text-white rounded-lg shadow-md hover:bg-blue-600 transition-all"
          onClick={() => {
            // Reset state to start a new session
            setRecommendations(null);
            setCurrIndex(0);
            setLikedIds([]);
            setDislikedIds([]);
            retrieveRandomImages().then((data) => {
              setImages(data);
            });
          }}
        >
          Start New Session
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <h1 className="text-3xl font-bold mb-6">Welcome to Binge</h1>
      <div className="mb-4 flex flex-col items-center">
        <Button 
          variant="outlined"
          onClick={getCurrentLocation}
          startIcon={<span role="img" aria-label="location">📍</span>}
          size="small"
        >
          Use My Location
        </Button>
        {locationStatus && (
          <p className="text-sm mt-1 text-blue-600">{locationStatus}</p>
        )}
      </div>
      <div className="relative w-96 h-96 flex items-center justify-center">
        {images.length > 0 && currIndex < images.length && (
          <TinderCard
            key={currIndex}
            onSwipe={(dir) => handleSwipe(dir)}
            preventSwipe={["up", "down"]}
            className="absolute w-full h-full"
          >
            <div
              style={{
                backgroundImage: !imageError ? `url(/photos/${images[currIndex]?.url})` : "none",
                backgroundColor: imageError ? "#f0f0f0" : "transparent",
                backgroundSize: "cover",
                backgroundPosition: "center",
                height: "500px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: imageError ? "#333" : "white",
                fontSize: "24px",
                fontWeight: "bold",
              }}
            >
              {imageError && (
                <div className="text-center p-4">
                  <p>Image could not be loaded</p>
                  <p className="text-sm mt-2">Please try swiping to the next image</p>
                </div>
              )}
              {/* Hidden image element to catch loading errors */}
              {images[currIndex]?.url && (
                <img
                  src={`/photos/${images[currIndex]?.url}`}
                  alt=""
                  style={{ display: "none" }}
                  onError={() => setImageError(true)}
                />
              )}
            </div>
          </TinderCard>
        )}
      </div>
      {/* Extra buttons for manual swipe registration */}
      <div className="flex gap-4 mt-4">
        <Button
          variant="contained"
          color="error"
          onClick={() => handleSwipe("left")}
        >
          Dislike
        </Button>
        <Button
          variant="contained"
          color="success"
          onClick={() => handleSwipe("right")}
        >
          Like
        </Button>
      </div>
    </div>
  );
}
