import { useState, useEffect } from "react";
import TinderCard from "react-tinder-card";
import { Button } from "@mui/material";

export default function SwipeImageUI() {
  const [images, setImages] = useState([]);
  const [currIndex, setCurrIndex] = useState(0);
  const [likedIds, setLikedIds] = useState([]);
  const [dislikedIds, setDislikedIds] = useState([]);
  const [imageError, setImageError] = useState(false);
  
  const retrieveRandomImages = async () => {
    const images = [];

    await fetch("http://localhost:8000/photos/random", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((response) => response.json())
      .then((data) => {
        for (let j = 0; j < data.length; j++) {
          images.push({
            id: j,
            url: data[j].photo_id + ".jpg",
          });
        }
      });
    console.log(images);
    return images;
  };
  
  const handleSwipe = (direction) => {
    console.log(`Swiped ${direction} on image ${currIndex}`);
    setImageError(false); // Reset error state for the next image
    setCurrIndex((prevIndex) => (prevIndex + 1) % images.length);
    if (direction === "right") {
      setLikedIds((prev) => [...prev, images[currIndex].id]);
    } else if (direction === "left") {
      setDislikedIds((prev) => [...prev, images[currIndex].id]);
    }
  };
  
  useEffect(() => {
    retrieveRandomImages().then((data) => {
      setImages(data);
    });
  }, []);
  
  // Reset image error state whenever current index changes
  useEffect(() => {
    setImageError(false);
  }, [currIndex]);
  
  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <h1 className="text-3xl font-bold mb-6">Welcome to Binge</h1>
      <div className="relative w-96 h-96 flex items-center justify-center">
        <TinderCard
          key={currIndex}
          onSwipe={(dir) => handleSwipe(dir)}
          preventSwipe={["up", "down"]}
          className="absolute w-full h-full"
        >
          <div
            style={{
              backgroundImage: !imageError ? `url(/photos/${images[currIndex]?.url})` : 'none',
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
            
            {/* Hidden image to detect loading errors */}
            {images[currIndex]?.url && (
              <img
                src={`/photos/${images[currIndex]?.url}`}
                alt=""
                style={{ display: 'none' }}
                onError={() => setImageError(true)}
              />
            )}
          </div>
        </TinderCard>
      </div>
      <Button className="mt-6 px-6 py-2 bg-blue-500 text-white rounded-lg shadow-md hover:bg-blue-600 transition-all">
        Recommendations
      </Button>
    </div>
  );
}