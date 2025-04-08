import { useState, useEffect } from "react";
import TinderCard from "react-tinder-card";
import {Button} from "@mui/material"

const images_dir = "../assets/images/";
export default function SwipeImageUI() {
  const [images, setImages] = useState([]);
  const [currIndex, setCurrIndex] = useState(0);
  
  const retrieveRandomImages = async (num_images=10) => {
    const images = [];
    for (let i = 1; i <= num_images; i++) {
      //pick random files from directory
      // import.meta.env.VITE_API_URL + "/restaurants/random_ids"
      const random_ids = await fetch("http://localhost:8000/restaurants/random_ids", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }).then((response) => response.json())
      .then((data) => {
        console.log(data);
        return data;
      }
      );
      // add images to array
      for (let j = 0; j < random_ids.length; j++) {
        images.push({
          id: j,
          url: `${images_dir}${random_ids[j]}.jpg`,
        });
      }
    }
    return images;
  }
  const handleSwipe = (direction) => {
    console.log(`Swiped ${direction} on image ${currIndex}`);
    setCurrIndex((prevIndex) => (prevIndex + 1) % images.length);
  };
  useEffect (() => {
    retrieveRandomImages(10).then((data) => {
      setImages(data);
    });
  }, []);
  return (
    <div className="flex flex-col items-center justify-center h-screen">
    <h1 className="text-3xl font-bold mb-6">Welcome to Binge</h1>
      <div className="relative w-96 h-96 flex items-center justify-center">
        <TinderCard
          key={currIndex}
          onSwipe={(dir) => handleSwipe(dir)}
          preventSwipe={['up', 'down']}
          className="absolute w-full h-full"
        >
          <div
  style={{
    backgroundImage: `url(${images[currIndex]?.url})`,
    backgroundSize: "cover",
    backgroundPosition: "center",
    height: "500px", // Adjust height
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "white", // Ensure text is visible
    fontSize: "24px",
    fontWeight: "bold",
  }}
>
  <h3>Test</h3>
</div>
        </TinderCard>
      </div>
      <Button className="mt-6 px-6 py-2 bg-blue-500 text-white rounded-lg shadow-md hover:bg-blue-600 transition-all">
        Recommendations
      </Button>
    </div>
  );
}
