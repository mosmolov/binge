import { useState } from "react";
import TinderCard from "react-tinder-card";
import {Button} from "@mui/material"

const images = [
  "/vite.svg",
  "/foodtest.jpeg",
];

export default function SwipeImageUI() {
  const [index, setIndex] = useState(0);
  
  const getRandomIndex = () => {
    let newIndex;
    do {
      newIndex = Math.floor(Math.random() * images.length);
    } while (newIndex === index);
    return newIndex;
  };

  const handleSwipe = (direction) => {
    console.log(`Swiped ${direction} on image ${index}`);
    setIndex(getRandomIndex());
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen">
    <h1 className="text-3xl font-bold mb-6">Welcome to Binge</h1>
      <div className="relative w-96 h-96 flex items-center justify-center">
        <TinderCard
          key={images[index]}
          onSwipe={(dir) => handleSwipe(dir)}
          preventSwipe={['up', 'down']}
          className="absolute w-full h-full"
        >
          <div
  style={{
    backgroundImage: `url(${images[index]})`,
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
