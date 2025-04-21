import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Rating, 
  MenuItem, 
  IconButton,
  Stack,
  FormControl,
  InputLabel,
  Select,
  Chip,
  OutlinedInput,
  FormHelperText,
  Divider,
  Alert,
  Snackbar,
  CircularProgress
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

export default function AddRestaurant({ onClose }) {
  const [name, setName] = useState('');
  const [address, setAddress] = useState('');
  const [rating, setRating] = useState(0);
  const [price, setPrice] = useState('');
  const [cuisine, setCuisine] = useState('');
  const [ambience, setAmbience] = useState('');
  const [goodFor, setGoodFor] = useState([]);
  const [attributeOptions, setAttributeOptions] = useState({
    Cuisines: [],
    Ambiences: [],
    GoodFor: []
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    // Fetch attribute categories for dropdowns
    const fetchAttributes = async () => {
      try {
        const res = await fetch('http://localhost:8000/recommendations/attributes');
        const data = await res.json();
        console.log(data);
        setAttributeOptions(data);
      } catch (err) {
        console.error('Failed to load attributes:', err);
        setError('Failed to load restaurant attributes. Please try again.');
      }
    };
    fetchAttributes();
  }, []);

  const handleGoodForChange = (event) => {
    const {
      target: { value },
    } = event;
    // On autofill we get a stringified value.
    setGoodFor(typeof value === 'string' ? value.split(',') : value);
  };

  // Geocode address to get latitude and longitude
  const geocodeAddress = async (address) => {
    try {
      // This uses the Nominatim OpenStreetMap API (free, no API key required)
      const response = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(address)}`);
      const data = await response.json();
      
      if (data && data.length > 0) {
        return {
          latitude: parseFloat(data[0].lat),
          longitude: parseFloat(data[0].lon)
        };
      } else {
        throw new Error('No location found for this address');
      }
    } catch (err) {
      console.error('Geocoding error:', err);
      throw err;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      // Get latitude and longitude from address
      const geoData = await geocodeAddress(address);
      
      // Prepare form data for API
      const formData = { 
        name, 
        address, 
        latitude: geoData.latitude,
        longitude: geoData.longitude,
        stars: rating, 
        price: price ? parseInt(price) : null, 
        attributes: {
          Cuisine: cuisine,
          Ambience: ambience,
          GoodFor: goodFor
        }
      };
      
      console.log('Submitting restaurant:', formData);
      
      // Send data to API
      const response = await fetch('http://localhost:8000/restaurants/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to add restaurant');
      }
      
      const result = await response.json();
      console.log('Restaurant added successfully:', result);
      setSuccess(true);
      
      // Reset form
      setName('');
      setAddress('');
      setRating(0);
      setPrice('');
      setCuisine('');
      setAmbience('');
      setGoodFor([]);
      
      // Close modal after short delay
      setTimeout(() => {
        if (onClose) onClose();
      }, 2000);
      
    } catch (err) {
      console.error('Error adding restaurant:', err);
      setError(err.message || 'Failed to add restaurant. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleCloseSnackbar = () => {
    setSuccess(false);
    setError(null);
  };

  return (
    <Container sx={{ position: 'relative', pt: 2, pb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5">Add a Restaurant</Typography>
        {onClose && (
          <IconButton onClick={onClose} aria-label="close" edge="end">
            <CloseIcon />
          </IconButton>
        )}
      </Box>
      
      <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <TextField 
          label="Name" 
          value={name} 
          onChange={(e) => setName(e.target.value)} 
          required 
          disabled={loading}
        />
        
        <TextField 
          label="Address" 
          value={address} 
          onChange={(e) => setAddress(e.target.value)} 
          required 
          disabled={loading}
          helperText="Please enter a complete address for accurate geocoding"
        />
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography component="legend">Rating</Typography>
          <Rating 
            value={rating} 
            onChange={(e, val) => setRating(val || 0)} 
            precision={0.5} 
            disabled={loading}
          />
        </Box>
        
        <TextField
          select
          label="Price Range"
          value={price}
          onChange={(e) => setPrice(e.target.value)}
          required
          disabled={loading}
        >
          {[0, 1, 2, 3].map((p) => (
            <MenuItem key={p} value={p}>${'$'.repeat(p)}</MenuItem>
          ))}
        </TextField>

        <Divider sx={{ my: 1 }}>
          <Typography variant="body2" color="text.secondary">Restaurant Attributes</Typography>
        </Divider>
        
        {/* Cuisine Selection - Single Select */}
        <FormControl fullWidth disabled={loading}>
          <InputLabel id="cuisine-label">Cuisine</InputLabel>
          <Select
            labelId="cuisine-label"
            id="cuisine-select"
            value={cuisine}
            label="Cuisine"
            onChange={(e) => setCuisine(e.target.value)}
          >
            {attributeOptions.Cuisines.map((option) => (
              <MenuItem key={option} value={option}>{option}</MenuItem>
            ))}
          </Select>
          <FormHelperText>Select one cuisine type</FormHelperText>
        </FormControl>
        
        {/* Ambience Selection - Single Select */}
        <FormControl fullWidth disabled={loading}>
          <InputLabel id="ambience-label">Ambience</InputLabel>
          <Select
            labelId="ambience-label"
            id="ambience-select"
            value={ambience}
            label="Ambience"
            onChange={(e) => setAmbience(e.target.value)}
          >
            {attributeOptions.Ambiences.map((option) => (
              <MenuItem key={option} value={option}>{option}</MenuItem>
            ))}
          </Select>
          <FormHelperText>Select one ambience type</FormHelperText>
        </FormControl>
        
        {/* GoodFor Selection - Multiple Select */}
        <FormControl fullWidth disabled={loading}>
          <InputLabel id="goodfor-label">Good For</InputLabel>
          <Select
            labelId="goodfor-label"
            id="goodfor-select"
            multiple
            value={goodFor}
            onChange={handleGoodForChange}
            input={<OutlinedInput id="select-multiple-chip" label="Good For" />}
            renderValue={(selected) => (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {selected.map((value) => (
                  <Chip key={value} label={value} />
                ))}
              </Box>
            )}
          >
            {attributeOptions.GoodFor.map((option) => (
              <MenuItem key={option} value={option}>{option}</MenuItem>
            ))}
          </Select>
          <FormHelperText>Select all that apply</FormHelperText>
        </FormControl>

        <Stack direction="row" spacing={2} justifyContent="flex-end" mt={2}>
          {onClose && (
            <Button variant="outlined" onClick={onClose} disabled={loading}>
              Cancel
            </Button>
          )}
          <Button 
            type="submit" 
            variant="contained" 
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Submit'}
          </Button>
        </Stack>
      </Box>
      
      <Snackbar open={!!error} autoHideDuration={6000} onClose={handleCloseSnackbar}>
        <Alert onClose={handleCloseSnackbar} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
      
      <Snackbar open={success} autoHideDuration={6000} onClose={handleCloseSnackbar}>
        <Alert onClose={handleCloseSnackbar} severity="success" sx={{ width: '100%' }}>
          Restaurant added successfully!
        </Alert>
      </Snackbar>
    </Container>
  );
}