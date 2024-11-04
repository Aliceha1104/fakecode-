import React, { useState, useRef } from 'react';
import { NavLink } from 'react-router-dom';
import '../styles/pages/predict.css';
import { suburbs } from '../data/suburb';
import { handleScroll, handleInput } from '../services/formAction';
import NavBar from '../components/NavBar';
import Footer from '../components/Footer';

export default function PredictPricePage() {
  const handleInputChange = (e) => handleInput(e, setFormData);
  const predictPriceRef = useRef(null);
  // State to hold form input values
  const initialFormData = {
    cbd_distance: '',
    bedroom: '',
    bathroom: '',
    car_garage: '',
    landsize: '',
    building_area: '',
    built_year: '',
    suburb_name: '',
    prop_type: ''
  };

  // State to hold predicted price and original AUD price
  const [price, setPrice] = useState(null);
  const [originalPrice, setOriginalPrice] = useState(null); // New state for original price
  const [currency, setCurrency] = useState('AUD');

  // Conversion rates
  const conversionRates = {
    AUD: 1, // Base currency
    USD: 0.65,
  };

  // Handle currency change and conversion
  const handleCurrencyChange = (e) => {
    const selectedCurrency = e.target.value;
    setCurrency(selectedCurrency);

    // If the originalPrice is set, perform conversion
    if (originalPrice !== null) {
      const convertedPrice = (originalPrice * conversionRates[selectedCurrency]).toLocaleString();
      setPrice(convertedPrice);
    }else {
      setPrice(null);
    }
  };

  // State to hold form input values and errors
  const [formData, setFormData] = useState(initialFormData);
  const [errors, setErrors] = useState({});

  //validate form input
  const validateForm = () => {
    const newErrors = {};
    // Check if all fields are empty
    const allFieldsEmpty =
    !formData.cbd_distance &&
    !formData.landsize &&
    !formData.building_area &&
    !formData.bedroom &&
    !formData.bathroom &&
    !formData.car_garage &&
    !formData.built_year &&
    !formData.suburb_name &&
    !formData.prop_type;

    if (allFieldsEmpty) {
      newErrors.general = "You must fill in at least one field.";
    }

    if (formData.cbd_distance < 0){
      newErrors.cbd_distance = 'Distance to CBD must be a non-negative value.';
    }

    if(formData.landsize == 0){
      newErrors.landsize = "Landsize cannot be 0. Leave the field emtpy if you don't know the value.";
    }

    if (formData.landsize < 0){
      newErrors.landsize = "Landsize must be a positive value, exclude 0.";
    }

    if (formData.building_area < 0){
      newErrors.building_area = 'Building area must be a positive value, exclude 0.';
    }

    if(!Number.isInteger(Number(formData.bedroom)) || formData.bedroom < 0){
      newErrors.bedroom = 'Number of bedrooms must be a positive integers.';
    }

    if(!Number.isInteger(Number(formData.bathroom)) || formData.bathroom < 0){
      newErrors.bathroom = 'Number of bathrooms must be a positive integers.';
    }

    if(!Number.isInteger(Number(formData.car_garage)) || formData.car_garage < 0){
      newErrors.car_garage = 'Number of car garage must be a positive integers.';
    }

    // Validate built_year: must be a number or "Before 1800"
    if(!Number.isInteger(Number(formData.built_year)) || formData.built_year < 0){
      newErrors.built_year = 'Built year must be a positive integers.';
    }

     // Check if entered suburb is in the suburbs list
    if (formData.suburb_name && !suburbs.includes(formData.suburb_name) && formData.suburb_name !== "Other") {
      newErrors.suburb_name = 'Please choose Other if the suburb you entered is not exist';
    }

    setErrors(newErrors);

    //return true if no errors, otherwise false
    return Object.keys(newErrors).length == 0;
  };


  // Submit the form, get predicted price from BE
  const handleSubmit = async (e) => {
    e.preventDefault();

    // Assign value to any empty fields
    const updatedFormData = {
      ...formData,
      bedroom: formData.bedroom || 1,
      bathroom: formData.bathroom || 1,
      car_garage: formData.car_garage || 0,
      cbd_distance: formData.cbd_distance || 0,
      built_year: formData.built_year || 2024,
      landsize: formData.landsize || 1,
      building_area: formData.building_area || 0,
      prop_type: formData.prop_type || 'u',
      suburb_name: formData.suburb_name || 'Other'
    };

    if(!validateForm()){
      return; //if form is invalid, dont process submission
    }
     //Handel scroll action
     handleScroll(predictPriceRef);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedFormData),
      });

      if (response.ok) {
        const data = await response.json();
        const predictedPrice = data.predicted_price
        setOriginalPrice(predictedPrice); // Store the original AUD price
        setPrice(predictedPrice); // Set the displayed price to original
      } else {
        const errorData = await response.json();
        // Check if detail is an array and extract the first error message
        if (Array.isArray(errorData.detail) && errorData.detail.length > 0) {
          const errorMessage = errorData.detail[0].msg; // Get the specific error message
          throw new Error(errorMessage);
        } else {
          throw new Error("Unknown error occurred");
        }
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };

  const handleReset = () => {
    setFormData(initialFormData);
    setErrors({}); //Clear any errors
    setPrice(null); // Reset the predicted price to its initial state (null)
    setOriginalPrice(null); // Reset the original price
    setCurrency('AUD'); // reset the currency to 'AUD'
  }

  return (
    <>
      <NavBar />
      <div className="container-fluid">
        <div className="row">
          <div className="col-lg-6 col-md-6 col-sm-12 col-12">
            <form onSubmit={handleSubmit}>
              <fieldset className='mt-5 mb-5 ms-5'>
                <legend className='form-legend text-center text-uppercase fw-bold'>
                  Get your own predicted house price
                </legend>
                <p className='fst-italic text-danger text-center'>If you don't know any data, leave the field empty.</p>
                <div className="form-bg p-4 rounded">
                  <div className="mb-3">
                    <input type='number' className='form-control' id='cbd_distance' value={formData.cbd_distance} placeholder='Distance to CBD (km)' onChange={handleInputChange} />
                    {errors.cbd_distance && <p className='errors-msg text-danger'>{errors.cbd_distance}</p>}
                  </div>
                  <div className="mb-3">
                    <div className="row">
                    <p className='form-label'>Number of Rooms in property</p>
                      <div className="col-lg-4 col-md-4 col-sm-4 col-4">
                        <input type='number' className='form-control' id='bedroom' value={formData.bedroom} placeholder='Bedrooms' onChange={handleInputChange} />
                        {errors.bedroom && <p className='errors-msg text-danger'>{errors.bedroom}</p>}
                      </div>
                      <div className="col-lg-4 col-md-4 col-sm-4 col-4">
                        <input type='number' className='form-control' id='bathroom' value={formData.bathroom} placeholder='Bathrooms' onChange={handleInputChange} />
                        {errors.bathroom && <p className='errors-msg text-danger'>{errors.bathroom}</p>}
                      </div>
                      <div className="col-lg-4 col-md-4 col-sm-4 col-4">
                        <input type='number' className='form-control' id='car_garage' value={formData.car_garage} placeholder='Garages' onChange={handleInputChange} />
                        {errors.car_garage && <p className='errors-msg text-danger'>{errors.car_garage}</p>}
                      </div>
                    </div>
                  </div>
                  <div className="mb-3">
                    <div className="row">
                      <div className="col-lg-6 col-md-6 col-sm-6 col-6">
                        <label htmlFor='landsize' className='form-label'>Landsize (sqm)</label>
                        <input type='number' className='form-control' id='landsize' value={formData.landsize} placeholder='...' onChange={handleInputChange} />
                        {errors.landsize && <p className='errors-msg text-danger'>{errors.landsize}</p>}
                      </div>
                      <div className="col-lg-6 col-md-6 col-sm-6 col-6">
                        <label htmlFor='area' className='form-label'>Building Area (sqm)</label>
                        <input type='number' className='form-control' id='building_area' value={formData.building_area} placeholder='...' onChange={handleInputChange} />
                        {errors.building_area && <p className='errors-msg text-danger'>{errors.building_area}</p>}
                      </div>
                    </div>
                  </div>
                  <div className="mb-3">
                    <div className="row">
                      <div className="col-lg-12 col-md-12 col-sm-12 col-12">
                        <label htmlFor='built_year' className='form-label'>Built Year</label>
                        <input type='number' className='form-control' id='built_year' value={formData.built_year} placeholder='Built year is...' onChange={handleInputChange}/>
                        {errors.built_year && <p className='errors-msg text-danger'>{errors.built_year}</p>}
                      </div>
                      </div>
                  </div>
                  <div className="mb-3">
                    <div className="row">
                      <div className="col-lg-12 col-md-12 col-sm-12 col-12">
                          <label htmlFor='suburb_name' className='form-label'>Suburb</label>
                            <input
                              list="suburbs"
                              className='form-control'
                              id='suburb_name'
                              value={formData.suburb_name}
                              onChange={handleInputChange}
                              placeholder='Type or select a suburb...'
                            />
                            <datalist id="suburbs">
                              {suburbs.map((suburb, index) => (
                                <option key={index} value={suburb} />
                              ))}
                              <option value='Other'></option>
                            </datalist>
                            {errors.suburb_name &&<p className='errors-msg text-danger'>{errors.suburb_name}</p>}
                        </div>
                    </div>
                    </div>
                  <div className="mb-3">
                    <label htmlFor='prop_type' className='form-label'>Property Type</label>
                    <select className='form-select' value={formData.prop_type} id='prop_type' onChange={handleInputChange}>
                      <option value=''>Select Property Type</option>
                      <option value='u'>Unit</option>
                      <option value='t'>Townhouse</option>
                      <option value='h'>House</option>
                    </select>
                  </div>
                  {errors.general && <p className='text-danger text-center'>{errors.general}</p>}
                  <div className="d-flex col-12 justify-content-center">
                    <button type="submit" className='btn btn-danger'>Predict Price</button>
                    <button type="button" className='btn btn-secondary ms-2' onClick={handleReset}>Reset</button>
                  </div>
                </div>
              </fieldset>
            </form>
          </div>
          <div ref={predictPriceRef} className="col-lg-6 col-md-6 col-sm-12 col-12 d-flex flex-column justify-content-center align-items-center mb-5">
            <div className="row justify-content-center">
              <div className="col-lg-12 col-md-12 col-sm-10 col-10 d-flex flex-column justify-content-center align-items-center text-center">
                <h2 className='text-uppercase'>The Predicted Price for this Property is</h2>
                <p className='display-4 fst-italic text-center'>{price !== null ? `${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) }` : '0'}</p>
                <select
                  className="form-select w-25"
                  value={currency}
                  onChange={handleCurrencyChange}
                >
                  <option value="AUD">AUD</option>
                  <option value="USD">USD</option>
                </select>
                <NavLink className='link mt-3 text-center' to="/predict-history">Your Predict History</NavLink>
              </div>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </>
  );
}
