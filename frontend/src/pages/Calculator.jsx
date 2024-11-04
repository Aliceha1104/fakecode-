import React, { useState, useRef } from 'react';
import '../styles/pages/calculator.css';
import {reAgencies} from '../data/reagency';
import { handleScroll, handleInput } from '../services/formAction';
import NavBar from '../components/NavBar';
import Footer from '../components/Footer';

export default function Calculator() {
  // Handle form input changes
  const handleInputChange = (e) => handleInput(e, setFormData);
  const predictResult = useRef(null);
  const initialFormData = {
    price: '',
    cbd_distance: '',
    bedroom: '',
    bathroom: '',
    car_garage: '',
    landsize: '',
    re_agency: '',
    median_price: '',
    median_rental: ''
  };

  const validateForm = () => {
    const newErrors = {};

    const allFieldsEmpty =
    !formData.price &&
    !formData.cbd_distance &&
    !formData.bedroom &&
    !formData.bathroom &&
    !formData.car_garage &&
    !formData.landsize &&
    !formData.re_agency &&
    !formData.median_price &&
    !formData.median_rental;

    if(allFieldsEmpty){
      newErrors.general = 'You must fill in at least one data';
    }
    if (formData.price < 0) {
      newErrors.price = 'Price must be a positive value, exclude 0';
    }

    if (formData.cbd_distance < 0) {
      newErrors.cbd_distance = 'Distance to CBD must be a positive value';
    }

    if (!Number.isInteger(Number(formData.bedroom)) || formData.bedroom < 0) {
      newErrors.bedroom = 'Number of bedrooms must be a positive integer.';
    }
    if (!Number.isInteger(Number(formData.bathroom)) || formData.bathroom < 0) {
      newErrors.bathroom = 'Number of bathrooms must be a positive integer.';
    }

    if (!Number.isInteger(Number(formData.car_garage)) || formData.car_garage < 0) {
      newErrors.car_garage = 'Number of car garage spaces must be a positive integer.';
    }

    if (formData.landsize < 0){
      newErrors.landsize = "Landsize must be a positive value, exclude 0.";
    }

    if (formData.median_price < 0){
      newErrors.median_price = "Median Price must be a positive value, exclude 0.";
    }

    if (formData.median_rental < 0){
      newErrors.median_rental = "Median Rental must be a positive value, exclude 0.";
    }

     // Check if entered re agency is in the list
    if (formData.re_agency && !reAgencies.includes(formData.re_agency) && formData.re_agency !== "Other") {
      newErrors.re_agency = 'Please choose Other if the re agency you entered is not exist';
    }
    setErrors(newErrors);

    //return true if no errors, otherwise false
    return Object.keys(newErrors).length == 0;
  }

  // State to hold form input values and errors and result
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [formData, setFormData] = useState(initialFormData);
  const [errors, setErrors] = useState({});

  const handleSubmit = async (e) => {
    e.preventDefault();

    //assign value to empty fiels
    const updatedFormData = {
    ...formData,
    price: formData.price || 0,
    cbd_distance: formData.cbd_distance || 0,
    bedroom: formData.bedroom || 1,
    bathroom: formData.bathroom || 1,
    car_garage: formData.car_garage || 0,
    landsize: formData.landsize || 0,
    re_agency: formData.re_agency || "Other",
    median_price: formData.median_price || 0,
    median_rental: formData.median_rental || 0
    };

    if(!validateForm()){
      return;
    }
    handleScroll(predictResult);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict-sale-potential/',{
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedFormData),
      });
      if(response.ok){
        const data = await response.json();
        const predictedResult = data.predicted_result;
        const predictedSales = data.predicted_status;
        setStatus(predictedSales);
        setResult(predictedResult);
      }else {
        const errorData = await response.json();
        if (Array.isArray(errorData.detail) && errorData.detail.length > 0) {
          const errorMessage = errorData.detail[0].msg; // Get the specific error message
          throw new Error(errorMessage);
        }else {
          throw new Error("Unknown error occurred");
        }
      }
    }catch(error){
      alert(`Error: ${error.message}`);
    }
  }

  //reset action
  const handleReset = () => {
    setFormData(initialFormData);
    setErrors({});
    setStatus(null);
    setResult(null);
  }
  return (
    <>
      <NavBar />
      <div className="container-fluid mt-3 mb-3 ps-5 pe-5">
        <div className="row">
          <div className="introduction-content col-lg-12 col-md-12 col-sm-12 text-center">
            <h1 className='text-uppercase'>Mortgage Calculator</h1>
            <p className='description me-5 ms-5 d-none d-lg-block d-md-block d-sm-block'>
              We've made it easy for you to better understand your finances with our handy home loan calculator. By working out your estimated loan amount, monthly repayments and ranking property based on market status. You can enjoy the confidence of knowing what you can afford.
            </p>
          </div>
          <div className="col-lg-5 col-md-6 col-sm-12 mt-4">
            <h2 className='text-uppercase text-center'>Property details</h2>
            <p className='fst-italic text-danger text-center'>If you don't know any data, leave the field empty.</p>
            <form onSubmit={handleSubmit}>
              <fieldset className='form-bg p-3 rounded mb-4'>
                <label htmlFor='price' className='form-label'>Property Price</label>
                <div className="input-group mb-3">
                  <span className='input-group-text'>$</span>
                  <input className='form-control' type='number' placeholder='0' id='price' value={formData.price} onChange={handleInputChange}/>
                </div>
                {errors.price && <p className='errors-msg text-danger'>{errors.price}</p>}
                <div className="mb-3">
                  <div className="row">
                    <div className="col-12">
                      <input className='form-control' type='number' placeholder='Distance to CBD...' id='cbd_distance' value={formData.cbd_distance} onChange={handleInputChange}/>
                      {errors.cbd_distance && <p className='errors-msg text-danger'>{errors.cbd_distance}</p>}
                    </div>
                  </div>
                </div>
                <div className="mb-3">
                  <div className="row">
                  <div className="col-4">
                      <label htmlFor='bedroom' className='form-label'>Bedroom</label>
                      <input className='form-control' type='number' placeholder='0' id='bedroom' value={formData.bedroom} onChange={handleInputChange}/>
                      {errors.bedroom && <p className='errors-msg text-danger'>{errors.bedroom}</p>}
                    </div>
                    <div className="col-4">
                      <label htmlFor='bathroom' className='form-label'>Bathroom</label>
                      <input className='form-control' type='number' placeholder='0' id='bathroom' value={formData.bathroom} onChange={handleInputChange}/>
                      {errors.bathroom && <p className='errors-msg text-danger'>{errors.bathroom}</p>}
                    </div>
                    <div className="col-4">
                      <label htmlFor='carGarage' className='form-label'>Car-Garage</label>
                      <input className='form-control' type='number' placeholder='0' id='car_garage' value={formData.car_garage} onChange={handleInputChange}/>
                      {errors.car_garage && <p className='errors-msg text-danger'>{errors.car_garage}</p>}
                    </div>
                    </div>
                </div>
                <div className="mb-3">
                  <div className="row">
                    <div className="col-6">
                    <label htmlFor='landsize' className='form-label'>Landsize (in sqm)</label>
                    <input className='form-control' type='number' placeholder='0' id='landsize' value={formData.landsize} onChange={handleInputChange}/>
                    {errors.landsize && <p className='errors-msg text-danger'>{errors.landsize}</p>}
                    </div>
                    <div className="col-6">
                      <label htmlFor='reAgency' className='form-label'>Real Estate Agency</label>
                      <input
                      list="reAgencies"
                      className='form-control'
                      id='re_agency'
                      value={formData.re_agency}
                      onChange={handleInputChange}
                      placeholder='Type or select a RE Agent...'/>
                      <datalist id="reAgencies">
                        {reAgencies.map((reAgency, index) => (
                          <option key={index} value={reAgency}></option>
                        ))}
                        <option value="Other"></option>
                      </datalist>
                      {errors.re_agency && <p className='errors-msg text-danger'>{errors.re_agency}</p>}
                    </div>
                  </div>
                </div>
                <div className="row">
                  <div className="col-12">
                    <label htmlFor='medianPrice' className='form-label'>Median Price</label>
                    <div className="input-group mb-3">
                      <span className='input-group-text'>$</span>
                      <input className='form-control' type='number' placeholder='0' id='median_price' value={formData.median_price} onChange={handleInputChange}/>
                    </div>
                    {errors.median_price && <p className='errors-msg text-danger'>{errors.median_price}</p>}
                  </div>
                </div>
                <div className="row">
                  <div className="col-12">
                  <label htmlFor='medianRental' className='form-label'>Median Rental</label>
                    <div className="input-group mb-3">
                      <span className='input-group-text'>$</span>
                      <input className='form-control' type='number' placeholder='0' id='median_rental' value={formData.median_rental} onChange={handleInputChange}/>
                    </div>
                    {errors.median_rental && <p className='errors-msg text-danger'>{errors.median_rental}</p>}
                  </div>
                </div>
                {errors.general && <p className='text-danger text-center'>{errors.general}</p>}
                <div className="d-flex col-12 justify-content-center">
                <button type="submit" className='btn btn-danger'>Get Result</button>
                <button type="button" className='btn btn-secondary ms-2' onClick={handleReset}>Reset</button>
                </div>
              </fieldset>
            </form>
          </div>
          <div  ref={predictResult} className="col-lg-7 col-md-6 col-sm-12 col-12 d-flex flex-column justify-content-center align-items-center text-center mt-4">
            <h2 className='text-danger text-uppercase'>Analyze Result</h2>
            <p className='analyze'>The potential sale of this property is <span className='fw-bold fst-italic sub-analyze'>{result !== null ? `${result}`: '0'} %</span></p>
            <p className='analyze'>This is a <span className='text-uppercase text-danger sub-analyze'>{status !== null ? `${status}` : '...'}</span> investment !!!</p>
          </div>
        </div>
      </div>
      <Footer />
    </>
  );
}