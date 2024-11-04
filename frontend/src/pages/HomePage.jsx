import React from 'react';
import { NavLink } from 'react-router-dom';
import { useState, useRef, useEffect } from 'react';
import '../styles/pages/home.css';
import house_logo from '../assets/house_logo.png';
import market_logo from '../assets/market_logo.png';
import ML_logo from '../assets/ML_logo.png';
import CL_logo from '../assets/calculator_logo.png'
import user_1 from '../assets/user_1.png';
import user_2 from '../assets/user_2.png';
import user_3 from '../assets/user_3.png';
import { properties } from '../data/properties';
import { handleScroll } from '../services/formAction';
import NavBar from '../components/NavBar';
import Footer from '../components/Footer';

export default function HomePage() {

    //scroll to third-content when click on second-content button
    const thirdContentRef = useRef(null);
    const scrollView = () => handleScroll(thirdContentRef);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [propertiesPerColumn, setPropertiesPerColumn] = useState(window.innerWidth < 992 ? 4 : 3);
    useEffect(() => {
        const handleResize = () => {
            let newPropertiesPerColumn = 3; // Default for larger screens
            if (window.innerWidth < 768) {
                newPropertiesPerColumn = 1; // Extra small devices
            } else if (window.innerWidth < 992) {
                newPropertiesPerColumn = 2; // Medium devices
            } else {
                newPropertiesPerColumn = 3; // Large devices
            }
            setPropertiesPerColumn(newPropertiesPerColumn);
        };
        // Initial check to set properties based on the current window size
        handleResize();
        window.addEventListener('resize', handleResize);
        // Cleanup function to remove event listener
        return () => {
            window.removeEventListener('resize', handleResize);
        };

    }, []);

    //filter house by type, price and distance
    //set up state for type, price, distance
    const [selectedType, setSelectedType] = useState('');
    const [selectedPrice, setSelectedPrice] = useState('');
    const [selectedDistance, setSelectedDistance] = useState('');

    const handleTypeChange = (event) => {
        setSelectedType(event.target.value);
        setCurrentIndex(0);
    }

    const handlePriceChange = (event) => {
        setSelectedPrice(event.target.value);
        setCurrentIndex(0);
    }

    const handleDistanceChange = (event) => {
        setSelectedDistance(event.target.value);
        setCurrentIndex(0);
    }

    //filter by price
    const filterByPrice = (price, selectedPrice) => {
        if (selectedPrice === "") return true; // Show all if no price selected
        switch (selectedPrice) {
            case '100000 - 300000':
                return price >= 100000 && price <= 300000;
            case '300000 - 500000':
                return price > 300000 && price <= 500000;
            case '500000 - 700000':
                return price > 500000 && price <= 700000;
            case '700000 - 900000':
                return price > 700000 && price <= 900000;
            case '900000+':
                return price > 900000;
            default:
                return true;
        }
    };

    //filter by distance
    const filterByDistance = (distance, selectedDistance) => {
        if(selectedDistance === "") return true;
        switch(selectedDistance){
            case '1-5':
                return distance >= 1 && distance <= 5;
            case '6-10':
                return distance >= 6 && distance <= 10;
            case '11-20':
                return distance >= 11 && distance <= 20;
            case '20+':
                return distance >20;
            default:
                return true;
        }
    };

    //filtered data
    const filteredPropertites = properties.filter((property) => {
        const matchesType = selectedType === '' || property.type === selectedType;
        const matchesPrice = filterByPrice(property.price, selectedPrice);
        const matchesDistance = filterByDistance(property.distance_to_cbd, selectedDistance);
        return matchesType && matchesPrice && matchesDistance;
    })

    //handle reset button
    const handleReset = () => {
        setSelectedDistance('');
        setSelectedType('');
        setSelectedPrice('');
    }

    //handle pagination for display house data
    const handleNext = () => {
        if (currentIndex + propertiesPerColumn < filteredPropertites.length) {
            setCurrentIndex((prevIndex) => prevIndex + propertiesPerColumn);
        }
    };

    const handlePrev = () => {
        if (currentIndex - propertiesPerColumn >= 0) {
            setCurrentIndex((prevIndex) => prevIndex - propertiesPerColumn);
        }
    };


    return (
        <>
            <NavBar />
            <div className="container-fluid first-content">
                <div className='container'>
                    <div className="about-content text-center text-black">
                        <div className="row mt-5">
                            <div className="col-lg-12 col-md-12 col-sm-12 col-12 about-box">
                                <h1 className='text-uppercase responsive-title'>About Us - UMD TEAM</h1>
                                <div className="row">
                                    <div className="col-lg-6 col-md-12 col-sm-12 col-12 p-3 introduction">
                                        <p className='team-intro'>UMD is creating a user-friendly platform that lets people input specific property details to get real-time market price predictions, powered by machine learning. </p>
                                        <p className='d-none d-lg-block'>With a blend of cutting-edge web development and advanced data analysis, UMD is redefining real estate insights, making it easier than ever for users to grasp property values and make informed decisions in today’s housing market.</p>
                                    </div>
                                    <div className="col-lg-6 col-md-12 col-sm-12 col-12">
                                        <div className="row mt-4">
                                            <div className="col-lg-4 col-md-4 col-sm-4 col-4">
                                                <img src={user_1} alt='user_logo' className='w-50'></img>
                                                <h5 className='name text-uppercase'>Melvin Goxhaj</h5>
                                                <p className='d-none d-lg-block'><small>AI Software Engineer, QA Manager.</small></p>
                                            </div>
                                            <div className="col-lg-4 col-md-4 col-sm-4 col-4">
                                                <img src={user_2} alt='user_logo' className='w-50'></img>
                                                <h5 className='name text-uppercase'>Upek Malankandalage</h5>
                                                <p className='d-none d-lg-block'><small>Games Developer, QA Manager.</small></p>
                                            </div>
                                            <div className="col-lg-4 col-md-4 col-sm-4 col-4">
                                                <img src={user_3} alt='user_logo' className='w-50'></img>
                                                <h5 className='name text-uppercase'>Phuong Doanh HA</h5>
                                                <p className='d-none d-lg-block'><small>Cyber security analyst, QA Manager.</small></p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div className="container-fluid second-content">
                <h1 className='text-center'>About Our Website</h1>
                <div className="container-fluid mt-4">
                    <div className="row text-center mx-auto my-auto">
                        <div className="col-lg-3 col-md-3 col-sm-6 col-6">
                            <img src={house_logo} className='border-0 logo' alt='house_logo'></img>
                            <h5 className='responsive-title'>Find your perfect home</h5>
                            <p className='d-none d-lg-block d-md-block responsive-text'>Let's find a suitable house for you! Our website have many comfortable and beautiful properties.</p>
                            <button className='border border-0 link' onClick={scrollView}>Search</button>
                        </div>
                        <div className="col-lg-3 col-md-3 col-sm-6 col-6">
                            <img src={market_logo} className='border-0 logo' alt='market_logo'></img>
                            <h5 className='responsive-title'>Market Research</h5>
                            <p className='d-none d-lg-block d-md-block responsive-text'>Three type of charts, each charts provided different data with interesting interactive for you!</p>
                            <NavLink className='link' to='/market-charts'>See Market Charts</NavLink>
                        </div>
                        <div className="col-lg-3 col-md-3 col-sm-6 col-6">
                            <img src={ML_logo} className='border-0 logo' alt='ML_logo'></img>
                            <h5 className='responsive-title'>Get your Predicted Price</h5>
                            <p className='d-none d-lg-block d-md-block responsive-text'>Our Machine Learning model can predict house prices based on various factors with an accuracy of up to 70%</p>
                            <NavLink className='link' to='/predict-price'>Predict Your Own Price</NavLink>
                        </div>
                        <div className="col-lg-3 col-md-3 col-sm-6 col-6">
                            <img src={CL_logo} className='border-0 logo' alt='ML_logo'></img>
                            <h5 className='responsive-title'>Mortage Calculator</h5>
                            <p className='d-none d-lg-block d-md-block responsive-text'>We've made it easy for you to better understand your finances with our handy home loan calculator.</p>
                            <NavLink className='link' to='/calculator'>Calculate</NavLink>
                        </div>
                    </div>
                </div>
            </div>
            <div ref={thirdContentRef} className="container-fluid third-content mb-5">
                <div className="container">
                    <div className="row text-center">
                        <div className="col-12">
                            <h2 className='title'>LET'S FIND A RIGHT PROPERTY FOR YOU!</h2>
                        </div>
                        <div className="col-12">
                            <p className='sub-title'>Don't miss the opportunity to own a home of your own!</p>
                        </div>
                    </div>
                    <div className="row justify-content-center mb-3">
                        <div className="col-lg-8 col-md-12 col-sm-12 col-12">
                            <div className="d-flex justify-content-between">
                                <select className="filter-select-input form-select mx-2" value={selectedType} onChange={handleTypeChange}>
                                    <option value="">Property Type</option>
                                    <option value="Unit">Unit</option>
                                    <option value="Townhouse">Townhouse</option>
                                    <option value="House">House</option>
                                </select>
                                <select className="filter-select-input form-select mx-2" value={selectedPrice} onChange={handlePriceChange}>
                                    <option value="">Price</option>
                                    <option value='100000 - 300000'>100,000 - 300,000</option>
                                    <option value='300000 - 500000'>300,000 - 500,000</option>
                                    <option value='500000 - 700000'>500,000 - 700,000</option>
                                    <option value='700000 - 900000'>700,000 - 900,000</option>
                                    <option value='900000+'>Above 900,000</option>
                                </select>
                                <select className="filter-select-input form-select mx-2" value={selectedDistance} onChange={handleDistanceChange}>
                                    <option value="">Distance to CBD</option>
                                    <option value="1-5"> 1km - 5km</option>
                                    <option value="6-10"> 6km - 10km</option>
                                    <option value="11-20">10km - 20km</option>
                                    <option value="20+"> More than 20km</option>
                                </select>
                            </div>
                        </div>
                        <div className="col-12 text-center mt-2">
                            <button type='button' className='btn btn-danger' onClick={handleReset}>Reset Filter</button>
                        </div>
                    </div>
                </div>
                <div className="container mt-3">
                    <div className='row justify-content-center' >
                            {filteredPropertites.length === 0 ? (
                                <p className='text-center text-danger fst-italic'>No matching house</p>
                            ) : (
                                filteredPropertites.slice(currentIndex, currentIndex + propertiesPerColumn).map((property, index) => (
                                    <div className="col-lg-4 col-md-6 col-sm-10 col-10 property-card mt-3" key={index}>
                                        <div className="card h-100">
                                            <img src={property.image} alt={property.name} className="card-img-top" height={300} />
                                            <div className="card-body text-center">
                                                <div className="bg-danger-subtle rounded mb-2">
                                                <h3 className="card-title text-uppercase">${property.price.toLocaleString()}</h3>
                                                <p className="card-text">Address: {property.address}, {property.suburb}</p>
                                                </div>
                                                <hr className='border-2'/>
                                                <span className="card-text">Landsize: {property.landsize}m<sup>2</sup></span><br/>
                                                <span>Building Area: {property.building_area}m<sup>2</sup></span>
                                                <p className="card-text">Type: {property.type}</p>
                                                <p className="card-text">Distance to CBD: {property.distance_to_cbd}km</p>
                                                <div className="card-text">
                                                    <i className="fa-solid fa-bed col-2" /> {property.bedroom}
                                                    <i className="fa-solid fa-bath col-2" /> {property.bathroom}
                                                    <i className="fa-solid fa-warehouse col-2" /> {property.car_garage}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                    </div>
                    <div className="row mt-4">
                        <div className="col-12 d-flex justify-content-between">
                            <button onClick={handlePrev} className="btn btn-danger" disabled={currentIndex === 0}><i className="fa-solid fa-arrow-left"></i></button>
                            <button onClick={handleNext} className="btn btn-danger" disabled={currentIndex + propertiesPerColumn >= filteredPropertites.length}><i className="fa-solid fa-arrow-right"></i></button>
                        </div>
                    </div>
                </div>
            </div>
            <Footer />
        </>
    );
}
