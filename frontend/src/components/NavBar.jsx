import { NavLink } from 'react-router-dom';
import React from 'react';
import pink_logo from '../assets/pink_logo.jpg';
import '../styles/component/navbar.css';

export default function NavBar() {
  return (
    <>
        <div className="d-flex flex-grow-1 justify-content-center align-items-center">
          <div className="row">
            <div className="col-6">
              <img className="border-end border-dark" src={pink_logo} alt="logo" width={100} height={100}></img>
            </div>
            <div className="col-6 ps-1 fs-2 fw-bold">
              <div className="row">
                <div className="col-12">
                  <h2>UMD<br/>
                  Group</h2>
                </div>
              </div>
            </div>
          </div>
        </div>
      <nav className='navbar navbar-expand-lg mt-1'>
        <div className='container-fluid'>
          <NavLink className="navbar-brand" to="#"></NavLink>
          <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className='collapse navbar-collapse justify-content-center mt-2' id='navbarSupportedContent'>
            <ul className='navbar-nav mb-2 mb-lg-0 fw-medium'>
              <li className='nav-item'>
                <NavLink
                  className={({ isActive }) => isActive ? 'nav-link text-white active-link' : 'nav-link text-black'}
                  to="/"
                >
                  Home
                </NavLink>
              </li>
              <li className='nav-item'>
                <NavLink
                  className={({ isActive }) => isActive ? 'nav-link text-white active-link' : 'nav-link text-black'}
                  to="/predict-price"
                >
                  Predict Price
                </NavLink>
              </li>
              <li className='nav-item'>
                <NavLink
                  className={({ isActive }) => isActive ? 'nav-link text-white active-link' : 'nav-link text-black'}
                  to="/predict-history"
                >
                  Predict History
                </NavLink>
              </li>
              <li className='nav-item'>
                <NavLink
                  className={({ isActive }) => isActive ? 'nav-link text-white active-link' : 'nav-link text-black'}
                  to="/market-charts"
                >
                  Market Charts
                </NavLink>
              </li>
              <li className='nav-item'>
                <NavLink
                  className={({ isActive }) => isActive ? 'nav-link text-white active-link' : 'nav-link text-black'}
                  to="/calculator"
                >
                  Calculator
                </NavLink>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    </>
  );
}
