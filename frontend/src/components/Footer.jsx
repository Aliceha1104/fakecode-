import React from 'react';
import '../styles/component/footer.css';

export default function Footer() {
  return (
        <footer className="container-fluid text-center">
          <div className="row">
            <div className="col-6 d-flex flex-column justify-content-center align-items-center pb-2">
              <h5 className="text">Get connected with us on social networks:</h5>
              <div className="d-flex justify-content-center">
                <a href="" className="text-reset mx-2">
                  <i className="fab fa-facebook-f"></i>
                </a>
                <a href="" className="text-reset mx-2">
                  <i className="fab fa-twitter"></i>
                </a>
                <a href="" className="text-reset mx-2">
                  <i className="fab fa-google"></i>
                </a>
                <a href="" className="text-reset mx-2">
                  <i className="fab fa-instagram"></i>
                </a>
                <a href="" className="text-reset mx-2">
                  <i className="fab fa-github"></i>
                </a>
              </div>
              <div className="sub-text d-none d-lg-block d-md-block">
                <span>Our team are free at all major social media platforms, feel free to contact from 9AM-5PM all weekdays.</span>
              </div>
            </div>
            <div className="col-6 d-flex flex-column justify-content-end align-items-center pb-2">
              <h4 className="text">Contact</h4>
              <span className='contact'><i className="fas fa-home me-3"></i> Shepparton, VIC 3630</span>
              <a href="mailto:melvingoxhaj@gmail.com" className='contact text-black'><i className="fas fa-envelope me-3"></i>melvingoxhaj@gmail.com</a>
              <p className='contact'><i className="fas fa-phone me-3"></i>0474213341</p>
            </div>
            <div className="col-12 copyright w-100 text-center">
                © {new Date().getFullYear()} All Rights Reserved
                <span className='d-none d-lg-block d-md-block d-sm-block'>| UMD Group | Website by team UMD</span>
            </div>
          </div>
      </footer>
  );
}
