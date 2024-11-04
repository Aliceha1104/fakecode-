import React, { useState, useEffect } from 'react';
import '../styles/pages/history.css';
import NavBar from '../components/NavBar';
import Footer from '../components/Footer';

export default function PredictHistory() {
  const [predictions, setPredictions] = useState([]);
  const [salesPredictions, setSalesPredictions] = useState([]);
  const [exportType, setExportType] = useState('json');

  const fetchPredictions = () => {
    fetch('http://127.0.0.1:8000/prediction-history/')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        return response.json();
      })
      .then((data) => {
        setPredictions(data.predictions);
      })
      .catch((error) => {
        alert('Error fetching history: ' + error.message);
      });
  };

  const fetchSalesPredictions = () => {
    fetch('http://127.0.0.1:8000/sale-prediction-history/')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        return response.json();
      })
      .then((data) => {
        setSalesPredictions(data.predictions);
      })
      .catch((error) => {
        alert('Error fetching sales prediction history: ' + error.message);
        setSalesPredictions([]); // Ensure state is set to an empty array on error
      });
  };

  useEffect(() => {
    fetchPredictions();
    fetchSalesPredictions();
  }, []);

  const handleDelete = (id) => {
    if (window.confirm('Delete this prediction?')) {
      fetch(`http://127.0.0.1:8000/delete-prediction/${id}`, {
        method: 'DELETE',
      })
        .then((response) => {
          if (response.ok) {
            alert('Deleted successfully!');
            fetchPredictions();
          } else {
            return response.json().then((data) => {
              alert(data.detail || 'Failed to delete the prediction. Please try again.');
            });
          }
        })
        .catch((error) => {
          alert('Error deleting prediction: ' + error.message);
        });
    }
  };

  const handleExport = () => {
    if (exportType === 'json') {
      exportAsJSON();
    } else if (exportType === 'csv') {
      exportAsCSV();
    }
  };

  const exportAsJSON = () => {
    const dataStr = JSON.stringify(predictions, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'predictions.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportAsCSV = () => {
    const csvRows = [
      ['ID', 'Distance to CBD', 'Built Year', 'Building Area (sqm)', 'Landsize (sqm)', 'Location', 'Total Rooms', 'Property Type', 'Predicted Price (AUD)']
    ];

    predictions.forEach((prediction) => {
      const totalRooms = getTotalRooms(prediction.house_data);
      const propType = getPropertyType(prediction.house_data.prop_type);
      const row = [
        prediction.id,
        prediction.house_data.cbd_distance,
        prediction.house_data.built_year,
        prediction.house_data.building_area,
        prediction.house_data.landsize,
        prediction.house_data.suburb_name,
        totalRooms,
        propType,
        prediction.predicted_price
      ];
      csvRows.push(row);
    });

    const csvContent = csvRows.map((row) => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'predictions-history.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getPropertyType = (propType) => {
    return propType === 'h' ? 'House' : propType === 'u' ? 'Unit' : 'Townhouse';
  };

  const getTotalRooms = (houseData) => {
    return houseData.bedroom + houseData.bathroom + houseData.car_garage;
  };

  return (
    <>
      <NavBar />
      <div className="container">
        <div className="row mt-3">
          <h1 className="text-center text-uppercase">Predictions History</h1>
          <div className="col-12">
            <table className="table">
              <thead className='table-danger text-center'>
                <tr>
                  <th scope="col">ID</th>
                  <th scope="col">House Data</th>
                  <th scope="col">Predicted Price (AUD)</th>
                  <th scope="col">Action</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((prediction) => (
                  <tr key={prediction.id}>
                    <th className='text-center' scope="row">{prediction.id}</th>
                    <td>
                      <ul>
                        {prediction.house_data.cbd_distance !== 0 && (
                          <li>Distance to CBD(km): {prediction.house_data.cbd_distance}</li>
                        )}
                        <li>Built year: {prediction.house_data.built_year}</li>
                        {prediction.house_data.building_area !== 0 && (
                          <li>Building area(sqm): {prediction.house_data.building_area}</li>
                        )}
                        {prediction.house_data.landsize !== 0 && (
                          <li>Landsize(sqm): {prediction.house_data.landsize}</li>
                        )}
                        <li>Location: {prediction.house_data.suburb_name}</li>
                        <li>
                          Total Rooms: {prediction.house_data.bedroom} bedroom, {prediction.house_data.bathroom} bathroom, {prediction.house_data.car_garage} car-garage
                        </li>
                        <li>Property Type: {getPropertyType(prediction.house_data.prop_type)}</li>
                      </ul>
                    </td>
                    <td className='text-center'>{prediction.predicted_price.toLocaleString()}</td>
                    <td className='text-center'>
                      <button
                        className="btn btn-danger"
                        onClick={() => handleDelete(prediction.id)}
                      >
                        X
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="d-flex flex-column justify-content-center align-items-center mt-3">
              <div className='ms-3 mb-3'>Export As
                <select className='ms-2' value={exportType} onChange={(e) => setExportType(e.target.value)}>
                  <option value='json'>JSON</option>
                  <option value='csv'>CSV</option>
                </select>
                <button className='btn btn-danger btn-sm ms-2' onClick={handleExport}>Export</button>
              </div>
            </div>
          </div>
        </div>

        <div className="row mt-5">
          <h1 className="text-center text-uppercase">Sales Predictions History</h1>
          <div className="col-12">
          <table className="table">
              <thead className='table-danger text-center'>
                <tr>
                  <th scope="col">ID</th>
                  <th scope="col">House and MarketData</th>
                  <th scope="col">Property Score Predict</th>
                  <th scope="col">Action</th>
                </tr>
              </thead>
              <tbody>
              {salesPredictions.map((salesPrediction) => (
                  <tr key={salesPrediction.id}>
                    <th className='text-center' scope="row">{salesPrediction.id}</th>
                    <td>
                      <ul>
                        <li>Predicted Result: {salesPrediction.predicted_result}</li>
                        <li>Predicted Status: {salesPrediction.predicted_status}</li>
                        <li>Price: {salesPrediction.price.toLocaleString()}</li>
                        <li>Median Price: {salesPrediction.median_price.toLocaleString()}</li>
                        <li>Median Rental: {salesPrediction.median_rental.toLocaleString()}</li>
                      </ul>
                    </td>
                    <td className='text-center'>{salesPrediction.predicted_result.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <Footer />
    </>
  );
}