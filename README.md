# Predictive Analytics Web Application

## Overview

This web application uses predictive analytics to recommend products for a given customer based on historical data. It is designed for an e-commerce company to enhance its marketing strategy.

## Features

- Predictive analytics powered by neural collaborative filtering.
- User-friendly web interface for entering customer ID and receiving product recommendations.
- Visual representation of recommended products with images.

## Prerequisites

- Python (3.10 or later)
- Flask
- Flask-Cors
- Numpy
- Requests
- Tensorflow
- Keras
- Pandas (for the example neural collaborative filtering implementation)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt

## Usage

1. **Run the Flask application:**
2. **Open your web browser and go to http://localhost:5000.**
3. **Enter a customer ID in the form and click "Predict."**
4. **View the recommended products with images.**

## Customization

- To customize the predictive model, modify the `neural_collaborative_filtering` function in `model_module.py`.
- Update the HTML and CSS files in the `templates` and `static` directories for UI customization.

## Contributing

If you'd like to contribute to this project, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
