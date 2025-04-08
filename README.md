# Unsupervised Learning Analyzer

An interactive Flask web application for analyzing data using K-means clustering. This application is based on the unsupervised learning code in the `unsupervised.py` file.

## Features

- K-means clustering with adjustable number of clusters
- Elbow method for determining optimal number of clusters
- Interactive data visualization
- Support for different analysis types:
  - Taxpayer analysis (Average net tax and Average total deductions)
  - Business analysis (Average total business income and Average total business expenses)

## Installation

```bash
# Clone the repository
git clone https://github.com/fenago/demo88.git
cd demo88

# Create a virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

## Running the Application

```bash
flask run
# or
python app.py
```

Then open your browser and navigate to http://127.0.0.1:5000/

## Data

The application uses Australian tax statistics data by default from:
https://raw.githubusercontent.com/fenago/datasets/refs/heads/main/taxstats2015.csv

You can also provide your own CSV data URL that contains similar columns.

## Application Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates
- `static/` - CSS and JavaScript files
- `unsupervised.py` - Original unsupervised learning code