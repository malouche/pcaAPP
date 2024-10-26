# PCA Dashboard

A Streamlit web application for performing Principal Component Analysis (PCA) on your data.

## Features

- Interactive file upload
- Variable selection
- Customizable PCA parameters
- Interactive visualizations:
  - Scree plot with explained variance
  - Loadings matrix
  - Score matrix
  - Biplot
- Download results as CSV files

## Demo

You can access the live demo here: [Your Streamlit Share URL]

## Local Installation

1. Clone this repository:
```bash
git clone https://github.com/[your-username]/pca-dashboard.git
cd pca-dashboard
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your web browser and go to `http://localhost:8501`

## Deploy to Streamlit Share

1. Fork this repository to your GitHub account

2. Go to [Streamlit Share](https://share.streamlit.io)

3. Sign in with your GitHub account

4. Click "New app" and select this repository

5. Select the main branch and enter:
   - Repository: pca-dashboard
   - Branch: main
   - Main file path: app.py

6. Click "Deploy"

## Usage

1. Upload your CSV file (should have a header row and index column)
2. Select the variables you want to include in the PCA
3. Adjust the PCA options (scaling, number of components)
4. Explore the results in the different tabs
5. Download results as needed

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
