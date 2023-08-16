# Breast Cancer Prediction Application using Linear Regression and Streamlit

This project implements a machine learning model using linear regression to predict whether a breast mass is benign or malignant based on cell nuclei measurements. The graphical user interface (GUI) is built using Streamlit, allowing users to modify cell nuclei measurements and observe real-time changes in the model's predictions. The GUI also features a radar chart to visualize the input data effectively.


## Streamlit App

The Streamlit app for this project can be accessed at [https://breast-cancer-prediction-1.streamlit.app](https://breast-cancer-prediction-1.streamlit.app). It offers an intuitive interface for interacting with the model and visualizing predictions.


## Project Overview

The aim of this project is to provide a user-friendly tool for predicting whether a breast mass is benign or malignant using a linear regression model. By leveraging cell nuclei measurements from digitized images of fine needle aspirates (FNAs), the model makes predictions and allows users to interactively explore the impact of changing input variables on the prediction outcome.

## Usage

1. Launch the Streamlit app [here](https://breast-cancer-prediction-1.streamlit.app).
2. Use the input sliders in the sidebar to alter cell nuclei measurements.
3. Observe how changes in the measurements affect the model's prediction.
4. The radar chart provides a visual representation of the data points you modify.

## Getting Started

To run this project locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies. You may need to set up a virtual environment for this purpose: pip install -r requirements.txt
3. Run the Streamlit app by going into the app folder and using the command: streamlit run main.py
4. Access the app through your web browser at `http://localhost:8501`.

## Dependencies

This project uses the following Python libraries:

- pickle5
- pandas
- plotly
- numpy
- streamlit


## Data

The features for the machine learning model are computed from digitized images of fine needle aspirates (FNAs) of breast masses. These features describe characteristics of the cell nuclei present in the image. The dataset can be found on kaggle at [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)


## Contributing

Contributions to this project are welcome. If you find any issues or want to enhance the functionality, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).



