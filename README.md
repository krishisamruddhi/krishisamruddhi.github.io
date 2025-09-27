# Krishisamruddhi - Crop Disease Detection

## Description

**Krishisamruddhi** is a web application built with Django that helps farmers detect diseases in their crops. Users can upload an image of a crop leaf, and the application will predict the disease with a certain confidence level. This tool aims to assist farmers in identifying crop diseases early, enabling them to take timely action and protect their harvest.

-----

## Features

  * **Disease Prediction:** Upload an image of a crop leaf to get a prediction of the disease.
  * **Confidence Score:** View the confidence level of the prediction to gauge its accuracy.
  * **Simple Interface:** Easy-to-use web interface for uploading images and viewing results.

-----

## Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://your-repository-url.git
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
      * On Windows:
        ```bash
        venv\Scripts\activate
        ```
      * On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the development server:**
    ```bash
    python manage.py runserver
    ```

-----

## Usage

1.  Navigate to the home page.
2.  Click on the "Detect Disease" button.
3.  Upload an image of a crop leaf.
4.  The application will display the predicted disease and the confidence score.

-----

## Deployment

This web application is deployed using **Render.com**.

-----

## Technologies Used

  * **Backend:** Django
  * **Machine Learning:** PyTorch
  * **Frontend:** HTML, CSS
  * **Deployment:** Render.com, Gunicorn, Whitenoise
