Parkinson Diagnosis Using MRI

This project focuses on detecting Parkinson’s disease using MRI brain scans with the help of machine learning. It is built as a full-stack application with a Python backend handling the model inference and a React frontend for user interaction.

The main idea is to make Parkinson’s diagnosis more accessible by allowing users to upload MRI scans and receive predictions through a simple web interface.

What this project does

Takes MRI images as input

Processes and analyzes them using a trained ML model

Predicts whether the MRI shows signs of Parkinson’s disease

Displays the result through a web interface

This project is intended for learning, experimentation, and demonstration purposes, not for real clinical diagnosis.

Project structure
parkinson-diagnosis-using-mri/
├── backend/
│   ├── app.py                  # Backend server and API endpoints
│   ├── model/                  # Trained model files
│   ├── utils.py                # Image preprocessing and helper functions
│   ├── requirements.txt        # Python dependencies
│
├── frontend-react/
│   ├── public/                 # Static files
│   ├── src/                    # React source code
│   ├── package.json            # Frontend dependencies
│
├── .gitignore
└── README.md

Tech stack

Backend

Python

Flask (or similar lightweight framework)

Machine Learning / Deep Learning libraries

OpenCV / PIL for image processing

Frontend

React

JavaScript

HTML and CSS

How it works

The user uploads an MRI scan from the frontend.

The image is sent to the backend API.

The backend preprocesses the image.

The trained model runs inference on the MRI.

The prediction result is sent back to the frontend and displayed.

Setup instructions
Backend setup

Go to the backend directory:

cd backend


Create a virtual environment:

python -m venv venv


Activate it:

Windows:

venv\Scripts\activate


macOS/Linux:

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Start the backend server:

python app.py


The backend should now be running locally.

Frontend setup

Go to the frontend directory:

cd frontend-react


Install dependencies:

npm install


Start the development server:

npm start


The frontend will run in your browser and connect to the backend API.

Model details

The model is trained to classify MRI scans into Parkinson’s or non-Parkinson’s categories.

Input images are preprocessed before inference.

The exact architecture and training details can be extended or replaced depending on experimentation needs.

You can improve results by training on a larger dataset or experimenting with different CNN architectures.

Limitations

This project is not medically certified.

Accuracy depends heavily on dataset quality and size.

MRI preprocessing assumptions may not generalize to all scanners or datasets.

Future improvements

Add detailed evaluation metrics (accuracy, precision, recall)

Improve UI and validation feedback

Add support for multiple MRI views

Deploy the application to cloud platforms

Add authentication and user history

Contributing

If you want to contribute:

Fork the repository

Create a new branch

Make your changes

Submit a pull request

Suggestions and improvements are always welcome.
