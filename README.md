# Parkinson Diagnosis Using MRI

This project focuses on detecting Parkinson’s disease using MRI brain scans with machine learning. It is built as a full-stack application with a Python backend that handles model inference and a React frontend for user interaction.

The goal of this project is to demonstrate how medical imaging and machine learning can be combined in a practical web application.

## What this project does

- Accepts MRI scans as input
- Preprocesses images before inference
- Uses a trained model to predict Parkinson’s disease
- Displays the prediction through a web interface

This project is for educational and demonstration purposes only.

## Project structure

## Project structure

```text
parkinson-diagnosis-using-mri/
├── backend/
│   ├── app.py
│   ├── model/
│   ├── utils.py
│   ├── requirements.txt
│
├── frontend-react/
│   ├── public/
│   ├── src/
│   ├── package.json
│
├── .gitignore
└── README.md
```


## Tech stack

### Backend
- Python
- Flask or similar lightweight framework
- Machine learning libraries
- OpenCV or PIL for image processing

### Frontend
- React
- JavaScript
- HTML
- CSS

## How it works

1. A user uploads an MRI scan through the frontend.
2. The image is sent to the backend API.
3. The backend preprocesses the image.
4. The trained model runs inference on the MRI scan.
5. The result is returned and displayed to the user.

## Setup instructions

### Backend setup

Go to the backend directory:


Create a virtual environment:

Activate the environment:

Windows:
