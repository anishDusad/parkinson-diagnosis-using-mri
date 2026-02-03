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
- FastAPI – High-performance backend API for MRI upload, inference, and streaming responses
- PyTorch – Deep learning framework for CNN-based Parkinson’s disease classification
- ANTs / ANTsPy – Medical-grade MRI preprocessing (skull stripping & MNI registration)
- DICOM → NIfTI Pipeline – Clinical MRI handling using dicom2nifti and nibabel
- Multi-View CNN Ensemble – Axial, coronal, and sagittal slice-based inference with weighted fusion
- Real-Time Streaming Logs – Progress updates during long MRI preprocessing and inference

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
```
cd backend
```

Create a virtual environment:
```
python -m venv venv
```

Activate the virtual environment:

Windows
```
venv\Scripts\activate
```
macOS / Linux
```
source venv/bin/activate
```

Install Python dependencies:
```
pip install -r requirements.txt
```

Run the backend server:
```
python app.py
```
### Frontend setup

Go to the frontend directory:
```
cd frontend-react
````

Install dependencies:
```
npm install
```

Start the development server:
```
npm start
```
