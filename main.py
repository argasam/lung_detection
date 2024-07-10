import httpx
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import io
import numpy as np
from PIL import Image
from model.model import preprocess_image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid


app = FastAPI()

ORTHANC_URL = "http://localhost:8042"  # Replace with your Orthanc server address
ORTHANC_USERNAME = "orthanc"
ORTHANC_PASSWORD = "orthanc"


@app.get("/")
async def root():
    return {"health_check": "OK", "model_version": "vx.x"}


@app.post("/prediction/")
def preprocess_dicom_to_jpg(dicom_data):
    # Read the DICOM file
    ds = pydicom.dcmread(io.BytesIO(dicom_data))

    # Convert to float to avoid overflow or underflow losses
    image = ds.pixel_array.astype(float)

    # Rescale slope and intercept
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        image = (image * ds.RescaleSlope) + ds.RescaleIntercept

    # Apply VOI LUT transformation
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        image = apply_voi_lut(image, ds)
    else:
        # If no window center/width, use min-max scaling
        image = (image - image.min()) / (image.max() - image.min())

    # Scale to 8-bit (0-255)
    image = (image * 255).astype(np.uint8)

    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Convert to RGB if it's not already
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    return pil_image


def create_structured_report(original_ds, class_label, confidence):
    # Create a new DICOM dataset for the Structured Report
    sr = Dataset()

    # Copy relevant patient and study information
    sr.PatientName = original_ds.PatientName
    sr.PatientID = original_ds.PatientID
    sr.StudyInstanceUID = original_ds.StudyInstanceUID
    sr.StudyID = original_ds.StudyID

    # Set SR-specific attributes
    sr.SOPClassUID = '1.2.840.10008.5.1.4.1.1.88.22'  # Basic Text SR
    sr.SOPInstanceUID = generate_uid()
    sr.SeriesInstanceUID = generate_uid()
    sr.SeriesNumber = 1001  # Arbitrary number for the new series
    sr.InstanceNumber = 1
    sr.ContentDate = original_ds.ContentDate
    sr.ContentTime = original_ds.ContentTime
    sr.Modality = 'SR'

    # Create the Content Sequence
    sr.ContentSequence = Sequence([
        Dataset(
            RelationshipType='CONTAINS',
            ValueType='CONTAINER',
            ConceptNameCodeSequence=Sequence([
                Dataset(
                    CodeValue='AI_RESULTS',
                    CodingSchemeDesignator='99MYAI',
                    CodeMeaning='AI Analysis Results'
                )
            ]),
            ContinuityOfContent='SEPARATE',
            ContentSequence=Sequence([
                Dataset(
                    RelationshipType='CONTAINS',
                    ValueType='TEXT',
                    ConceptNameCodeSequence=Sequence([
                        Dataset(
                            CodeValue='PREDICTION',
                            CodingSchemeDesignator='99MYAI',
                            CodeMeaning='AI Prediction'
                        )
                    ]),
                    TextValue=class_label
                ),
                Dataset(
                    RelationshipType='CONTAINS',
                    ValueType='NUM',
                    ConceptNameCodeSequence=Sequence([
                        Dataset(
                            CodeValue='CONFIDENCE',
                            CodingSchemeDesignator='99MYAI',
                            CodeMeaning='Confidence Score'
                        )
                    ]),
                    MeasuredValueSequence=Sequence([
                        Dataset(
                            NumericValue=str(confidence),
                            MeasurementUnitsCodeSequence=Sequence([
                                Dataset(
                                    CodeValue='1',
                                    CodingSchemeDesignator='UCUM',
                                    CodeMeaning='no units'
                                )
                            ])
                        )
                    ])
                )
            ])
        )
    ])

    return sr

async def predict_study(background_tasks: BackgroundTasks, study_id: str):
    background_tasks.add_task(process_study, study_id)
    return JSONResponse(content={"message": "Processing started", "study_id": study_id})

async def process_study(study_id: str):
    async with httpx.AsyncClient() as client:
        # Get the list of instances in the study
        response = await client.get(
            f"{ORTHANC_URL}/studies/{study_id}/instances",
            auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD)
        )
        instances = response.json()

        for instance in instances:
            # Get the DICOM file
            response = await client.get(
                f"{ORTHANC_URL}/instances/{instance}/file",
                auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD)
            )
            dicom_data = response.content

            # Preprocess DICOM to JPG
            img = preprocess_dicom_to_jpg(dicom_data)

            # Convert to bytes for prediction
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)

            # Predict
            prediction = preprocess_image(img_bytes)
            class_label = 'Pneumonia' if prediction > 0.5 else 'Normal'
            confidence = float(prediction) if class_label == 'Pneumonia' else float(1 - prediction)

            # Read the original DICOM dataset for SR creation
            ds = pydicom.dcmread(io.BytesIO(dicom_data))

            # Create Structured Report
            sr = create_structured_report(ds, class_label, confidence)

            # Save SR to bytes
            sr_bytes = io.BytesIO()
            sr.save_as(sr_bytes)
            sr_bytes.seek(0)

            # Send the SR to Orthanc
            response = await client.post(
                f"{ORTHANC_URL}/instances",
                content=sr_bytes.getvalue(),
                auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD)
            )

            if response.status_code != 200:
                print(f"Error sending SR to Orthanc: {response.text}")

async def store_results(study_id: str, results: list):
    # Implement storing results in your database
    pass
# async def predict(file: UploadFile = File(...)):
#     # Read the file as bytes
#     file_bytes = await file.read()
#     # Open the image
#     img = Image.open(BytesIO(file_bytes)).convert("RGB")
#     img_bytes = BytesIO()
#     img.save(img_bytes, format='JPEG')  # Save the PIL image to the BytesIO object
#     img_bytes.seek(0)
#     # Predict
#     prediction = preprocess_image(img_bytes)
#     # Determine the class label
#     class_label = 'Pneumonia' if prediction > 0.5 else 'Normal'
#
#     # Return the response
#     return JSONResponse(content={
#         "filename": file.filename,
#         "prediction": class_label,
#         "confidence": float(prediction) if class_label == 'Pneumonia' else float(1 - prediction)
#     })
