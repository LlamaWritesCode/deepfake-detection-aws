## 📚 Contents


- [🔍 Overview](#-overview)
    * [Project Link](#try-it-out--aws-deepfake-detector)
    * [Demo Video](#demo-video)
- [💡 What It Does](#-what-it-does)
- [🏗️ Architecture](#-architecture)
- [🧠 Deepfake Detection Flow](#-deepfake-detection-flow)
- [🛠️ Technologies Used](#-technologies-used)
- [🧬 Lambda Function](#-lambda-function)
- [📊 Streamlit Dashboard](#-streamlit-dashboard)
- [🔐 IAM & Permissions](#-iam--permissions)
- [📁 Project Structure](#-project-structure)
- [🧪 Example Input & Output](#-example-input--output)
- [🌍 How This Solves a Real-World Problem](#-how-this-solves-a-real-world-problem)
- [📦 Submission Requirements](#-submission-requirements)
- [💻 How to Run Locally](#-how-to-run-locally)
- [🚀 Future Improvements](#-future-improvements)



🔍 **Overview**
-----------

This serverless web application helps researchers and developers detect deepfake images and collect annotated data for training future detection models. The system leverages the power of **AWS Lambda**, **Amazon SageMaker**, **Hugging Face models**, **S3**, and **DynamoDB**, with an intuitive **Streamlit** dashboard interface.

#### Try it out 👉 [AWS Deepfake Detector](https://deepfake-detection-aws-snxamqcshc9utzlvqyr2ff.streamlit.app/)
#### Demo Video 👉 [PASTE THE YOUTUBE LINK HERE] 


## 💡 **What It Does**

*   Accepts a publicly accessible image URL through a web UI.
    
*   Invokes a **Lambda function** (triggered via **API Gateway**) that:
    
    *   Downloads the image.
        
    *   Uploads it to **Amazon S3**.
        
    *   Sends the image to an **Amazon SageMaker** endpoint running a **Hugging Face deepfake detection model**.
        
    *   Parses the model response and stores structured metadata in **DynamoDB**.
        
*   Displays the analysis results on a **Streamlit** dashboard.
    
*   Provides researchers the ability to:
    
    *   Visualize images stored in S3.
        
    *   Download detection data from DynamoDB in CSV format.
        
    *   Delete individual files from S3 for data curation.
 

## 🏗️ **Architecture**

 
                                                      [User Browser]

                                                            ↓

                                                  [Streamlit Frontend]

                                                            ↓

                                                  [API Gateway Trigger]

                                                            ↓

                                                      [AWS Lambda]

                                                            ↓

                                    ┌─────────────────────────────────────────────┐

                                    | Core Lambda Responsibilities:               |

                                    | • Download image from URL                   |

                                    | • Upload to Amazon S3                       |

                                    | • Invoke Hugging Face model via SageMaker   |

                                    | • Store metadata in DynamoDB                |

                                    └─────────────────────────────────────────────┘

                                            ↓                            ↓

                             [Amazon S3] (stores images)    [Amazon DynamoDB] (stores annotations)



## 🧠 **Deepfake Detection Flow**

1.  **User Input**: The user provides an image URL via the Streamlit UI.
    
2.  **API Gateway** triggers the **Lambda function**.
    
3.  **Lambda**:
    
    *   Downloads the image.
        
    *   Uploads it to **S3** under uploads/.
        
    *   Sends the image bytes to a **SageMaker** real-time inference endpoint (deepfake-detector-hf-v1) which uses a Hugging Face model prithivMLmods/deepfake-detector-model-v1.
        
    *   Receives a prediction (real or fake) and confidence score.
        
    *   Stores metadata including prediction, confidence, image S3 path, and timestamp in **DynamoDB** under the table DeepfakeDetections.

  

## 🛠️ Technologies Used

| Component        | Service or Tool                                |
|------------------|------------------------------------------------|
| Serverless Logic | AWS Lambda                                     |
| API Trigger      | API Gateway                                    |
| Model Inference  | Amazon SageMaker + Hugging Face Model Hub      |
| Data Storage     | Amazon DynamoDB (structured metadata)          |
| Image Storage    | Amazon S3 (raw uploaded images)                |
| Frontend UI      | Streamlit                                      |
| Data Format      | JSON, CSV                                       |
| Deepfake Detection Model  | HuggingFace: [deepfake-detector-model-v1](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1)    |




## 🧬 Lambda Function

The Lambda function performs the core processing:

- **Trigger**: HTTP POST request with an image URL (via API Gateway).
- **Steps**:
  1. Download image from provided URL.
  2. Upload to S3: `deepfake-uploads/uploads/{uuid}.jpg`
  3. Invoke SageMaker endpoint with image bytes.
  4. Parse model response and write metadata to DynamoDB.

**Main AWS Lambda Code**: 
```
import boto3
import requests
import uuid
import json
import time
import os
from decimal import Decimal


s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sm_runtime = boto3.client('sagemaker-runtime')

TABLE_NAME = 'DeepfakeDetections'
BUCKET_NAME = 'deepfake-uploads'
ENDPOINT_NAME = 'deepfake-detector-hf-v1' 

table = dynamodb.Table(TABLE_NAME)

def lambda_handler(event, context):
    try:
        body = json.loads(event['body']) if 'body' in event else event
    except Exception as e:
        return _response(400, {"error": f"Invalid JSON input: {str(e)}"})

    file_url = body.get('file_url')
    if not file_url:
        return _response(400, {"error": "Missing 'file_url'"})

    try:
        # Download image bytes
        response = requests.get(file_url)
        if response.status_code != 200:
            return _response(400, {"error": "Could not download image"})
        file_bytes = response.content
    except Exception as e:
        return _response(400, {"error": f"Failed to download image: {str(e)}"})

    file_id = str(uuid.uuid4())
    s3_key = f"uploads/{file_id}.jpg"

    # Upload image to S3
    s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=file_bytes)

    try:
        # Invoke SageMaker endpoint
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
             ContentType='image/png',
            Body=file_bytes
        )
        result = response['Body'].read().decode('utf-8')
        prediction = json.loads(result)
        label = prediction[0]['label'].lower()
        confidence = float(prediction[0]['score'])
    except Exception as e:
        return _response(500, {"error": f"SageMaker invoke failed: {str(e)}"})

    # Save detection result in DynamoDB
    item = {
        'uuid': file_id,
        's3_key': s3_key,
        'label': label,
        'confidence': Decimal(str(confidence)),
        'timestamp': int(time.time()),
        'source_type': 'url',
        'model_used': 'HuggingFace-prithivMLmods',
        'model_version': 'v1',
        'recheck_status': 'pending'
    }
    table.put_item(Item=item)

    return _response(200, {
        'verdict': label,
        'confidence': round(confidence, 3),
        'uuid': file_id,
        'model_used': 'HuggingFace-prithivMLmods',
        'model_version': 'v1'
    })


def _response(status_code, body_dict):
    return {
        'statusCode': status_code,
        'body': json.dumps(body_dict),
        'headers': {
            'Content-Type': 'application/json',
        }
    }


```



## 📊 Streamlit Dashboard

The dashboard provides:

- **Image Detection Upload Form**
- **Results Table (verdict + confidence)**
- **DynamoDB CSV Export Button**
- **S3 Table (filename, size, date)**
- **Delete buttons to remove images from S3**

Researchers can use the exported CSV to train and benchmark future models.


## 🔐 IAM & Permissions

The following AWS IAM permissions are required:

- **Lambda Function Role**:
  - `s3:PutObject`, `s3:GetObject`, `s3:DeleteObject`
  - `dynamodb:PutItem`, `dynamodb:Scan`
  - `sagemaker:InvokeEndpoint`

- **Streamlit Boto3 Access**:
  - Must have local AWS credentials via `aws configure` or instance role


## 📁 **Project Structure**

├── lambda\_function.py # AWS Lambda source 

├── streamlit\_dashboard.py # Streamlit app for researchers

├── README.md # You're reading this!

├── requirements.txt # Python dependencies


## 🧪 Example Input & Output

### Input:
``` 
  {  
    "file_url": "https://deepfake-uploads.s3.us-east-2.amazonaws.com/test-images/l.png"  
  }
```


### Output:
```
{
  "verdict": "fake",
  "confidence": 0.988,
  "uuid": "1234abcd-5678-efgh-9012-3456ijklmnop",
  "model_used": "HuggingFace-prithivMLmods",
  "model_version": "v1"
}
```

## 🌍 **How This Solves a Real-World Problem**

With the rise of misinformation and generative AI, journalists, platforms, and researchers need scalable tools to detect synthetic content and build better defenses.

This application:

*   Provides a **fully automated, serverless solution** to classify deepfakes.
    
*   Offers a way to **collect and store labeled media** for ongoing research and AI training.
    
*   Scales instantly with **no need for provisioning infrastructure**.
    

## 📦 **Submission Requirements**

*   **Lambda Core**: Image detection and classification runs entirely inside a Lambda function.
    
*   **Trigger**: Lambda is invoked via API Gateway when a user submits an image URL.
    
*   **AWS Services Used**:
    
    *   Lambda (core compute)
        
    *   API Gateway (trigger)
        
    *   SageMaker (model inference)
        
    *   S3 (image storage)
        
    *   DynamoDB (metadata storage)
        
*   **Frontend**: Streamlit app for interacting with the system


## 💻 **How to Run Locally**
```
pip install -r requirements.txt
streamlit run deepfake_dashboard.py
```

Ensure you have AWS credentials configured:

```
aws configure
```


## 🚀 **Future Improvements**
1. Add support for uploading local files
2. Integrate with AWS Bedrock for multi-modal analysis
3. Add re-verification flow for flagged samples
4. Publish an open dataset with labels for community use
