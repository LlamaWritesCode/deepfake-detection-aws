import streamlit as st
import boto3
import pandas as pd
import requests
import json
from datetime import datetime
from PIL import Image
from io import BytesIO

# AWS resources
import os
dynamodb = boto3.resource('dynamodb',
    region_name=st.secrets["aws"]["region_name"],
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
)

s3 = boto3.client('s3',
    region_name=st.secrets["aws"]["region_name"],
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
)
table = dynamodb.Table('DeepfakeDetections')
S3_BUCKET = 'deepfake-uploads'

st.set_page_config(page_title="🧠 Deepfake Detection Research Dashboard", layout="wide")

st.title("🧠 Deepfake Detection Research Dashboard")
st.markdown("Upload an image URL, view results, and download training data from DynamoDB.")

# -- Upload section
st.header("📤 Upload an Image URL for Detection")

with st.form("upload_form"):
    file_url = st.text_input("Enter a publicly accessible image URL:")
    submit = st.form_submit_button("Detect & Store")

# Store results here for response table
results_df = pd.DataFrame()

if submit:
    response = requests.post(
        "https://heql3b8ph0.execute-api.us-east-2.amazonaws.com/detect",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"file_url": file_url})
    )

    if response.status_code == 200:
        result = response.json()
        st.success(f"✅ Detected: {result['verdict']} (confidence: {result['confidence']})")
        # Show response in table
        results_df = pd.DataFrame([result])
        st.dataframe(results_df)
    else:
        st.error(f"❌ Detection failed: {response.text}")

# -- S3 viewer
def list_s3_objects():
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="uploads/")
    items = response.get('Contents', [])
    data = []
    for obj in items:
        data.append({
            "Key": obj['Key'],
            "Size (KB)": round(obj['Size'] / 1024, 2),
            "Last Modified": obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
        })
    return pd.DataFrame(data)

df_s3 = list_s3_objects()

if df_s3.empty:
    st.warning("No objects found in S3 bucket under 'uploads/' prefix.")
else:
    st.header("🖼️ S3 Stored Images")

    for idx, row in df_s3.iterrows():
        cols = st.columns([6, 1, 2, 1])
        cols[0].write(row['Key'])
        cols[1].write(row['Size (KB)'])
        cols[2].write(row['Last Modified'])
        
        if cols[3].button("Delete", key=f"delete_{idx}"):
            try:
                s3.delete_object(Bucket=S3_BUCKET, Key=row['Key'])
                st.success(f"Deleted {row['Key']}")
                try:
                    st.experimental_rerun()
                except AttributeError:
                    st.warning("Please refresh the page to see updated changes.")
            except Exception as e:
                st.error(f"Failed to delete {row['Key']}: {str(e)}")




# -- DynamoDB CSV Downloader

def load_dynamodb_data():
    response = table.scan()
    items = response.get('Items', [])
    for item in items:
        item['timestamp'] = datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')
    return pd.DataFrame(items)


df = load_dynamodb_data()

if not df.empty:
    st.header("📥 Download Annotated Deepfake Data")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Download CSV from DynamoDB",
        data=csv_data,
        file_name="deepfake_detections.csv",
        mime="text/csv"
    )
else:
    st.warning("No data found in DynamoDB.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using AWS Lambda, AWS API Gateway, S3, DynamoDB, Streamlit and Hugging Face.")