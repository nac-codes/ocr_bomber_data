import os
import json
import base64
import time
from openai import OpenAI
from PIL import Image
import io

# Initialize the OpenAI client
client = OpenAI()

# Directory containing images
image_dir = '/Users/chim/Working/Thesis/Attack_Images/Sandbox/IMG_0272_output'
batch_input_file = 'batch_input_4.jsonl'
batch_output_file = 'batch_output_4.jsonl'

def optimize_image(image_path, max_size=(800, 800), quality=85):
    with Image.open(image_path) as img:
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize if necessary
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size)
        
        # Save as JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def extract_column_name(filename, prompt_keys):
    for key in prompt_keys:
        if key.lower().replace("_", "") in filename.lower().replace("_", ""):
            return key
    return None

# Prepare batch input file
with open(batch_input_file, 'w') as outfile:
    prompts = {
        "Day": "Transcribe the Day. Expect a single day (1-31) or a range (e.g., 15-16) for overnight operations.",
        "Month": "Transcribe the Month. Expect a number from 1 to 12.",
        "Year": "Transcribe the Year. Expect a single digit (0-4) corresponding to 1940-1944.",
        "Time_of_Attack": "Transcribe the Time of Attack. Expect 24-hour time format (e.g., 1434).",
        "Air_Force": "Transcribe the Air Force. Expect 8, 9, 12, 15 for US Air Forces, or R for Royal Air Force.",
        "Group_Squadron_Number": "Transcribe the Group/Squadron Number. Expect an ID with numbers followed by letters (e.g., 305AG, 305PG, 305CG) or just G or S.",
        "Number_of_Aircraft_Bombing": "Transcribe the Number of Aircraft Bombing. Expect a number ranging from 1 to hundreds.",
        "Altitude_of_Release": "Transcribe the Altitude of Release. Expect a number typically in the 200s.",
        "Sighting": "Transcribe the Sighting. Expect a number from 1 to 10.",
        "Visibility_of_Target": "Transcribe the Visibility of Target. Expect a single letter, typically N, G, or P.",
        "Target_Priority": "Transcribe the Target Priority. Expect a number typically ranging from 1 to 4.",
        "HE_Bombs_Number": "Transcribe the Number of HE Bombs. Expect a number from 1 to hundreds.",
        "HE_Bombs_Size": "Transcribe the Size of HE Bombs. Expect a number from 1 to 5.",
        "HE_Bombs_Tons": "Transcribe the Tons of HE Bombs. Expect a number from 1 to hundreds.",
        "Fuzing_Nose": "Transcribe the Nose Fuzing. Expect a number from 1 to 5.",
        "Fuzing_Tail": "Transcribe the Tail Fuzing. Expect a number from 1 to 5.",
        "Incendiary_Bombs_Number": "Transcribe the Number of Incendiary Bombs. Expect a number from 1 to hundreds.",
        "Incendiary_Bombs_Size": "Transcribe the Size of Incendiary Bombs. Expect a number.",
        "Incendiary_Bombs_Tons": "Transcribe the Tons of Incendiary Bombs. Expect a number from 1 to hundreds.",
        "Fragmentation_Bombs_Number": "Transcribe the Number of Fragmentation Bombs. Expect a number from 1 to hundreds.",
        "Fragmentation_Bombs_Size": "Transcribe the Size of Fragmentation Bombs. Expect a number.",
        "Fragmentation_Bombs_Tons": "Transcribe the Tons of Fragmentation Bombs. Expect a number from 1 to hundreds.",
        "Total_Tons": "Transcribe the Total Tons of Bombs. Expect a number from 1 to hundreds.",
        "Target_Location": "Transcribe the Target Location. Expect a town name (e.g., Berlin) and possibly a Target Name.",
        "Target_Name": "Transcribe the Target Name. Expect the name of an industrial target (e.g., Marshalling yard or a specific factory).",
        "Latitude": "Transcribe the Latitude. Expect a value like 4824N, representing a location in Europe.",
        "Longitude": "Transcribe the Longitude. Expect a value like 1000E, representing a location in Europe.",
        "Target_Code": "Transcribe the Target Code. Expect a series of digits, typically a set of 5 digits followed by a set of 3 digits."
    }

    prompt_keys = prompts.keys()
    
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png')):
            file_path = os.path.join(image_dir, filename)
            optimized_image = optimize_image(file_path)
            
            column_name = extract_column_name(filename, prompt_keys)
            if column_name:
                custom_prompt = prompts.get(column_name, "Transcribe the content in this image accurately.")
                prompt = f"{custom_prompt} Return only the transcription."
            else:
                prompt = "Transcribe the content in this image accurately. Only return the transcription."
            
            request = {
                "custom_id": filename,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a historical document transcriber for the US Strategic Bombing Survey. Your task is to transcribe the following image content accurately, focusing on the specific data requested."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{optimized_image}",
                                        "detail": "low"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "max_tokens": 1000
                }
            }
            outfile.write(json.dumps(request) + '\n')

input("Press Enter to continue...")

# Upload the batch input file
uploaded_file = client.files.create(
    file=open(batch_input_file, "rb"),
    purpose="batch"
)

print("Waiting for file to be uploaded...")
while not uploaded_file.id:
    print(".", end="", flush=True)
    time.sleep(1)
print(f"\nFile uploaded successfully. File ID: {uploaded_file.id}")

# Create the batch job
batch_job = client.batches.create(
    input_file_id=uploaded_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "Batch job for transcribing historical documents"
    }
)

print(f"Batch job created. Job ID: {batch_job.id}")

# Check the status of the batch job
while True:
    batch_check = client.batches.retrieve(batch_job.id)
    print(f"Batch Status: {batch_check.status}")
    if batch_check.status in ['completed', 'failed']:
        break
    time.sleep(60)  # Check every minute

# Process results
if batch_check.status == 'completed':
    output_file_id = batch_check.output_file_id
    file_response = client.files.content(output_file_id)
    
    with open(batch_output_file, 'w') as outfile:
        outfile.write(file_response.text)
    
    print(f"Batch processing complete. Results saved to '{batch_output_file}'.")
else:
    print("Batch processing failed.")