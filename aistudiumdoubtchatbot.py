import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn


# Initialize Google Gemini Pro
genai.configure(api_key="AIzaSyCH5foXWnw35EWPs9PHOStSRwt6rb-bD5I")

# Create the model with the configuration
generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# FastAPI app initialization
app = FastAPI()

# Pydantic model for request payload
class SubjectRequest(BaseModel):
    ques: str

# Route to generate the course syllabus
@app.post("/doubt-chatbot")
async def generate_syllabus(request: SubjectRequest):
    # Get the subject from the request body
    subject = request.ques
    
    # Construct the prompt with the provided subject
    prompt = f"""
    You are a doubt chatbot for students and you have to resolve students doubts, The question is:{subject}
    """
    
    # Generate the response from the model
    response = model.generate_content(prompt)
    
    # Return the generated syllabus as a JSON response
    return {"answer": response.text}

# Run the FastAPI app
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)