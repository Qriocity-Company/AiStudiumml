import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_KEY"))

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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change to specific origins as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

class ContentRequest(BaseModel):
    content: str

class DomainRequest(BaseModel):
    domain: str

@app.post("/detect-domain")
async def detect_domain(request: ContentRequest):
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    
    prompt = f"""
    Analyze the following educational content and determine its subject domain (e.g., Mathematics, Physics, Biology, History, etc.).
    Also provide a brief explanation for why you classified it as that domain.
    Format your response as JSON with two fields: 'domain' and 'explanation'.
    
    Content: {request.content}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-courses")
async def recommend_courses(request: DomainRequest):
    if not request.domain.strip():
        raise HTTPException(status_code=400, detail="Domain cannot be empty")
    
    prompt = f"""
    Given the domain '{request.domain}', suggest 6 specific course names that a teacher could teach to students.
    Return the response in pure JSON format with an array field named courses.
    without backtick or 'json' in the response.
    Each course should be specialized and specific rather than generic.
    Make sure course names are practical and commonly taught in educational institutions.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# TESTING CODE
# import os
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import google.generativeai as genai
# from dotenv import load_dotenv
# import uvicorn

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_GEMINI_KEY"))

# generation_config = {
#     "temperature": 0.3,
#     "top_p": 0.95,
#     "top_k": 64,
#     "max_output_tokens": 8192,
#     "response_mime_type": "text/plain",
# }

# model = genai.GenerativeModel(
#     model_name="gemini-1.5-flash",
#     generation_config=generation_config,
# )

# app = FastAPI()

# class ContentRequest(BaseModel):
#     content: str

# @app.post("/detect-domain")
# async def detect_domain(request: ContentRequest):
#     if not request.content.strip():
#         raise HTTPException(status_code=400, detail="Content cannot be empty")
    
#     prompt = f"""
#     Analyze the following educational content and determine its subject domain (e.g., Mathematics, Physics, Biology, History, etc.).
#     Also provide a brief explanation for why you classified it as that domain.
#     Format your response as JSON with two fields: 'domain' and 'explanation'.
    
#     Content: {request.content}
#     """
    
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8000)