import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
import google.generativeai as genai
from dotenv import load_dotenv
import json
import uvicorn
import re

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_KEY"))

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class CourseRequest(BaseModel):
    subject: str = Field(..., description="The subject of the course")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level of the course")
    focus_area: str = Field(..., description="Specific area to focus on within the subject")
    units: int = Field(..., ge=1, le=10, description="Number of units desired")

generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

app = FastAPI()

@app.post("/generate-course-structure")
async def generate_course_structure(request: CourseRequest):
    """Generate basic course structure with unit titles"""
    try:
        structure_prompt = f"""
        Generate a comprehensive course structure for {request.subject} with exactly {request.units} units.
        Focus area: {request.focus_area}
        Difficulty: {request.difficulty}

        Return ONLY unit titles in this JSON format:
        {{
            "courseTitle": "",
            "difficultyLevel": "",
            "description": "",
            "prerequisites": ["prerequisite 1", "prerequisite 2"],
            "learningOutcomes": ["outcome 1", "outcome 2"],
            "units": [
                {{
                    "unitTitle": "",
                    "unitDescription": ""
                }}
            ],
            "overview": "",
            "assessmentMethods": ["method 1", "method 2"]
        }}
        """
        
        structure_response = model.generate_content(structure_prompt)
        cleaned_json = re.sub(r"^```json|```$", "", structure_response.text, flags=re.MULTILINE).strip()
        print(f"Course structure response: {cleaned_json}")
        
        course_structure = json.loads(cleaned_json)
        return course_structure
        
    except Exception as e:
        print(f"Error in generate_course_structure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)