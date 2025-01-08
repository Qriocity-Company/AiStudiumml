import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
import google.generativeai as genai
from dotenv import load_dotenv
import json
import uvicorn
import re
from typing import List

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_KEY"))

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class UnitRequest(BaseModel):
    unit_title: str = Field(..., description="Title of the unit")
    subject: str = Field(..., description="The subject of the course")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level of the course")
    focus_area: str = Field(..., description="Specific area to focus on within the subject")

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

async def generate_unit_content(unit_data: dict, subject: str, difficulty: str, focus_area: str):
    """Generate detailed content for each topic in a unit"""
    content_prompt = f"""
    Generate detailed educational content for the unit "{unit_data['unitTitle']}" in {subject}.
    Topics to cover: {', '.join(unit_data['topicsCovered'])}
    Learning objectives: {', '.join(unit_data['learningObjectives'])}
    Difficulty level: {difficulty}
    Focus area: {focus_area}

    Return the response in this JSON format:
    {{
        "topicContents": [
            {{
                "topic": "Topic Name",
                "content": "Detailed explanation and educational content",
                "examples": ["example 1", "example 2"],
                "exercises": ["exercise 1", "exercise 2"]
            }}
        ],
        "practicalProject": {{
            "title": "Project title",
            "description": "Project description",
            "steps": ["step 1", "step 2"],
            "deliverables": ["deliverable 1", "deliverable 2"]
        }},
        "additionalResources": [
            {{
                "type": "reading/video/tool",
                "title": "Resource title",
                "description": "Brief description"
            }}
        ]
    }}

    Ensure content is practical and matches the specified difficulty level.
    """
    
    try:
        response = model.generate_content(content_prompt)
        cleaned_json = re.sub(r"^```json|```$", "", response.text, flags=re.MULTILINE).strip()
        print(f"Content generation response for {unit_data['unitTitle']}: {cleaned_json}")
        
        content_data = json.loads(cleaned_json)
        return content_data
    except Exception as e:
        print(f"Error generating content for unit {unit_data['unitTitle']}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate content for unit {unit_data['unitTitle']}: {str(e)}"
        )

@app.post("/generate-unit-details")
async def generate_unit_details(request: UnitRequest):
    """Generate complete unit details including structure and content"""
    try:
        # First generate unit structure
        unit_prompt = f"""
        Generate a detailed unit structure for "{request.unit_title}" in {request.subject} course.
        Difficulty level: {request.difficulty}
        Focus area: {request.focus_area}

        Return the response in this JSON format:
        {{
            "unitTitle": "{request.unit_title}",
            "learningObjectives": ["detailed objective 1", "detailed objective 2", ...],
            "topicsCovered": ["detailed topic 1", "detailed topic 2", ...],
            "practicalAssignments": ["assignment 1", "assignment 2", ...],
            "resources": ["resource 1", "resource 2", ..."],
            "estimatedDuration": "X weeks"
        }}

        Ensure content matches the difficulty level and focuses on practical applications.
        """
        
        structure_response = model.generate_content(unit_prompt)
        cleaned_json = re.sub(r"^```json|```$", "", structure_response.text, flags=re.MULTILINE).strip()
        print(f"Unit structure response for {request.unit_title}: {cleaned_json}")
        
        unit_data = json.loads(cleaned_json)
        
        # Generate detailed content
        detailed_content = await generate_unit_content(
            unit_data,
            request.subject,
            request.difficulty,
            request.focus_area
        )
        
        # Merge structure with content
        unit_data["detailedContent"] = detailed_content
        
        return unit_data
        
    except Exception as e:
        print(f"Error in generate_unit_details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate unit details for {request.unit_title}: {str(e)}"
        )

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.1", port=8001)