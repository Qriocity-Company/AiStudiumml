import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
from typing import List
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

class MCQRequest(BaseModel):
    unit_title: str = Field(..., description="Title of the unit")
    subject: str = Field(..., description="The subject of the course")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level of the course")
    focus_area: str = Field(..., description="Specific area to focus on within the subject")
    learning_objectives: List[str] = Field(..., description="Learning objectives for the unit")
    topics_covered: List[str] = Field(..., description="Topics covered in the unit")

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

@app.post("/generate-mcqs")
async def generate_mcqs(request: MCQRequest):
    """Generate MCQs for a unit"""
    try:
        mcq_prompt = f"""
        Generate Multiple Choice Questions (MCQs) for the unit "{request.unit_title}" in {request.subject}.
        Topics to cover: {', '.join(request.topics_covered)}
        Learning objectives: {', '.join(request.learning_objectives)}
        Difficulty level: {request.difficulty}
        Focus area: {request.focus_area}

        Return the response in this JSON format:
        {{
            "unitTitle": "{request.unit_title}",
            "unitAssessment": [
                {{
                    "topic": "Topic Name",
                    "questions": [
                        {{
                            "questionId": "unique_id",
                            "question": "Question text",
                            "options": [
                                "Option A",
                                "Option B",
                                "Option C",
                                "Option D"
                            ],
                            "correctAnswer": "Correct option",
                            "explanation": "Explanation of the correct answer"
                        }}
                    ]
                }}
            ],
            "reviewMaterial": [
                {{
                    "topic": "Topic Name",
                    "keyPoints": ["key point 1", "key point 2"]
                }}
            ]
        }}

        Generate at least 3 MCQs per topic, ensuring they match the difficulty level.
        """
        
        response = model.generate_content(mcq_prompt)
        cleaned_json = re.sub(r"^```json|```$", "", response.text, flags=re.MULTILINE).strip()
        print(f"MCQ generation response for {request.unit_title}: {cleaned_json}")
        
        mcq_data = json.loads(cleaned_json)
        return mcq_data
        
    except Exception as e:
        print(f"Error generating MCQs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate MCQs: {str(e)}"
        )

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.1", port=8001)