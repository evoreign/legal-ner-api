import asyncio
import json
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from openai import OpenAI, APIError
from pydantic import BaseModel, Field

# ---
# Configuration
# ---
# Use environment variables for configuration, with sensible defaults.

MAX_CONCURRENCY = int(10)
MODEL_NAME = "azure/gpt-5-mini"

# ---
# Initialization
load_dotenv()
# ---

# Initialize OpenAI client
client = OpenAI(
	api_key=os.getenv("OPENAI_API_KEY"),
	base_url="https://proxyllm.ximplify.id"
)
# Initialize FastAPI app
app = FastAPI(
    title="Legal Named Entity Recognition API",
    description="Extracts regulation names and actions from legal text using an LLM.",
    version="1.0.0",
)

# Thread pool for running synchronous OpenAI calls in async context
executor = ThreadPoolExecutor()

# Semaphore to limit concurrent requests to the LLM
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# ---
# Pydantic Models
# ---
class NerRequest(BaseModel):
    text: str = Field(..., example="bahwa berdasarkan ketentuan Pasal 10 ayat (3) Undang-Undang Nomor 15 Tahun 2006 tentang Badan Pemeriksa Keuangan,")

class ExternalNerEntry(BaseModel):
    base_regulation: str | None = Field(default=None, description="The primary regulation taking the action.", example="Undang-Undang Nomor 61 Tahun 2024")
    action: str = Field(..., description="The action being performed.", example="revise")
    target_regulation: str | None = Field(default=None, description="The regulation being acted upon, if any.", example="Undang-Undang Nomor 39 Tahun 2008")

class InternalNerEntry(BaseModel):
    type: str = Field(..., description="The type of the reference, e.g., 'Pasal' or 'ayat'.", example="Pasal")
    value: str = Field(..., description="The value of the reference, e.g., '6' or '(4)'.", example="6")
    children: list['InternalNerEntry'] | None = Field(default=None, description="A list of nested child references.")

# Rebuild the model to resolve the forward reference in the 'children' field.
InternalNerEntry.model_rebuild()

# ---
# System Prompt
# ---
EXTERNAL_NER_SYSTEM_PROMPT = """
You are a specialized legal assistant. Your task is to analyze a snippet of Indonesian legal text and identify ALL regulations mentioned and their associated actions. A single text can contain multiple regulations.

You must call the `extract_legal_ner` function with a list of all findings.

Example Input:
"...mencabut Peraturan Pemerintah Nomor 20 Tahun 2010 dan mengubah Undang-Undang Nomor 15 Tahun 2006..."

Example `extractions` argument for a complex amendment:
[
  {
    "base_regulation": "Undang-Undang Nomor 61 Tahun 2024",
    "action": "revise",
    "target_regulation": "Undang-Undang Nomor 39 Tahun 2008"
  }
]

If a regulation is simply mentioned, the `target_regulation` can be null.
"""

# ---
# OpenAI Tool Definition
# ---
EXTERNAL_NER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_legal_ner",
            "description": "Extracts all regulations and their corresponding actions from a legal text snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "extractions": {
                        "type": "array",
                        "description": "A list of all regulation-action pairs found in the text.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "base_regulation": {
                                    "type": "string",
                                    "description": "The primary regulation taking the action (e.g., an amending law)."
                                },
                                "action": {
                                    "type": "string",
                                    "enum": ["mention", "revise", "replace", "revoke", "suspend", "clarify", "implement"]
                                },
                                "target_regulation": {
                                    "type": "string",
                                    "description": "The regulation being acted upon (e.g., the law being amended)."
                                }
                            },
                            "required": ["action"]
                        }
                    }
                },
                "required": ["extractions"]
            },
        },
    }
]

INTERNAL_NER_SYSTEM_PROMPT = """
You are a specialized legal assistant. Your task is to analyze a snippet of Indonesian legal text and identify ALL structural internal references.

Focus ONLY on keywords that refer to parts of a legal document, such as:
- Pasal (article)
- ayat (clause/paragraph)

You must IGNORE other specific reference types like 'huruf' or 'angka'.

A reference should only be extracted if it is explicitly preceded by one of the keywords (e.g., "pada ayat (1)"). Do not extract numbers that simply start a sentence (e.g., "(2) Dalam hal...").

You must IGNORE simple numerical quantities or counts.

Example of what to extract (nested structure):
- Input: "...sebagaimana dimaksud dalam Pasal 6 ayat (4) untuk menilai..."
- Output:
[
  {
    "type": "Pasal",
    "value": "6",
    "children": [
      {
        "type": "ayat",
        "value": "(4)",
        "children": null
      }
    ]
  }
]

Example of what NOT to extract:
- Input: "...terdiri atas jabatan fungsional dan/atau paling banyak 3 (tiga) subbagian."
- Output: [] (The model should return an empty list because '3 (tiga)' is a quantity, not a structural reference).
"""

INTERNAL_NER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_internal_references",
            "description": "Extracts all internal references (e.g., Pasal, ayat) from a legal text snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "references": {
                        "type": "array",
                        "description": "A list of all internal references found.",
                        "items": {
                            "$ref": "#/definitions/reference_item"
                        }
                    }
                },
                "required": ["references"]
            },
            "definitions": {
                "reference_item": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "The type of the reference.",
                            "enum": ["Pasal", "ayat"]
                        },
                        "value": {
                            "type": "string",
                            "description": "The value or number of the reference, e.g., '17' or '(4)'."
                        },
                        "children": {
                            "type": ["array", "null"],
                            "items": {
                                "$ref": "#/definitions/reference_item"
                            },
                            "description": "A list of nested child references."
                        }
                    },
                    "required": ["type", "value"]
                }
            }
        },
    }
]

# ---
# Helper Functions
# ---
def run_external_ner_completion(text: str) -> list[ExternalNerEntry]:
    """Runs the synchronous OpenAI API call using tool use."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": EXTERNAL_NER_SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            tools=EXTERNAL_NER_TOOLS,
            tool_choice={"type": "function", "function": {"name": "extract_legal_ner"}}
        )

        message = completion.choices[0].message
        if not message.tool_calls:
            raise HTTPException(status_code=502, detail="Model did not call the required function.")

        # The model returns arguments as a JSON string, which we parse
        tool_arguments = message.tool_calls[0].function.arguments
        response_data = json.loads(tool_arguments)
        # The response is a list of extractions
        return [ExternalNerEntry(**item) for item in response_data.get("extractions", [])]
    except APIError as e:
        # Handle specific OpenAI API errors (e.g., rate limits, auth)
        raise HTTPException(status_code=502, detail=f"OpenAI API Error: {e.message}")
    except (json.JSONDecodeError, TypeError):
        # Handle cases where the model's output is not valid JSON
        raise HTTPException(status_code=502, detail="Failed to parse valid JSON from model response.")

def run_internal_ner_completion(text: str) -> list[InternalNerEntry]:
    """Runs the synchronous OpenAI API call for internal references."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": INTERNAL_NER_SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            tools=INTERNAL_NER_TOOLS,
            tool_choice={"type": "function", "function": {"name": "extract_internal_references"}}
        )

        message = completion.choices[0].message
        if not message.tool_calls:
            # It's okay if no references are found, return an empty list.
            return []

        tool_arguments = message.tool_calls[0].function.arguments
        response_data = json.loads(tool_arguments)
        return [InternalNerEntry(**item) for item in response_data.get("references", [])]
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API Error: {e.message}")
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=502, detail="Failed to parse valid JSON from model response.")

# ---
# API Endpoint
# ---
@app.get("/health")
def health_check():
    """
    Health check endpoint for Docker.
    """
    return {"status": "ok"}


@app.post("/ner_out", response_model=list[ExternalNerEntry])
async def external_named_entity_recognition(request: NerRequest):
    """
    Accepts a string of legal text and returns the identified regulation and action.
    This endpoint is concurrent and will handle multiple requests efficiently.
    """
    async with semaphore:
        loop = asyncio.get_running_loop()
        try:
            # Offload the synchronous, blocking OpenAI call to the thread pool
            response = await loop.run_in_executor(
                executor,
                run_external_ner_completion,
                request.text
            )
            return response
        except HTTPException as e:
            # Re-raise HTTPExceptions to be handled by FastAPI
            raise e
        except Exception as e:
            # Catch any other unexpected errors
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.post("/ner_in", response_model=list[InternalNerEntry])
async def internal_named_entity_recognition(request: NerRequest):
    """
    Accepts legal text and returns identified internal references (e.g., Pasal, ayat).
    """
    async with semaphore:
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                executor,
                run_internal_ner_completion,
                request.text
            )
            return response
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# ---
# Uvicorn Entrypoint
# ---
if __name__ == "__main__":
    # This allows running the app directly with `python NER/api.py`
    # For development, it's better to use: `uvicorn NER.api:app --reload`
    uvicorn.run(app, host="0.0.0.0", port=8000)
