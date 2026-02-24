import requests
import pprint
import json
import re
from config import *

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"


# ======================================================
# OLLAMA CALL
# ======================================================

def call_ollama(prompt: str) -> str:
    """
    Calls Ollama API and returns raw model text response.
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "top_p": 1,
                "repeat_penalty": 1
            }
        }
    )

    response.raise_for_status()  # safer

    result = response.json()
    return result.get("response", "")


# ======================================================
# JSON EXTRACTION
# ======================================================

def extract_json(text: str) -> dict:
    """
    Extract first valid JSON object from model output.
    """

    match = re.search(r'\{.*\}', text, re.DOTALL)

    if not match:
        raise ValueError("No JSON found in model response")

    json_text = match.group(0)

    # Fix common LLM mistakes
    json_text = json_text.replace("None", "null")
    json_text = json_text.replace("True", "true")
    json_text = json_text.replace("False", "false")

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        print("FAILED JSON:")
        print(json_text)
        raise


# ======================================================
# EQUATION EXTRACTION
# ======================================================

# def get_equation_dict(frd_document: str) -> dict:

#     full_prompt = f"""
# You are a Python equation extraction engine.

# Strict rules:

# 1. Identify ALL equations.
# 2. Identify ALL variables.
# 3. Convert conditions into valid Python expressions.
# 4. Combine nested IFs with AND.
# 5. Flatten conditions.
# 6. Use null when no condition.
# 7. Use == for equality.
# 8. Replace ^ with **.
# 9. Return ONLY valid JSON.

# Format:
# {equation_example}

# Document:
# {frd_document}
# """

#     response = call_ollama(full_prompt)
#     return extract_json(response)


def get_equation_dict(frd_document: str) -> dict:

    full_prompt = f"""
You are a deterministic FRD → structured JSON compiler.

Your job is to extract equations precisely.

STRICT RULES:

1. Extract ALL assignments.
2. Split every assignment into:
   - lhs
   - rhs
3. Classify type:

   constant:
     RHS contains only numbers and operators
     examples: 25, 1/0.000125, 3.14*2

   derived:
     RHS contains variables

4. CONSTANTS:
   - MUST always have condition = null
   - NEVER inherit IF conditions

5. DERIVED:
   - inherit all IF conditions
   - combine nested IFs using AND

6. Use valid Python expressions.
7. Use null instead of None.
8. Return ONLY JSON.

FORMAT:

{{
  "variables": [],
  "equations": [
    {{
      "lhs": "variable_name",
      "rhs": "expression_string",
      "type": "constant | derived",
      "condition": null or "boolean expression"
    }}
  ]
}}

Document:
{frd_document}
"""

    response = call_ollama(full_prompt)
    return extract_json(response)


# ======================================================
# RANGE EXTRACTION
# ======================================================

def get_ranges_dict(frd_document: str) -> dict:

    full_prompt = f"""
You are a python range extractor.

Strict rules:

1. Return ONLY valid JSON.
2. Use square brackets [].
3. NEVER use tuples.
4. Format: "variable": [start, end, step]

Format example:
{range_example}

Document:
{frd_document}
"""

    response = call_ollama(full_prompt)
    return extract_json(response)


# ======================================================
# MAIN WRAPPER
# ======================================================

def get_few_shot_prompting_response(FRD: str) -> dict:
    """
    Returns:
    {
        variables,
        equations,
        ranges
    }
    """

    few_shot_response = get_equation_dict(FRD)

    if "_RANGE_END_TAG_" in FRD:
        part1 = FRD.split("_RANGE_END_TAG_")[0]
        ranges_response = get_ranges_dict(part1)

        few_shot_response["ranges"] = ranges_response.get("ranges", {})
    else:
        few_shot_response["ranges"] = {}

    pprint.pprint(few_shot_response)

    return few_shot_response