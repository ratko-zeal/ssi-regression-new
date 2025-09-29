"""
JSON Schemas for Responses API structured outputs.
"""
ANSWER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Answer",
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["scores_summary","explanation","mixed","focus_recommendations"]},
        "country": {"type": ["string","null"]},
        "as_of_date": {"type": ["string","null"]},
        "version": {"type": ["string","null"]},
        "final_score": {"type": ["number","null"]},
        "display_domain_scores": {"type": ["object","null"], "additionalProperties": {"type":"number"}},
        "indicator_scores": {"type": ["object","null"], "additionalProperties": {"type":"number"}},
        "focus_recommendations": {
            "type": ["array","null"],
            "items": {
                "type":"object",
                "properties": {
                    "indicator": {"type":"string"},
                    "tier": {"type":"string", "enum":["High","Medium","Low"]},
                    "rationale": {"type":"string"}
                },
                "required":["indicator","tier","rationale"]
            }
        },
        "citations": {"type": "array", "items": {"type":"string"}},
        "narrative": {"type": "string"},
        "errors": {
            "type": "array",
            "items": {
                "type":"object",
                "properties": {
                    "code": {"type":"string"},
                    "message": {"type":"string"},
                    "details": {"type":["object","null"]}
                },
                "required":["code","message"]
            }
        }
    },
    "required": ["kind","narrative"]
}

ERROR_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ErrorEnvelope",
    "type": "object",
    "properties": {
        "error": {
            "type":"object",
            "properties": {
                "code": {"type":"string"},
                "message": {"type":"string"},
                "details": {"type":["object","null"]}
            },
            "required":["code","message"]
        },
        "session_id": {"type":"string"},
        "response_id": {"type":["string","null"]}
    },
    "required": ["error","session_id"]
}
