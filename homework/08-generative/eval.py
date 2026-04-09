def evaluate(submission_dir, data_dir):
    \"\"\"
    Stub for generative homework evaluation.
    This homework is purely judge-graded.
    \"\"\"
    import json
    import os
    
    results_path = os.path.join(submission_dir, "results.json")
    if not os.path.exists(results_path):
        return {"ok": False, "error": "results.json not found"}
        
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return {"ok": False, "error": f"Invalid JSON: {e}"}
        
    return {"ok": True, "score": 1.0, "data": data}
