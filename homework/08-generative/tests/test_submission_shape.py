import os
import json

def test_submission_files_exist():
    assert os.path.exists("submission/notebook.py"), "notebook.py is missing"
    assert os.path.exists("submission/MODEL.md"), "MODEL.md is missing"

def test_results_json_valid():
    results_path = "submission/results.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                assert False, "results.json is not a valid JSON file"
            
        assert "final_elbo" in data, "Missing final_elbo in results.json"
        assert "num_epochs" in data, "Missing num_epochs in results.json"
