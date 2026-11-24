"""STEP A - The 5 tools.
Part 1: the real Python functions.   Part 2: the schemas that describe them to Llama.

NOTE: predict_* use a tiny hand-made "demo model" so this project runs by itself.
To use your real scikit-learn models, only change the two lines marked  # <-- SWAP
"""

# ---------- Valid values (used for input validation) ----------
JOB_ROLES = [
    "Healthcare Representative", "Human Resources", "Laboratory Technician",
    "Manager", "Manufacturing Director", "Research Director",
    "Research Scientist", "Sales Executive", "Sales Representative",
]

# The tools remember every prediction here, so explain / what-if can find it by id.
PREDICTIONS = {}


# ---------- Demo "models" (replace with your real ones) ----------
def attrition_model(job_role, job_level, over_time):
    """Return (probability, contributions). Contributions = how much each input pushed the result."""
    role_effect = {"Sales Representative": 0.15, "Laboratory Technician": 0.08,
                   "Human Resources": 0.05, "Research Scientist": 0.04,
                   "Sales Executive": 0.02, "Healthcare Representative": -0.02,
                   "Manufacturing Director": -0.03, "Manager": -0.04,
                   "Research Director": -0.04}[job_role]
    level_effect = 0.04 * (1 - job_level)
    ot_effect = 0.15 if over_time == "Yes" else 0.0
    probability = 0.15 + role_effect + level_effect + ot_effect
    probability = round(min(max(probability, 0.01), 0.99), 2)
    return probability, {"JobRole": role_effect, "JobLevel": level_effect, "OverTime": ot_effect}
    # <-- SWAP: with a real model use
    #     probability = attrition_pipeline.predict_proba(row_as_dataframe)[0][1]


def salary_model(job_role, job_level):
    role_part = {"Manager": 9000, "Research Director": 8000, "Healthcare Representative": 3000,
                 "Manufacturing Director": 3000, "Sales Executive": 2500, "Human Resources": 1000,
                 "Research Scientist": 500, "Laboratory Technician": 500,
                 "Sales Representative": 0}[job_role]
    level_part = 2200 * (job_level - 1)
    salary = 2000 + role_part + level_part
    return salary, {"JobRole": role_part, "JobLevel": level_part}
    # <-- SWAP: with a real model use
    #     salary = salary_pipeline.predict(row_as_dataframe)[0]


# ---------- Small helper: check the inputs ----------
def check_inputs(job_role, job_level, over_time=None):
    """Return an error message if something is wrong, otherwise None."""
    if job_role not in JOB_ROLES:
        return f"JobRole must be one of: {', '.join(JOB_ROLES)}"
    if job_level not in (1, 2, 3, 4, 5):
        return "JobLevel must be a whole number from 1 to 5"
    if over_time is not None and over_time not in ("Yes", "No"):
        return "OverTime must be 'Yes' or 'No'"
    return None


def run_model(model, inputs):
    """Run the right demo model. Returns (value, contributions)."""
    if model == "attrition":
        return attrition_model(inputs["JobRole"], inputs["JobLevel"], inputs["OverTime"])
    return salary_model(inputs["JobRole"], inputs["JobLevel"])


# ---------- Tool 1 ----------
def predict_attrition(JobRole, JobLevel, OverTime):
    JobLevel = int(JobLevel)
    error = check_inputs(JobRole, JobLevel, OverTime)
    if error:
        return {"error": error}
    inputs = {"JobRole": JobRole, "JobLevel": JobLevel, "OverTime": OverTime}
    probability, _ = run_model("attrition", inputs)
    prediction_id = f"pred_{len(PREDICTIONS) + 1}"
    PREDICTIONS[prediction_id] = {"model": "attrition", "inputs": inputs, "value": probability}
    return {"prediction_id": prediction_id, "attrition_probability": probability}


# ---------- Tool 2 ----------
def predict_monthly_salary(JobRole, JobLevel):
    JobLevel = int(JobLevel)
    error = check_inputs(JobRole, JobLevel)
    if error:
        return {"error": error}
    inputs = {"JobRole": JobRole, "JobLevel": JobLevel}
    salary, _ = run_model("salary", inputs)
    prediction_id = f"pred_{len(PREDICTIONS) + 1}"
    PREDICTIONS[prediction_id] = {"model": "salary", "inputs": inputs, "value": salary}
    return {"prediction_id": prediction_id, "monthly_income": salary}


# ---------- Tool 3 ----------
def explain_prediction(prediction_id, model):
    saved = PREDICTIONS.get(prediction_id)
    if saved is None or saved["model"] != model:
        return {"error": f"No {model} prediction with id '{prediction_id}'"}
    _, contributions = run_model(model, saved["inputs"])
    ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    return {"prediction_id": prediction_id, "value": saved["value"],
            "top_features": [{"feature": name, "effect": effect} for name, effect in ranked]}


# ---------- Tool 4 ----------
def what_if_analysis(prediction_id, model, changes):
    saved = PREDICTIONS.get(prediction_id)
    if saved is None or saved["model"] != model:
        return {"error": f"No {model} prediction with id '{prediction_id}'"}
    new_inputs = dict(saved["inputs"])
    for field, value in changes.items():
        if field not in new_inputs:
            return {"error": f"Cannot change '{field}'. Allowed fields: {', '.join(new_inputs)}"}
        new_inputs[field] = int(value) if field == "JobLevel" else value
    error = check_inputs(new_inputs["JobRole"], new_inputs["JobLevel"], new_inputs.get("OverTime"))
    if error:
        return {"error": error}
    new_value, _ = run_model(model, new_inputs)
    return {"before": saved["value"], "after": new_value, "changes": changes}


# ---------- Tool 5 ----------
FEATURE_INFO = {
    "JobRole": "The employee's job title. Values: " + ", ".join(JOB_ROLES),
    "JobLevel": "Seniority level, a whole number from 1 (junior) to 5 (most senior)",
    "OverTime": "Does the employee work overtime? 'Yes' or 'No'",
    "Attrition": "Target of the classification model: did the employee leave? 'Yes' or 'No'",
    "MonthlyIncome": "Target of the regression model: monthly salary in dollars",
}


def get_feature_info(feature_name):
    info = FEATURE_INFO.get(feature_name)
    if info is None:
        return {"error": f"Unknown feature. Known features: {', '.join(FEATURE_INFO)}"}
    return {"feature_name": feature_name, "meaning": info}


# ---------- Name -> function (the agent uses this to run a tool) ----------
TOOL_FUNCTIONS = {
    "predict_attrition": predict_attrition,
    "predict_monthly_salary": predict_monthly_salary,
    "explain_prediction": explain_prediction,
    "what_if_analysis": what_if_analysis,
    "get_feature_info": get_feature_info,
}


# ---------- Part 2: schemas = the "menu" we show to Llama ----------
def _tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


_role = {"type": "string", "enum": JOB_ROLES}
_level = {"type": "integer", "description": "1 to 5"}
_ot = {"type": "string", "enum": ["Yes", "No"]}
_id = {"type": "string", "description": "id returned by a predict tool, like pred_1"}
_model = {"type": "string", "enum": ["attrition", "salary"]}

TOOLS = [
    _tool("predict_attrition", "Probability that an employee leaves the company.",
          {"JobRole": _role, "JobLevel": _level, "OverTime": _ot},
          ["JobRole", "JobLevel", "OverTime"]),
    _tool("predict_monthly_salary", "Estimated monthly salary of an employee.",
          {"JobRole": _role, "JobLevel": _level},
          ["JobRole", "JobLevel"]),
    _tool("explain_prediction", "Top features behind an earlier prediction.",
          {"prediction_id": _id, "model": _model},
          ["prediction_id", "model"]),
    _tool("what_if_analysis", "Re-run an earlier prediction with some fields changed.",
          {"prediction_id": _id, "model": _model,
           "changes": {"type": "object", "description": 'Fields to change, e.g. {"OverTime": "No"}'}},
          ["prediction_id", "model", "changes"]),
    _tool("get_feature_info", "Meaning and allowed values of a data column.",
          {"feature_name": {"type": "string", "description": "e.g. JobLevel"}},
          ["feature_name"]),
]
