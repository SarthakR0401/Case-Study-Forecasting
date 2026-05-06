import joblib
import os

MODEL_DIR = 'saved_models'

def load_model_for_state(state):
    """
    Loads the best model and the model mapping.
    """
    map_path = os.path.join(MODEL_DIR, "state_model_map.joblib")
    model_path = os.path.join(MODEL_DIR, f"{state}_best_model.joblib")
    
    if os.path.exists(map_path) and os.path.exists(model_path):
        model_map = joblib.load(map_path)
        model = joblib.load(model_path)
        return model, model_map.get(state, "unknown")
    return None, None
