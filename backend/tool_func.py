import sqlite3
import os
import torch
from model_train import RecommendationModel

DATABASE = os.path.join(os.path.dirname(__file__), '../database/recommendation_system.db')


def get_db_connection():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_model(model_path, num_users, num_items, num_categories, device):
    model = RecommendationModel(num_users, num_items, num_categories)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, user_id, item_ids, item_categories, hours, device):
    user_id = torch.tensor([user_id], dtype=torch.long).to(device)
    item_ids = torch.tensor(item_ids, dtype=torch.long).to(device)
    item_categories = torch.tensor(item_categories, dtype=torch.long).to(device)
    hours = torch.tensor(hours, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(user_id.repeat(len(item_ids)), item_ids, item_categories, hours)
        return predictions.cpu().numpy()