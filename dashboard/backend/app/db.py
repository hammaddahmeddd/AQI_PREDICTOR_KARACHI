from pymongo import MongoClient, DESCENDING, ASCENDING
from app.settings import settings

_client: MongoClient | None = None

def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=10000)
        _client.admin.command("ping")
    return _client

def get_db():
    return get_client()[settings.MONGODB_DB]

def latest_doc(collection_name: str, sort_field: str = "datetime"):
    db = get_db()
    return db[collection_name].find_one({}, {"_id": 0}, sort=[(sort_field, DESCENDING)])
