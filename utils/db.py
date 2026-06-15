"""
Database layer with in-memory fallback when MongoDB is unavailable.
"""
from config import Config

client = None
db = None
_use_memory = False


class InMemoryCollection:
    """Minimal MongoDB-compatible in-memory collection."""
    def __init__(self):
        self._docs = []

    def insert_one(self, doc):
        doc = dict(doc)
        self._docs.append(doc)
        return type("Result", (), {"inserted_id": id(doc)})()

    def find_one(self, query=None, sort=None, **kwargs):
        results = self._find(query)
        if sort:
            key, direction = sort[0] if isinstance(sort, list) else sort
            results.sort(key=lambda d: d.get(key, ""), reverse=(direction == -1))
        return results[0] if results else None

    def find(self, query=None, projection=None, **kwargs):
        return InMemoryCursor(self._find(query))

    def count_documents(self, query=None):
        return len(self._find(query or {}))

    def update_one(self, query, update):
        for doc in self._docs:
            if self._match(doc, query):
                if "$set" in update:
                    doc.update(update["$set"])
                return type("Result", (), {"modified_count": 1})()
        return type("Result", (), {"modified_count": 0})()

    def create_index(self, *args, **kwargs):
        pass  # no-op for in-memory

    def _find(self, query=None):
        if not query:
            return list(self._docs)
        return [d for d in self._docs if self._match(d, query)]

    @staticmethod
    def _match(doc, query):
        for key, val in (query or {}).items():
            if key == "$or":
                if not any(InMemoryCollection._match(doc, q) for q in val):
                    return False
                continue
            doc_val = doc.get(key)
            if isinstance(val, dict):
                if "$regex" in val:
                    import re
                    flags = re.IGNORECASE if val.get("$options", "") == "i" else 0
                    if not re.search(val["$regex"], str(doc_val or ""), flags):
                        return False
            elif doc_val != val:
                return False
        return True


class InMemoryCursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, key, direction=-1):
        if isinstance(key, list):
            key, direction = key[0]
        self._docs.sort(key=lambda d: d.get(key, ""), reverse=(direction == -1))
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)

    def __list__(self):
        return self._docs


class InMemoryDB:
    """Dict-like object that auto-creates collections."""
    def __init__(self):
        self._collections = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name not in self._collections:
            self._collections[name] = InMemoryCollection()
        return self._collections[name]


def init_db(app=None):
    global client, db, _use_memory
    uri = Config.MONGO_URI if app is None else app.config.get("MONGO_URI", Config.MONGO_URI)
    try:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        # Force connection test
        client.admin.command("ping")
        db = client.get_default_database() if "/" in uri.split("//")[-1] else client["traffic_management"]

        db.admins.create_index("email", unique=True)
        db.traffic_logs.create_index([("timestamp", -1)])
        db.incidents.create_index([("timestamp", -1)])
        db.detections.create_index([("timestamp", -1)])
        _use_memory = False
        print("[DB] MongoDB connected successfully")
    except Exception as e:
        print(f"[DB] MongoDB unavailable: {e}")
        print("[DB] Using in-memory storage (data will reset on restart)")
        db = InMemoryDB()
        _use_memory = True
    return db


def get_db():
    global db
    if db is None:
        init_db()
    return db


def seed_default_admin():
    """Create a default admin if none exists."""
    import bcrypt
    database = get_db()

    # TEMPORARY - delete existing admin records
    database.admins.delete_many({})
    if database.admins.count_documents({}) == 0:
        hashed = bcrypt.hashpw("anu@123".encode("utf-8"), bcrypt.gensalt())
        database.admins.insert_one({
            "email": "emmeroj1210@gmail.com",
            "password": hashed,
            "name": "Traffic Admin",
            "role": "super_admin"
        })
        print("[DB] Default admin created: emmeroj1210@gmail.com")