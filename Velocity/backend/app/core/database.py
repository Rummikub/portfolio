from motor.motor_asyncio import AsyncIOMotorClient
from flask import Flask, g
import asyncio
from functools import wraps
from typing import Optional

# Global database client
db_client: Optional[AsyncIOMotorClient] = None
db = None

def init_db(app: Flask):
    """Initialize MongoDB connection"""
    global db_client, db
    
    mongo_uri = app.config['MONGO_URI']
    db_name = app.config['MONGO_DB_NAME']
    
    try:
        db_client = AsyncIOMotorClient(mongo_uri)
        db = db_client[db_name]
        app.logger.info(f"Connected to MongoDB: {db_name}")
    except Exception as e:
        app.logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def get_db():
    """Get database instance"""
    return db

def get_collection(collection_name: str):
    """Get a specific collection from the database"""
    return db[collection_name]

async def create_indexes():
    """Create database indexes for optimal performance"""
    # Users collection indexes
    users_collection = get_collection('users')
    await users_collection.create_index("email", unique=True)
    await users_collection.create_index([("workspace_id", 1), ("created_at", -1)])
    
    # Agents collection indexes
    agents_collection = get_collection('agents')
    await agents_collection.create_index([("workspace_id", 1), ("created_at", -1)])
    await agents_collection.create_index([("workspace_id", 1), ("status", 1)])
    
    # Docs collection indexes
    docs_collection = get_collection('docs')
    await docs_collection.create_index([("workspace_id", 1), ("created_at", -1)])
    await docs_collection.create_index([("workspace_id", 1), ("type", 1)])
    
    # Design Partners collection indexes
    design_partners_collection = get_collection('design_partners')
    await design_partners_collection.create_index([("workspace_id", 1), ("created_at", -1)])
    await design_partners_collection.create_index([("workspace_id", 1), ("status", 1)])
    
    # Research collection indexes
    research_collection = get_collection('research')
    await research_collection.create_index([("workspace_id", 1), ("created_at", -1)])
    await research_collection.create_index([("workspace_id", 1), ("type", 1)])

def async_route(f):
    """Decorator to run async functions in Flask routes"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(f(*args, **kwargs))
    return wrapper

async def close_db():
    """Close database connection"""
    global db_client
    if db_client:
        db_client.close()