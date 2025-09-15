from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from bson import ObjectId
import base64
import json
from app.schemas.base import PaginationRequest, PaginationResponse

def create_cursor(doc: Dict[str, Any]) -> str:
    """Create a cursor from a document"""
    cursor_data = {
        'id': str(doc['_id']),
        'created_at': doc['created_at'].isoformat()
    }
    cursor_json = json.dumps(cursor_data, sort_keys=True)
    return base64.b64encode(cursor_json.encode()).decode()

def parse_cursor(cursor: str) -> Dict[str, Any]:
    """Parse a cursor back to data"""
    try:
        cursor_json = base64.b64decode(cursor.encode()).decode()
        return json.loads(cursor_json)
    except Exception:
        raise ValueError("Invalid cursor format")

async def paginate_collection(
    collection,
    workspace_id: str,
    pagination: PaginationRequest,
    filter_query: Dict[str, Any] = None,
    sort_field: str = "created_at",
    sort_direction: int = -1
) -> Tuple[List[Dict[str, Any]], PaginationResponse]:
    """
    Paginate a MongoDB collection with cursor support
    
    Args:
        collection: MongoDB collection
        workspace_id: Workspace ID to filter by
        pagination: Pagination parameters
        filter_query: Additional filter query
        sort_field: Field to sort by
        sort_direction: Sort direction (1 for asc, -1 for desc)
    
    Returns:
        Tuple of (documents, pagination_info)
    """
    
    # Build base query
    query = {"workspace_id": ObjectId(workspace_id)}
    if filter_query:
        query.update(filter_query)
    
    # Handle cursor pagination
    if pagination.cursor:
        try:
            cursor_data = parse_cursor(pagination.cursor)
            cursor_created_at = datetime.fromisoformat(cursor_data['created_at'])
            cursor_id = ObjectId(cursor_data['id'])
            
            # Add cursor filter to query
            if sort_direction == -1:  # Descending
                query[sort_field] = {"$lt": cursor_created_at}
            else:  # Ascending
                query[sort_field] = {"$gt": cursor_created_at}
                
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid cursor: {e}")
    
    # Execute query with limit + 1 to check if there are more results
    cursor = collection.find(query).sort(sort_field, sort_direction).limit(pagination.size + 1)
    documents = await cursor.to_list(length=pagination.size + 1)
    
    # Check if there are more results
    has_next = len(documents) > pagination.size
    if has_next:
        documents = documents[:-1]  # Remove the extra document
    
    # Create next cursor if there are more results
    next_cursor = None
    if has_next and documents:
        next_cursor = create_cursor(documents[-1])
    
    # Create pagination response
    pagination_response = PaginationResponse(
        page=pagination.page,
        size=len(documents),
        has_next=has_next,
        next_cursor=next_cursor
    )
    
    return documents, pagination_response

def convert_objectids_to_strings(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ObjectId fields to strings in a document"""
    if isinstance(doc, dict):
        result = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, dict):
                result[key] = convert_objectids_to_strings(value)
            elif isinstance(value, list):
                result[key] = [convert_objectids_to_strings(item) if isinstance(item, dict) else str(item) if isinstance(item, ObjectId) else item for item in value]
            else:
                result[key] = value
        return result
    return doc