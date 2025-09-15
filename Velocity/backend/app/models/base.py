from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from bson import ObjectId
from pydantic import BaseModel, Field
from enum import Enum

class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid ObjectId')
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='string')

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"

class BaseDBModel(BaseModel):
    """Base model for all database documents"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias='_id')
    workspace_id: PyObjectId
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[PyObjectId] = None
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserModel(BaseDBModel):
    """User model"""
    email: str
    password_hash: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True
    last_login: Optional[datetime] = None
    
class AgentModel(BaseDBModel):
    """Agent model"""
    name: str
    description: Optional[str] = None
    type: str  # e.g., "market_discovery", "validation", "customer_finder"
    config: Dict[str, Any] = Field(default_factory=dict)
    status: str = "idle"  # idle, running, completed, failed
    last_run: Optional[datetime] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

class DocModel(BaseDBModel):
    """Document model"""
    title: str
    content: Optional[str] = None
    type: str  # e.g., "research", "analysis", "report"
    tags: List[str] = Field(default_factory=list)
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DesignPartnerModel(BaseDBModel):
    """Design Partner model"""
    name: str
    email: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    status: str = "prospect"  # prospect, contacted, engaged, partner
    contact_history: List[Dict[str, Any]] = Field(default_factory=list)
    notes: Optional[str] = None
    
class ResearchModel(BaseDBModel):
    """Research model"""
    title: str
    type: str  # e.g., "market_analysis", "competitor_research", "user_feedback"
    data: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
    confidence_score: Optional[float] = None
    tags: List[str] = Field(default_factory=list)