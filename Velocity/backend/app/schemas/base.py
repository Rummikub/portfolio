from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from app.models.base import UserRole

class PaginationRequest(BaseModel):
    """Base pagination request schema"""
    page: int = Field(1, ge=1, description="Page number starting from 1")
    size: int = Field(20, ge=1, le=100, description="Number of items per page")
    cursor: Optional[str] = Field(None, description="Cursor for pagination")

class PaginationResponse(BaseModel):
    """Base pagination response schema"""
    page: int
    size: int
    total: Optional[int] = None
    has_next: bool
    next_cursor: Optional[str] = None

class BaseResponse(BaseModel):
    """Base response schema"""
    ok: bool = True
    
class ErrorResponse(BaseModel):
    """Error response schema"""
    ok: bool = False
    error: Dict[str, Any]

class UserSchema(BaseModel):
    """User schema for responses"""
    id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

class CreateUserRequest(BaseModel):
    """Create user request schema"""
    email: EmailStr
    password: str = Field(min_length=8, description="Minimum 8 characters")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: UserRole = UserRole.USER

class LoginRequest(BaseModel):
    """Login request schema"""
    email: EmailStr
    password: str

class LoginResponse(BaseResponse):
    """Login response schema"""
    access_token: str
    refresh_token: str
    user: UserSchema

class RefreshTokenRequest(BaseModel):
    """Refresh token request schema"""
    refresh_token: str

class AgentSchema(BaseModel):
    """Agent schema for responses"""
    id: str
    name: str
    description: Optional[str] = None
    type: str
    config: Dict[str, Any] = {}
    status: str
    last_run: Optional[datetime] = None
    metrics: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime

class CreateAgentRequest(BaseModel):
    """Create agent request schema"""
    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    type: str = Field(min_length=1, max_length=50)
    config: Dict[str, Any] = Field(default_factory=dict)

class UpdateAgentRequest(BaseModel):
    """Update agent request schema"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None

class DocSchema(BaseModel):
    """Document schema for responses"""
    id: str
    title: str
    content: Optional[str] = None
    type: str
    tags: List[str] = []
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime

class CreateDocRequest(BaseModel):
    """Create document request schema"""
    title: str = Field(min_length=1, max_length=200)
    content: Optional[str] = None
    type: str = Field(min_length=1, max_length=50)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DesignPartnerSchema(BaseModel):
    """Design Partner schema for responses"""
    id: str
    name: str
    email: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    status: str
    contact_history: List[Dict[str, Any]] = []
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class CreateDesignPartnerRequest(BaseModel):
    """Create design partner request schema"""
    name: str = Field(min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    company: Optional[str] = Field(None, max_length=100)
    industry: Optional[str] = Field(None, max_length=50)
    status: str = Field("prospect", max_length=20)
    notes: Optional[str] = Field(None, max_length=1000)

class ResearchSchema(BaseModel):
    """Research schema for responses"""
    id: str
    title: str
    type: str
    data: Dict[str, Any] = {}
    source: Optional[str] = None
    confidence_score: Optional[float] = None
    tags: List[str] = []
    created_at: datetime
    updated_at: datetime

class CreateResearchRequest(BaseModel):
    """Create research request schema"""
    title: str = Field(min_length=1, max_length=200)
    type: str = Field(min_length=1, max_length=50)
    data: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = Field(None, max_length=100)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)