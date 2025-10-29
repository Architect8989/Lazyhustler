"""
Authentication and authorization for production
JWT tokens, API keys, and RBAC
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
import hashlib
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
import logging
from config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer()


class TokenManager:
    """JWT token management"""
    
    @staticmethod
    def create_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm
        )
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )


class APIKeyManager:
    """API Key management"""
    
    @staticmethod
    def hash_key(key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(key.encode()).hexdigest()
    
    @staticmethod
    def generate_key() -> str:
        """Generate new API key"""
        return f"sk_{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def verify_key(provided_key: str, stored_hash: str) -> bool:
        """Verify API key"""
        return APIKeyManager.hash_key(provided_key) == stored_hash


class RBACManager:
    """Role-Based Access Control"""
    
    PERMISSIONS = {
        "user": ["read:tasks", "create:tasks", "read:own_tasks"],
        "admin": ["read:tasks", "create:tasks", "update:tasks", "delete:tasks", "read:users"],
        "service": ["read:tasks", "create:tasks", "update:tasks", "read:metrics"],
    }
    
    @staticmethod
    def has_permission(role: str, permission: str) -> bool:
        """Check if role has permission"""
        role_permissions = RBACManager.PERMISSIONS.get(role, [])
        return permission in role_permissions
    
    @staticmethod
    def check_permission(role: str, required_permission: str):
        """Check permission and raise exception if not allowed"""
        if not RBACManager.has_permission(role, required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {required_permission}"
            )


async def get_current_user(credentials: HTTPAuthCredentials = Depends(security)) -> Dict:
    """Get current authenticated user from JWT token"""
    token = credentials.credentials
    payload = TokenManager.verify_token(token)
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    return {
        "user_id": user_id,
        "role": payload.get("role", "user"),
        "email": payload.get("email"),
        "scopes": payload.get("scopes", [])
    }


async def get_current_admin(current_user: Dict = Depends(get_current_user)) -> Dict:
    """Get current admin user"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def verify_api_key(api_key: str) -> Dict:
    """Verify API key (to be implemented with database lookup)"""
    # This would normally check against database
    if not api_key.startswith("sk_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return {"api_key": api_key, "role": "service"}


class PasswordManager:
    """Password hashing and verification"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        import bcrypt
        return bcrypt.checkpw(password.encode(), password_hash.encode())


class SecurityHeaders:
    """Security headers for responses"""
    
    @staticmethod
    def get_headers() -> Dict[str, str]:
        """Get security headers"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }
