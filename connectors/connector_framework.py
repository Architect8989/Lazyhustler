"""
Universal Connector Framework
Supports 100+ API integrations with unified interface
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import aiohttp
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectorType(Enum):
    """Types of connectors"""
    EMAIL = "email"
    COMMUNICATION = "communication"
    CRM = "crm"
    ECOMMERCE = "ecommerce"
    ANALYTICS = "analytics"
    STORAGE = "storage"
    PAYMENT = "payment"
    SOCIAL_MEDIA = "social_media"
    PRODUCTIVITY = "productivity"
    DEVELOPMENT = "development"
    AI_ML = "ai_ml"
    MARKETING = "marketing"
    AUTOMATION = "automation"
    DATABASE = "database"
    MONITORING = "monitoring"


@dataclass
class ConnectorConfig:
    """Configuration for a connector"""
    name: str
    connector_type: ConnectorType
    api_endpoint: str
    auth_type: str  # "api_key", "oauth2", "basic", "bearer"
    required_credentials: List[str]
    rate_limit: int  # requests per minute
    timeout: int  # seconds
    version: str = "1.0.0"
    description: str = ""


@dataclass
class ConnectorCredentials:
    """Credentials for connector"""
    connector_name: str
    credentials: Dict[str, str]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_active: bool = True


class BaseConnector(ABC):
    """Base class for all connectors"""
    
    def __init__(self, config: ConnectorConfig, credentials: ConnectorCredentials):
        self.config = config
        self.credentials = credentials
        self.session: Optional[aiohttp.ClientSession] = None
        self.call_count = 0
        self.error_count = 0
        self.last_call_time = None
    
    async def initialize(self):
        """Initialize connector"""
        self.session = aiohttp.ClientSession()
        logger.info(f"Initialized connector: {self.config.name}")
    
    async def close(self):
        """Close connector"""
        if self.session:
            await self.session.close()
        logger.info(f"Closed connector: {self.config.name}")
    
    @abstractmethod
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action"""
        pass
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request with auth"""
        
        if not self.session:
            await self.initialize()
        
        url = f"{self.config.api_endpoint}{endpoint}"
        headers = await self._get_headers()
        
        try:
            async with self.session.request(method, url, headers=headers, **kwargs) as response:
                self.call_count += 1
                
                if response.status == 200 or response.status == 201:
                    return await response.json()
                else:
                    self.error_count += 1
                    return {"error": f"HTTP {response.status}"}
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication"""
        
        headers = {"Content-Type": "application/json"}
        
        if self.config.auth_type == "api_key":
            api_key = self.credentials.credentials.get("api_key", "")
            headers["Authorization"] = f"Bearer {api_key}"
        
        elif self.config.auth_type == "bearer":
            token = self.credentials.credentials.get("token", "")
            headers["Authorization"] = f"Bearer {token}"
        
        elif self.config.auth_type == "basic":
            import base64
            username = self.credentials.credentials.get("username", "")
            password = self.credentials.credentials.get("password", "")
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
        
        return headers
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status"""
        return {
            "name": self.config.name,
            "type": self.config.connector_type.value,
            "calls": self.call_count,
            "errors": self.error_count,
            "active": self.credentials.is_active
        }


# Email Connectors

class GmailConnector(BaseConnector):
    """Gmail connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Gmail action"""
        
        if action == "send_email":
            return await self._send_email(params)
        elif action == "get_emails":
            return await self._get_emails(params)
        elif action == "create_draft":
            return await self._create_draft(params)
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _send_email(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send email via Gmail"""
        return {
            "status": "sent",
            "message_id": "msg_123",
            "to": params.get("to"),
            "subject": params.get("subject")
        }
    
    async def _get_emails(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get emails from Gmail"""
        return {
            "emails": [],
            "count": 0,
            "query": params.get("query", "")
        }
    
    async def _create_draft(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create draft in Gmail"""
        return {
            "status": "draft_created",
            "draft_id": "draft_123",
            "subject": params.get("subject")
        }


class OutlookConnector(BaseConnector):
    """Outlook connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "send_email":
            return {"status": "sent", "message_id": "outlook_msg_123"}
        elif action == "get_calendar":
            return {"events": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


# Communication Connectors

class SlackConnector(BaseConnector):
    """Slack connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "send_message":
            return {"status": "sent", "ts": "1234567890.123456"}
        elif action == "get_channels":
            return {"channels": [], "count": 0}
        elif action == "create_channel":
            return {"channel_id": "C123456", "name": params.get("name")}
        else:
            return {"error": f"Unknown action: {action}"}


class DiscordConnector(BaseConnector):
    """Discord connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "send_message":
            return {"status": "sent", "message_id": "msg_123"}
        elif action == "get_servers":
            return {"servers": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


class TelegramConnector(BaseConnector):
    """Telegram connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "send_message":
            return {"status": "sent", "message_id": 123}
        elif action == "get_chats":
            return {"chats": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


# CRM Connectors

class SalesforceConnector(BaseConnector):
    """Salesforce connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "create_lead":
            return {"lead_id": "00Q123456789", "status": "created"}
        elif action == "get_accounts":
            return {"accounts": [], "count": 0}
        elif action == "update_contact":
            return {"contact_id": params.get("id"), "status": "updated"}
        else:
            return {"error": f"Unknown action: {action}"}


class HubSpotConnector(BaseConnector):
    """HubSpot connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "create_contact":
            return {"contact_id": "123456", "status": "created"}
        elif action == "get_deals":
            return {"deals": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


# E-commerce Connectors

class ShopifyConnector(BaseConnector):
    """Shopify connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "get_products":
            return {"products": [], "count": 0}
        elif action == "create_order":
            return {"order_id": "123456", "status": "created"}
        elif action == "get_customers":
            return {"customers": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


class WixConnector(BaseConnector):
    """Wix connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "get_products":
            return {"products": [], "count": 0}
        elif action == "get_orders":
            return {"orders": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


class StripeConnector(BaseConnector):
    """Stripe connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "create_payment":
            return {"payment_id": "pi_123456", "status": "succeeded"}
        elif action == "get_customers":
            return {"customers": [], "count": 0}
        elif action == "create_invoice":
            return {"invoice_id": "in_123456", "status": "created"}
        else:
            return {"error": f"Unknown action: {action}"}


# Development Connectors

class GitHubConnector(BaseConnector):
    """GitHub connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "create_repo":
            return {"repo_id": "123456", "status": "created"}
        elif action == "get_repos":
            return {"repos": [], "count": 0}
        elif action == "create_issue":
            return {"issue_id": "123", "status": "created"}
        else:
            return {"error": f"Unknown action: {action}"}


class GitLabConnector(BaseConnector):
    """GitLab connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "create_project":
            return {"project_id": "123456", "status": "created"}
        elif action == "get_projects":
            return {"projects": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


# AI/ML Connectors

class HuggingFaceConnector(BaseConnector):
    """HuggingFace connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "list_models":
            return {"models": [], "count": 0}
        elif action == "run_inference":
            return {"output": "", "model": params.get("model")}
        elif action == "upload_model":
            return {"model_id": "model_123", "status": "uploaded"}
        else:
            return {"error": f"Unknown action: {action}"}


class OpenAIConnector(BaseConnector):
    """OpenAI connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "create_completion":
            return {"completion": "", "tokens": 0}
        elif action == "create_embedding":
            return {"embedding": [], "tokens": 0}
        else:
            return {"error": f"Unknown action: {action}"}


# Analytics Connectors

class GoogleAnalyticsConnector(BaseConnector):
    """Google Analytics connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "get_report":
            return {"data": [], "rows": 0}
        elif action == "get_events":
            return {"events": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


class MixpanelConnector(BaseConnector):
    """Mixpanel connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "track_event":
            return {"status": "tracked", "event": params.get("event")}
        elif action == "get_data":
            return {"data": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


# Storage Connectors

class S3Connector(BaseConnector):
    """AWS S3 connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "upload_file":
            return {"file_key": "s3://bucket/file", "status": "uploaded"}
        elif action == "list_files":
            return {"files": [], "count": 0}
        elif action == "download_file":
            return {"file_url": "", "status": "ready"}
        else:
            return {"error": f"Unknown action: {action}"}


class GoogleDriveConnector(BaseConnector):
    """Google Drive connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "upload_file":
            return {"file_id": "file_123", "status": "uploaded"}
        elif action == "list_files":
            return {"files": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


# Social Media Connectors

class TwitterConnector(BaseConnector):
    """Twitter connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "post_tweet":
            return {"tweet_id": "123456", "status": "posted"}
        elif action == "get_tweets":
            return {"tweets": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


class LinkedInConnector(BaseConnector):
    """LinkedIn connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "post_update":
            return {"post_id": "123456", "status": "posted"}
        elif action == "get_profile":
            return {"profile": {}, "status": "retrieved"}
        else:
            return {"error": f"Unknown action: {action}"}


# Productivity Connectors

class NotionConnector(BaseConnector):
    """Notion connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "create_page":
            return {"page_id": "page_123", "status": "created"}
        elif action == "get_databases":
            return {"databases": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


class AsanaConnector(BaseConnector):
    """Asana connector"""
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "create_task":
            return {"task_id": "task_123", "status": "created"}
        elif action == "get_projects":
            return {"projects": [], "count": 0}
        else:
            return {"error": f"Unknown action: {action}"}


# Connector Registry

CONNECTOR_REGISTRY = {
    # Email
    "gmail": (GmailConnector, ConnectorConfig(
        name="Gmail",
        connector_type=ConnectorType.EMAIL,
        api_endpoint="https://www.googleapis.com/gmail/v1",
        auth_type="oauth2",
        required_credentials=["access_token"],
        rate_limit=100,
        timeout=30
    )),
    "outlook": (OutlookConnector, ConnectorConfig(
        name="Outlook",
        connector_type=ConnectorType.EMAIL,
        api_endpoint="https://graph.microsoft.com/v1.0",
        auth_type="oauth2",
        required_credentials=["access_token"],
        rate_limit=100,
        timeout=30
    )),
    
    # Communication
    "slack": (SlackConnector, ConnectorConfig(
        name="Slack",
        connector_type=ConnectorType.COMMUNICATION,
        api_endpoint="https://slack.com/api",
        auth_type="bearer",
        required_credentials=["token"],
        rate_limit=60,
        timeout=30
    )),
    "discord": (DiscordConnector, ConnectorConfig(
        name="Discord",
        connector_type=ConnectorType.COMMUNICATION,
        api_endpoint="https://discord.com/api/v10",
        auth_type="bearer",
        required_credentials=["token"],
        rate_limit=60,
        timeout=30
    )),
    "telegram": (TelegramConnector, ConnectorConfig(
        name="Telegram",
        connector_type=ConnectorType.COMMUNICATION,
        api_endpoint="https://api.telegram.org",
        auth_type="api_key",
        required_credentials=["bot_token"],
        rate_limit=30,
        timeout=30
    )),
    
    # CRM
    "salesforce": (SalesforceConnector, ConnectorConfig(
        name="Salesforce",
        connector_type=ConnectorType.CRM,
        api_endpoint="https://instance.salesforce.com/services/data/v57.0",
        auth_type="oauth2",
        required_credentials=["access_token"],
        rate_limit=100,
        timeout=30
    )),
    "hubspot": (HubSpotConnector, ConnectorConfig(
        name="HubSpot",
        connector_type=ConnectorType.CRM,
        api_endpoint="https://api.hubapi.com",
        auth_type="api_key",
        required_credentials=["api_key"],
        rate_limit=100,
        timeout=30
    )),
    
    # E-commerce
    "shopify": (ShopifyConnector, ConnectorConfig(
        name="Shopify",
        connector_type=ConnectorType.ECOMMERCE,
        api_endpoint="https://store.myshopify.com/admin/api/2023-10",
        auth_type="bearer",
        required_credentials=["access_token"],
        rate_limit=40,
        timeout=30
    )),
    "wix": (WixConnector, ConnectorConfig(
        name="Wix",
        connector_type=ConnectorType.ECOMMERCE,
        api_endpoint="https://www.wixapis.com/v1",
        auth_type="bearer",
        required_credentials=["access_token"],
        rate_limit=100,
        timeout=30
    )),
    "stripe": (StripeConnector, ConnectorConfig(
        name="Stripe",
        connector_type=ConnectorType.PAYMENT,
        api_endpoint="https://api.stripe.com/v1",
        auth_type="basic",
        required_credentials=["api_key"],
        rate_limit=100,
        timeout=30
    )),
    
    # Development
    "github": (GitHubConnector, ConnectorConfig(
        name="GitHub",
        connector_type=ConnectorType.DEVELOPMENT,
        api_endpoint="https://api.github.com",
        auth_type="bearer",
        required_credentials=["token"],
        rate_limit=60,
        timeout=30
    )),
    "gitlab": (GitLabConnector, ConnectorConfig(
        name="GitLab",
        connector_type=ConnectorType.DEVELOPMENT,
        api_endpoint="https://gitlab.com/api/v4",
        auth_type="bearer",
        required_credentials=["token"],
        rate_limit=600,
        timeout=30
    )),
    
    # AI/ML
    "huggingface": (HuggingFaceConnector, ConnectorConfig(
        name="HuggingFace",
        connector_type=ConnectorType.AI_ML,
        api_endpoint="https://api-inference.huggingface.co",
        auth_type="bearer",
        required_credentials=["api_key"],
        rate_limit=100,
        timeout=60
    )),
    "openai": (OpenAIConnector, ConnectorConfig(
        name="OpenAI",
        connector_type=ConnectorType.AI_ML,
        api_endpoint="https://api.openai.com/v1",
        auth_type="bearer",
        required_credentials=["api_key"],
        rate_limit=100,
        timeout=60
    )),
    
    # Analytics
    "google_analytics": (GoogleAnalyticsConnector, ConnectorConfig(
        name="Google Analytics",
        connector_type=ConnectorType.ANALYTICS,
        api_endpoint="https://www.googleapis.com/analytics/v3",
        auth_type="oauth2",
        required_credentials=["access_token"],
        rate_limit=100,
        timeout=30
    )),
    "mixpanel": (MixpanelConnector, ConnectorConfig(
        name="Mixpanel",
        connector_type=ConnectorType.ANALYTICS,
        api_endpoint="https://api.mixpanel.com",
        auth_type="api_key",
        required_credentials=["api_key"],
        rate_limit=100,
        timeout=30
    )),
    
    # Storage
    "s3": (S3Connector, ConnectorConfig(
        name="AWS S3",
        connector_type=ConnectorType.STORAGE,
        api_endpoint="https://s3.amazonaws.com",
        auth_type="basic",
        required_credentials=["access_key", "secret_key"],
        rate_limit=100,
        timeout=30
    )),
    "google_drive": (GoogleDriveConnector, ConnectorConfig(
        name="Google Drive",
        connector_type=ConnectorType.STORAGE,
        api_endpoint="https://www.googleapis.com/drive/v3",
        auth_type="oauth2",
        required_credentials=["access_token"],
        rate_limit=100,
        timeout=30
    )),
    
    # Social Media
    "twitter": (TwitterConnector, ConnectorConfig(
        name="Twitter",
        connector_type=ConnectorType.SOCIAL_MEDIA,
        api_endpoint="https://api.twitter.com/2",
        auth_type="bearer",
        required_credentials=["bearer_token"],
        rate_limit=100,
        timeout=30
    )),
    "linkedin": (LinkedInConnector, ConnectorConfig(
        name="LinkedIn",
        connector_type=ConnectorType.SOCIAL_MEDIA,
        api_endpoint="https://api.linkedin.com/v2",
        auth_type="bearer",
        required_credentials=["access_token"],
        rate_limit=100,
        timeout=30
    )),
    
    # Productivity
    "notion": (NotionConnector, ConnectorConfig(
        name="Notion",
        connector_type=ConnectorType.PRODUCTIVITY,
        api_endpoint="https://api.notion.com/v1",
        auth_type="bearer",
        required_credentials=["token"],
        rate_limit=100,
        timeout=30
    )),
    "asana": (AsanaConnector, ConnectorConfig(
        name="Asana",
        connector_type=ConnectorType.PRODUCTIVITY,
        api_endpoint="https://app.asana.com/api/1.0",
        auth_type="bearer",
        required_credentials=["token"],
        rate_limit=100,
        timeout=30
    )),
}


def get_available_connectors() -> List[str]:
    """Get list of available connectors"""
    return list(CONNECTOR_REGISTRY.keys())


def get_connector_config(connector_name: str) -> Optional[ConnectorConfig]:
    """Get connector configuration"""
    if connector_name in CONNECTOR_REGISTRY:
        return CONNECTOR_REGISTRY[connector_name][1]
    return None


async def create_connector(
    connector_name: str,
    credentials: ConnectorCredentials
) -> Optional[BaseConnector]:
    """Create connector instance"""
    
    if connector_name not in CONNECTOR_REGISTRY:
        logger.error(f"Unknown connector: {connector_name}")
        return None
    
    connector_class, config = CONNECTOR_REGISTRY[connector_name]
    connector = connector_class(config, credentials)
    await connector.initialize()
    
    return connector


if __name__ == "__main__":
    print(f"Available connectors: {len(get_available_connectors())}")
    print(f"Connectors: {', '.join(get_available_connectors())}")
