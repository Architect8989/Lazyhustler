"""
Connector Manager
Manages connector lifecycle, authentication, and credential storage
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import sqlite3
from pathlib import Path

from .connector_framework import (
    BaseConnector, ConnectorCredentials, create_connector,
    get_available_connectors, get_connector_config
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectorManager:
    """Manages all connectors and their credentials"""
    
    def __init__(self, db_path: str = "connectors.db"):
        self.db_path = db_path
        self.connectors: Dict[str, BaseConnector] = {}
        self.credentials_store: Dict[str, ConnectorCredentials] = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for credentials"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connector_credentials (
                id INTEGER PRIMARY KEY,
                connector_name TEXT NOT NULL,
                credentials TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connector_usage (
                id INTEGER PRIMARY KEY,
                connector_name TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_ms INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized: {self.db_path}")
    
    async def register_connector(
        self,
        connector_name: str,
        credentials_dict: Dict[str, str]
    ) -> bool:
        """Register a connector with credentials"""
        
        if connector_name not in get_available_connectors():
            logger.error(f"Unknown connector: {connector_name}")
            return False
        
        try:
            # Create credentials object
            credentials = ConnectorCredentials(
                connector_name=connector_name,
                credentials=credentials_dict
            )
            
            # Create connector instance
            connector = await create_connector(connector_name, credentials)
            
            if connector:
                self.connectors[connector_name] = connector
                self.credentials_store[connector_name] = credentials
                
                # Store in database
                self._store_credentials(connector_name, credentials_dict)
                
                logger.info(f"Registered connector: {connector_name}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to register connector {connector_name}: {e}")
            return False
    
    def _store_credentials(self, connector_name: str, credentials: Dict[str, str]):
        """Store credentials in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Encrypt credentials in production
        credentials_json = json.dumps(credentials)
        
        cursor.execute("""
            INSERT INTO connector_credentials (connector_name, credentials)
            VALUES (?, ?)
        """, (connector_name, credentials_json))
        
        conn.commit()
        conn.close()
    
    async def execute_connector_action(
        self,
        connector_name: str,
        action: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an action on a connector"""
        
        if connector_name not in self.connectors:
            return {"error": f"Connector not registered: {connector_name}"}
        
        try:
            connector = self.connectors[connector_name]
            start_time = datetime.now()
            
            result = await connector.execute(action, params)
            
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Log usage
            self._log_usage(connector_name, action, "success", duration_ms)
            
            return result
        
        except Exception as e:
            logger.error(f"Action failed: {e}")
            self._log_usage(connector_name, action, "error", 0)
            return {"error": str(e)}
    
    def _log_usage(self, connector_name: str, action: str, status: str, duration_ms: int):
        """Log connector usage"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO connector_usage (connector_name, action, status, duration_ms)
            VALUES (?, ?, ?, ?)
        """, (connector_name, action, status, duration_ms))
        
        conn.commit()
        conn.close()
    
    def get_connector_status(self, connector_name: str) -> Dict[str, Any]:
        """Get status of a connector"""
        
        if connector_name not in self.connectors:
            return {"error": f"Connector not registered: {connector_name}"}
        
        connector = self.connectors[connector_name]
        return connector.get_status()
    
    def get_all_connector_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connectors"""
        
        return {
            name: connector.get_status()
            for name, connector in self.connectors.items()
        }
    
    def get_usage_stats(self, connector_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics for a connector"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute("""
            SELECT action, status, COUNT(*) as count, AVG(duration_ms) as avg_duration
            FROM connector_usage
            WHERE connector_name = ? AND timestamp > ?
            GROUP BY action, status
        """, (connector_name, cutoff_time.isoformat()))
        
        rows = cursor.fetchall()
        conn.close()
        
        stats = {
            "connector": connector_name,
            "period_hours": hours,
            "actions": {}
        }
        
        for action, status, count, avg_duration in rows:
            key = f"{action}_{status}"
            stats["actions"][key] = {
                "count": count,
                "avg_duration_ms": avg_duration
            }
        
        return stats
    
    async def test_connector(self, connector_name: str) -> bool:
        """Test if connector is working"""
        
        if connector_name not in self.connectors:
            return False
        
        try:
            connector = self.connectors[connector_name]
            # Try a simple action
            result = await connector.execute("test", {})
            return "error" not in result
        
        except Exception as e:
            logger.error(f"Connector test failed: {e}")
            return False
    
    async def close_all(self):
        """Close all connectors"""
        
        for connector in self.connectors.values():
            await connector.close()
        
        logger.info("All connectors closed")
    
    def list_available_connectors(self) -> List[Dict[str, Any]]:
        """List all available connectors"""
        
        connectors = []
        for name in get_available_connectors():
            config = get_connector_config(name)
            connectors.append({
                "name": name,
                "type": config.connector_type.value,
                "description": config.description,
                "auth_type": config.auth_type,
                "required_credentials": config.required_credentials,
                "registered": name in self.connectors
            })
        
        return connectors
    
    def get_registered_connectors(self) -> List[str]:
        """Get list of registered connectors"""
        return list(self.connectors.keys())
    
    async def batch_execute(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute multiple connector actions in parallel"""
        
        async def execute_task(task):
            connector_name = task.get("connector")
            action = task.get("action")
            params = task.get("params", {})
            
            return await self.execute_connector_action(connector_name, action, params)
        
        results = await asyncio.gather(
            *[execute_task(task) for task in tasks],
            return_exceptions=True
        )
        
        return results


# Global connector manager instance
_manager: Optional[ConnectorManager] = None


def get_connector_manager() -> ConnectorManager:
    """Get global connector manager"""
    global _manager
    if _manager is None:
        _manager = ConnectorManager()
    return _manager


async def initialize_connectors(connector_configs: Dict[str, Dict[str, str]]) -> bool:
    """Initialize connectors with provided credentials"""
    
    manager = get_connector_manager()
    
    for connector_name, credentials in connector_configs.items():
        success = await manager.register_connector(connector_name, credentials)
        if not success:
            logger.warning(f"Failed to initialize connector: {connector_name}")
    
    return True


if __name__ == "__main__":
    # Test connector manager
    async def test():
        manager = ConnectorManager()
        
        # List available connectors
        available = manager.list_available_connectors()
        print(f"Available connectors: {len(available)}")
        for conn in available[:5]:
            print(f"  - {conn['name']} ({conn['type']})")
        
        # Register a test connector
        await manager.register_connector("slack", {
            "token": "xoxb-test-token"
        })
        
        # Get status
        status = manager.get_all_connector_status()
        print(f"\nConnector status: {json.dumps(status, indent=2, default=str)}")
        
        # Close
        await manager.close_all()
    
    asyncio.run(test())
