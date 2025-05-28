from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseConnector(ABC):
    """Abstract base class for all data connectors in the integration platform."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the connector with a configuration dictionary.
        
        Args:
            config: Dictionary containing connector-specific settings.
        """
        self.config = config
        self.connected = False

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the data source.
        
        Raises:
            ConnectionError: If the connection cannot be established.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the data source."""
        pass

    @abstractmethod
    def fetch_data(self) -> Any:
        """Fetch data from the source.
        
        Returns:
            The data retrieved from the source in a connector-specific format.
        
        Raises:
            RuntimeError: If the connector is not connected.
        """
        if not self.connected:
            raise RuntimeError("Connector is not connected. Call connect() first.")
        pass

    def is_connected(self) -> bool:
        """Check if the connector is currently connected.
        
        Returns:
            True if connected, False otherwise.
        """
        return self.connected