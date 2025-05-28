from abc import abstractmethod
from typing import Any, Dict, Generator
from core.base_connector import BaseConnector
import logging

class BaseDatabaseConnector(BaseConnector):
    """Abstract base class for database connectors.
    
    This class provides a common interface for connecting to and fetching data from databases.
    Subclasses must implement database-specific logic for creating connections and executing queries.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the database connector with a configuration dictionary.
        
        Args:
            config: Dictionary containing database connection settings.
        """
        super().__init__(config)
        self.connection = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def _create_connection(self) -> Any:
        """Create and return a database connection object.
        
        This method must be implemented by subclasses to handle database-specific connection logic.
        
        Returns:
            The database connection object.
        """
        pass

    @abstractmethod
    def _execute_query(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """Execute a query on the database and yield the results row by row.
        
        This method must be implemented by subclasses to handle database-specific query execution.
        
        Args:
            query: The SQL query to execute.
        
        Yields:
            Dictionaries representing each row of the query result.
        """
        pass

    def connect(self) -> None:
        """Establish a connection to the database.
        
        This method creates a connection using the subclass's _create_connection method and sets the connected flag.
        
        Raises:
            ConnectionError: If the connection cannot be established.
        """
        try:
            self.connection = self._create_connection()
            self.connected = True
            self.logger.info("Successfully connected to the database.")
        except Exception as e:
            self.logger.error(f"Failed to connect to the database: {e}")
            raise ConnectionError(f"Could not connect to the database: {e}")

    def disconnect(self) -> None:
        """Close the connection to the database.
        
        This method closes the connection if it is open and resets the connected flag.
        """
        if self.connection:
            try:
                self.connection.close()
                self.logger.info("Disconnected from the database.")
            except Exception as e:
                self.logger.error(f"Error while disconnecting: {e}")
            finally:
                self.connection = None
                self.connected = False

    def fetch_data(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """Fetch data from the database by executing the given query.
        
        This method yields rows one by one as dictionaries.
        
        Args:
            query: The SQL query to execute.
        
        Yields:
            Dictionaries representing each row of the query result.
        
        Raises:
            RuntimeError: If the connector is not connected.
        """
        if not self.is_connected():
            raise RuntimeError("Connector is not connected. Call connect() first.")
        try:
            yield from self._execute_query(query)
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            raise