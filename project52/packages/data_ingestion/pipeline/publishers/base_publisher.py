from abc import ABC, abstractmethod
from typing import Any, Dict
from core.base_pipeline_component import BasePipelineComponent

class BasePublisher(BasePipelineComponent, ABC):
    """Abstract base class for all publishers in the data ingestion platform.
    
    Publishers are responsible for sending the processed data to its final destination,
    such as a database, message queue, or file system.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the publisher with a configuration dictionary.
        
        Args:
            config: Dictionary containing publisher-specific settings.
        """
        super().__init__(config)

    @abstractmethod
    def publish(self, data: Any) -> Dict[str, Any]:
        """Publish the data to the destination.
        
        This method must be implemented by all subclasses to define the specific
        publishing logic.
        
        Args:
            data: The data to be published.
        
        Returns:
            A dictionary containing status information about the publishing operation.
        """
        pass

    def process(self, data: Any) -> Dict[str, Any]:
        """Process method to integrate with the pipeline.
        
        This method calls the publish method and returns the status.
        
        Args:
            data: The data to be published.
        
        Returns:
            A dictionary containing status information about the publishing operation.
        """
        return self.publish(data)