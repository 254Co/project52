from abc import ABC, abstractmethod
from typing import Any, Dict
from core.base_pipeline_component import BasePipelineComponent

class BaseTransformer(BasePipelineComponent, ABC):
    """Abstract base class for all transformers in the data ingestion platform.
    
    Transformers are responsible for processing or transforming data fetched from sources.
    This class defines the interface that all transformers must implement.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the transformer with a configuration dictionary.
        
        Args:
            config: Dictionary containing transformer-specific settings.
        """
        super().__init__(config)

    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transform the input data and return the result.
        
        This method must be implemented by all subclasses to define the specific
        data transformation logic.
        
        Args:
            data: The input data to be transformed.
        
        Returns:
            The transformed data.
        
        Raises:
            Exception: If an error occurs during transformation.
        """
        pass

    def process(self, data: Any) -> Any:
        """Process method to integrate with the pipeline.
        
        This method calls the transform method and returns the result.
        
        Args:
            data: The input data to be transformed.
        
        Returns:
            The transformed data.
        """
        return self.transform(data)