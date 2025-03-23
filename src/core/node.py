import uuid
from src.utils.helper import _format_value
from typing import Dict, Any, Optional, List, Set, Union

class InputSchema:
    """
    Schema for validating node inputs.
    
    Attributes:
        required_keys: List of keys that must be present in inputs
        optional_keys: List of keys that may be present in inputs
    """
    def __init__(self, required_keys: List[str] = None, optional_keys: List[str] = None):
        self.required_keys = required_keys or []
        self.optional_keys = optional_keys or []
        
    def validate(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate inputs against schema requirements.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            bool: True if all required keys are present, False otherwise
        """
        return all(key in inputs for key in self.required_keys)
    
    def get_all_keys(self) -> Set[str]:
        """
        Get all keys (both required and optional).
        
        Returns:
            Set of all valid input keys
        """
        return set(self.required_keys + self.optional_keys)

class OutputSchema:
    """
    Schema for validating node outputs.
    
    Attributes:
        keys: List of keys expected in the outputs
    """
    def __init__(self, keys: List[str] = None):
        self.keys = keys or []
        
    def validate(self, outputs: Dict[str, Any]) -> bool:
        """
        Validate outputs against schema.
        
        Args:
            outputs: Dictionary of output values
            
        Returns:
            bool: True if all expected keys are present, False otherwise
        """
        return all(key in outputs for key in self.keys)

class BaseNode:
    """
    Base class for all nodes in the workflow system.
    Nodes are fully declarative and can be connected to form data processing pipelines.
    """
    
    def __init__(self, name=None, description=None, execute_mode=None, **inputs):
        """
        Initialize node with both configuration parameters and inputs.
        
        Args:
            name: Name of the node
            description: Description of the node's purpose
            execute_mode: Execution mode for the node
            **inputs: Input values passed directly to the node
        """
        self.id = str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.description = description
        self.execute_mode = execute_mode
        self._inputs = inputs  # All kwargs beyond the explicit ones become inputs
        self._outputs = {}
        self._define_schemas()
    
    def _define_schemas(self):
        """Define input and output schemas. Override in subclasses."""
        self.input_schema = InputSchema()
        self.output_schema = OutputSchema()
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate inputs against schema.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            bool: True if inputs are valid
        """
        return self.input_schema.validate(inputs)
    
    def validate_outputs(self, outputs: Dict[str, Any]) -> bool:
        """
        Validate outputs against schema.
        
        Args:
            outputs: Dictionary of output values
            
        Returns:
            bool: True if outputs are valid
        """
        return self.output_schema.validate(outputs)
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the node's operation with the given inputs.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Dictionary of output values
        """
        raise NotImplementedError("Nodes must implement the process method")
    
    def set_inputs(self, **kwargs) -> 'BaseNode':
        """
        Set additional input values for the node.
        
        Args:
            **kwargs: Input values to set
            
        Returns:
            Self for method chaining
        """
        self._inputs.update(kwargs)
        return self
    
    def get_outputs(self) -> Dict[str, Any]:
        """
        Get the node's current output values.
        
        Returns:
            Dictionary of output values
        """
        return self._outputs
    
    def get_output(self, key: str) -> Any:
        """
        Get a specific output value by key.
        
        Args:
            key: The output key to retrieve
            
        Returns:
            The output value for the specified key
        """
        return self._outputs.get(key)
    
    def connect_to(self, target_node: 'BaseNode', input_mappings: Dict[str, str] = None) -> 'BaseNode':
        """
        Connect this node to another node by mapping outputs to inputs.
        
        Args:
            target_node: The node to connect to
            input_mappings: Dictionary mapping this node's output keys to target node's input keys
                           If None, attempt to match by key name
        
        Returns:
            The target node for chaining
        """
        if input_mappings is None:
            # Default behavior: match keys with the same name
            all_target_keys = target_node.input_schema.get_all_keys()
            input_mappings = {key: key for key in self._outputs.keys() 
                             if key in all_target_keys}
        
        # Transfer outputs to target's inputs based on mappings
        target_inputs = {target_key: self._outputs[source_key] 
                        for source_key, target_key in input_mappings.items()
                        if source_key in self._outputs}
        
        target_node.set_inputs(**target_inputs)
        return target_node
    
    def run(self, **kwargs) -> 'BaseNode':
        """
        Run the node with the given inputs.
        
        Args:
            **kwargs: Additional input values that will be merged with any previously set inputs
            
        Returns:
            Self for method chaining
        """
        # Combine existing inputs with provided kwargs
        combined_inputs = {**self._inputs, **kwargs}
        
        # Validate inputs
        if not self.validate_inputs(combined_inputs):
            raise ValueError(f"Invalid inputs for node {self.name}. Required keys: {self.input_schema.required_keys}")
        
        # Process the inputs
        self._outputs = self.process(combined_inputs)
        
        # Validate outputs
        if not self.validate_outputs(self._outputs):
            raise ValueError(f"Invalid outputs from node {self.name}. Expected keys: {self.output_schema.keys}")
        
        return self
        
    #
    # Node Inspection Methods 
    #
    
    def inspect(self, verbose=False) -> Dict[str, Any]:
        """
        Get detailed information about this node.
        
        Args:
            verbose: Whether to include detailed input/output data
            
        Returns:
            Dictionary with node information
        """
        info = {
            'id': self.id,
            'type': self.__class__.__name__,
            'name': self.name,
            'description': self.description,
            'execute_mode': self.execute_mode,
            'input_schema': {
                'required': self.input_schema.required_keys,
                'optional': self.input_schema.optional_keys
            },
            'output_schema': self.output_schema.keys,
        }
        
        if verbose:
            info['inputs'] = self._inputs
            info['outputs'] = self._outputs
        
        return info
    
    def get_input_values(self) -> Dict[str, Any]:
        """
        Get the current input values.
        
        Returns:
            Dictionary of input values
        """
        return self._inputs
    
    def print_schema(self):
        """Print the input and output schema for this node."""
        print(f"\n=== {self.name} ({self.__class__.__name__}) ===")
        if self.description:
            print(f"Description: {self.description}")
        
        print("\nInput Schema:")
        if self.input_schema.required_keys:
            print(f"  Required: {', '.join(self.input_schema.required_keys)}")
        if self.input_schema.optional_keys:
            print(f"  Optional: {', '.join(self.input_schema.optional_keys)}")
        if not self.input_schema.required_keys and not self.input_schema.optional_keys:
            print("  No inputs required")
        
        print("\nOutput Schema:")
        if self.output_schema.keys:
            print(f"  Keys: {', '.join(self.output_schema.keys)}")
        else:
            print("  No outputs defined")
            
        print("\nCurrent Status:")
        print(f"  Inputs set: {len(self._inputs)}")
        print(f"  Outputs available: {len(self._outputs)}")
    
    def summarize_inputs(self):
        """Print a summary of the current input values."""
        print(f"\n=== Input Summary for {self.name} ===")
        
        if not self._inputs:
            print("No inputs set")
            return
        
        for key, value in self._inputs.items():
            value_str = _format_value(value)
            print(f"  {key}: {value_str}")
    
    def summarize_outputs(self):
        """Print a summary of the current output values."""
        print(f"\n=== Output Summary for {self.name} ===")
        
        if not self._outputs:
            print("No outputs available (node may not have been run)")
            return
        
        for key, value in self._outputs.items():
            value_str = _format_value(value)
            print(f"  {key}: {value_str}")
