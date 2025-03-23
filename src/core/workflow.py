import re
import uuid

from enum import Enum
from src.utils.helper import _format_value
from typing import Dict, Any, List, Optional, Callable, Union, Set


class WorkflowState(Enum):
    """States for the StateMachineWorkflow state machine."""
    BUILDING = "building"  # Initial state - workflow is being defined
    READY = "ready"        # Workflow is validated and ready to run
    RUNNING = "running"    # Workflow is currently executing
    COMPLETED = "completed"  # Workflow completed successfully
    FAILED = "failed"      # Workflow execution failed


class WorkflowError(Exception):
    """Base exception for workflow errors."""
    pass


class StateTransition:
    """Represents a transition between workflow states."""
    
    def __init__(self, from_state: str, to_state: str, condition=None):
        self.id = str(uuid.uuid4())
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition  # Can be a function or a condition string
    
    def should_transition(self, context: Dict[str, Any]) -> bool:
        """Determine if this transition should be taken based on the condition."""
        if self.condition is None:
            return True
        
        if callable(self.condition):
            return self.condition(context)
        elif isinstance(self.condition, str):
            # Evaluate condition string like "inventory_check.available == false"
            return self._evaluate_condition_string(self.condition, context)
        
        return False
    
    def _evaluate_condition_string(self, condition_str: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string against the context."""
        # Parse the condition string into parts: "state_name.property operation value"
        # For example: "inventory_check.available == false"
        
        # First, replace variables with their context values
        eval_str = self._replace_variables(condition_str, context)
        
        # Safely evaluate the condition
        try:
            # Replace string literals "true" and "false" with Python booleans
            eval_str = eval_str.replace(" == true", " == True").replace(" == false", " == False")
            return eval(eval_str)
        except Exception as e:
            print(f"Error evaluating condition '{condition_str}': {str(e)}")
            return False
    
    def _replace_variables(self, condition_str: str, context: Dict[str, Any]) -> str:
        """Replace variable references with their actual values from context."""
        # Match patterns like "state_name.property"
        var_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)'
        
        def replace_var(match):
            var_path = match.group(1)
            parts = var_path.split('.')
            
            if len(parts) != 2:
                return "None"  # Invalid variable reference
            
            state_name, property_name = parts
            
            # Check if the state exists in context
            if state_name not in context:
                return "None"  # State not found
            
            # Get the state's output values
            state_outputs = context[state_name]
            
            # Check if the property exists in the state's outputs
            if property_name not in state_outputs:
                return "None"  # Property not found
            
            # Return the value
            value = state_outputs[property_name]
            
            # Convert to appropriate string representation
            if isinstance(value, bool):
                return str(value).lower()  # 'true' or 'false'
            elif isinstance(value, str):
                return f'"{value}"'  # Quote strings
            return str(value)
        
        # Replace all variable references
        return re.sub(var_pattern, replace_var, condition_str)

    def inspect(self) -> Dict[str, Any]:
        """
        Get detailed information about this transition.
        
        Returns:
            Dictionary with transition information
        """
        return {
            'id': self.id,
            'from_state': self.from_state,
            'to_state': self.to_state,
            'has_condition': self.condition is not None,
            'condition_type': type(self.condition).__name__ if self.condition else None,
            'condition_str': str(self.condition) if isinstance(self.condition, str) else None
        }


class StateDefinition:
    """Represents a state in the workflow state machine with its node and transitions."""
    
    def __init__(self, name: str, node, is_start: bool = False, is_terminal: bool = False):
        self.name = name
        self.node = node
        self.is_start = is_start
        self.is_terminal = is_terminal
        self.transitions = []
    
    def add_transition(self, transition: StateTransition):
        """Add a transition from this state."""
        self.transitions.append(transition)
    
    def get_next_state(self, context: Dict[str, Any]) -> Optional[str]:
        """Determine the next state based on transitions and context."""
        for transition in self.transitions:
            if transition.should_transition(context):
                return transition.to_state
        
        return None
    
    def inspect(self, include_node: bool = False) -> Dict[str, Any]:
        """
        Get detailed information about this state.
        
        Args:
            include_node: Whether to include detailed node information
            
        Returns:
            Dictionary with state information
        """
        info = {
            'name': self.name,
            'is_start': self.is_start,
            'is_terminal': self.is_terminal,
            'node_type': self.node.__class__.__name__,
            'node_name': self.node.name,
            'transitions': [t.inspect() for t in self.transitions]
        }
        
        if include_node:
            info['node'] = self.node.inspect()
        
        return info


class WhenBuilder:
    """Builder for conditional transitions using the 'when' syntax."""
    
    def __init__(self, workflow_builder, current_state: str, condition: str):
        self.workflow_builder = workflow_builder
        self.current_state = current_state
        self.condition = condition
    
    def then(self, next_state: str):
        """Specify the target state for this conditional transition."""
        # Create a new transition with the condition
        transition = StateTransition(
            from_state=self.current_state,
            to_state=next_state,
            condition=self.condition
        )
        
        # Add the transition to the current state
        self.workflow_builder.states[self.current_state].add_transition(transition)
        
        # Return the workflow builder for continued chaining
        return self.workflow_builder


class Workflow:
    """
    Builder for defining workflow state machines with a fluent API.
    
    Example usage:
    workflow = (WorkflowBuilder("Order Processing")
               .version("1.0.0")
               .description("Process customer orders from receipt to delivery")
               .state("order_received", OrderReceivedNode())
                 .start()
                 .next("inventory_check")
               .state("inventory_check", InventoryCheckNode())
                 .when("inventory_check.available == false").then("order_cancelled")
                 .when("inventory_check.available == true").then("packing")
               .build())
    """
    
    def __init__(self, name: str):
        self.name = name
        self._version = "1.0.0"
        self._description = ""
        self.states = {}  # Dictionary of state_name -> StateDefinition
        self.current_state_name = None  # For tracking the state being defined
        self.start_state = None
        self._build_status = WorkflowState.BUILDING
    
    def version(self, version: str):
        """
        Set the workflow version.
        
        Returns:
            Self for method chaining
        """
        self._version = version
        return self
    
    def description(self, description: str):
        """
        Set the workflow description.
        
        Returns:
            Self for method chaining
        """
        self._description = description
        return self
    
    def state(self, state_name: str, node):
        """
        Define a new state in the workflow.
        
        Args:
            state_name: Name of the state
            node: The node to execute when in this state
        
        Returns:
            Self for method chaining
        """
        self.states[state_name] = StateDefinition(state_name, node)
        self.current_state_name = state_name
        return self
    
    def start(self):
        """
        Mark the current state as the starting state.
        
        Returns:
            Self for method chaining
        """
        if self.current_state_name:
            self.states[self.current_state_name].is_start = True
            self.start_state = self.current_state_name
        return self
    
    def terminal(self):
        """
        Mark the current state as a terminal state.
        
        Returns:
            Self for method chaining
        """
        if self.current_state_name and self.current_state_name in self.states:
            self.states[self.current_state_name].is_terminal = True
        return self
    
    def next(self, next_state: str):
        """
        Define an unconditional transition to the next state.
        
        Args:
            next_state: Name of the next state
        
        Returns:
            Self for method chaining
        """
        if self.current_state_name and self.current_state_name in self.states:
            transition = StateTransition(
                from_state=self.current_state_name,
                to_state=next_state
            )
            self.states[self.current_state_name].add_transition(transition)
        return self
    
    def when(self, condition: str):
        """
        Begin defining a conditional transition.
        
        Args:
            condition: A condition string like "inventory_check.available == false"
        
        Returns:
            WhenBuilder for specifying the transition target
        """
        return WhenBuilder(self, self.current_state_name, condition)
    
    def _validate_workflow(self):
        """
        Validate the workflow definition.
        
        Raises:
            WorkflowError: If the workflow is invalid
        """
        # Ensure we have at least one state
        if not self.states:
            raise WorkflowError("Workflow must have at least one state")
        
        # Ensure we have a start state
        if not self.start_state:
            raise WorkflowError("Workflow must have a start state")
        
        # Ensure all referenced states exist
        for state_name, state in self.states.items():
            for transition in state.transitions:
                if transition.to_state not in self.states:
                    raise WorkflowError(f"State '{state_name}' transitions to non-existent state '{transition.to_state}'")
        
        # Detect cycles to prevent infinite loops
        if self._detect_cycles():
            raise WorkflowError("Workflow contains cycles, which must be handled explicitly")
        
        # Ensure terminal states don't have outgoing transitions
        for state_name, state in self.states.items():
            if state.is_terminal and state.transitions:
                raise WorkflowError(f"Terminal state '{state_name}' has outgoing transitions")
    
    def _detect_cycles(self) -> bool:
        """
        Detect cycles in the workflow graph using depth-first search.
        
        Returns:
            True if cycles are detected, False otherwise
        """
        visited = set()
        path = set()
        
        def visit(state_name):
            if state_name in path:
                return True  # Cycle detected
            if state_name in visited:
                return False
            
            path.add(state_name)
            visited.add(state_name)
            
            state = self.states[state_name]
            for transition in state.transitions:
                if visit(transition.to_state):
                    return True
            
            path.remove(state_name)
            return False
        
        return any(visit(state_name) for state_name in self.states if state_name not in visited)
    
    def build(self):
        """
        Build and return the completed workflow.
        
        Returns:
            StateMachineWorkflow instance
        """
        # Validate the workflow
        self._validate_workflow()
        
        # Create and return the workflow
        workflow = StateMachineWorkflow(
            name=self.name,
            version=self._version,
            description=self._description,
            states=self.states,
            start_state=self.start_state
        )
        
        self._build_status = WorkflowState.READY
        return workflow
        
    #
    # Workflow Inspection Methods
    #
    
    def get_states(self) -> List[str]:
        """
        Get all state names in the workflow.
        
        Returns:
            List of state names
        """
        return list(self.states.keys())
    
    def get_state_info(self, state_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a state or all states.
        
        Args:
            state_name: Name of the state to inspect, or None for all states
            
        Returns:
            Dictionary with state information
        """
        if state_name is not None:
            if state_name not in self.states:
                raise ValueError(f"State '{state_name}' not found in workflow")
            
            state = self.states[state_name]
            return {
                'name': state.name,
                'node_type': state.node.__class__.__name__,
                'node_name': state.node.name,
                'is_start': state.is_start,
                'is_terminal': state.is_terminal,
                'transitions': [
                    {
                        'to_state': transition.to_state,
                        'has_condition': transition.condition is not None
                    }
                    for transition in state.transitions
                ]
            }
        else:
            # Return info for all states
            return {
                name: {
                    'node_type': state.node.__class__.__name__,
                    'node_name': state.node.name,
                    'is_start': state.is_start,
                    'is_terminal': state.is_terminal,
                    'transitions_count': len(state.transitions)
                }
                for name, state in self.states.items()
            }
    
    def get_node(self, state_name: str):
        """
        Get the node object for a specific state.
        
        Args:
            state_name: Name of the state
            
        Returns:
            The node object for the specified state
        """
        if state_name not in self.states:
            raise ValueError(f"State '{state_name}' not found in workflow")
        
        return self.states[state_name].node
    
    def get_input_schema(self, state_name: str) -> Dict[str, List[str]]:
        """
        Get the input schema for a specific state's node.
        
        Args:
            state_name: Name of the state
            
        Returns:
            Dictionary with required and optional input keys
        """
        node = self.get_node(state_name)
        return {
            'required': node.input_schema.required_keys,
            'optional': node.input_schema.optional_keys
        }
    
    def get_output_schema(self, state_name: str) -> List[str]:
        """
        Get the output schema for a specific state's node.
        
        Args:
            state_name: Name of the state
            
        Returns:
            List of output keys
        """
        node = self.get_node(state_name)
        return node.output_schema.keys
    
    def get_all_inputs(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get the input schemas for all states in the workflow.
        
        Returns:
            Dictionary mapping state names to their input schemas
        """
        return {
            state_name: self.get_input_schema(state_name)
            for state_name in self.states
        }
    
    def get_all_outputs(self) -> Dict[str, List[str]]:
        """
        Get the output schemas for all states in the workflow.
        
        Returns:
            Dictionary mapping state names to their output schemas
        """
        return {
            state_name: self.get_output_schema(state_name)
            for state_name in self.states
        }
    
    def get_transition_graph(self) -> Dict[str, List[str]]:
        """
        Get a graph representation of state transitions.
        
        Returns:
            Dictionary mapping state names to lists of target state names
        """
        graph = {}
        for state_name, state in self.states.items():
            graph[state_name] = [t.to_state for t in state.transitions]
        return graph
    
    def print_workflow_structure(self):
        """Print a formatted overview of the workflow structure."""
        print(f"\n=== {self.name} v{self._version} ===")
        print(f"Description: {self._description}")
        print(f"Build Status: {self._build_status.value}")
        print("\nStates:")
        
        for state_name, state in self.states.items():
            is_start = "(START)" if state.is_start else ""
            is_terminal = "(TERMINAL)" if state.is_terminal else ""
            status = f"{is_start} {is_terminal}".strip()
            
            print(f"  - {state_name} [{state.node.__class__.__name__}] {status}")
            
            # Show transitions
            if state.transitions:
                print("    Transitions:")
                for transition in state.transitions:
                    condition = " (conditional)" if transition.condition else ""
                    print(f"      → {transition.to_state}{condition}")
            
            # Show input requirements
            input_schema = self.get_input_schema(state_name)
            if input_schema['required']:
                print(f"    Required inputs: {', '.join(input_schema['required'])}")
            if input_schema['optional']:
                print(f"    Optional inputs: {', '.join(input_schema['optional'])}")
            
            # Show output
            output_schema = self.get_output_schema(state_name)
            if output_schema:
                print(f"    Outputs: {', '.join(output_schema)}")
            
            print()  # Empty line between states
    
    def get_status(self) -> WorkflowState:
        """Get the current status of the workflow."""
        return self._build_status
    
    def inspect(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Get a complete overview of the workflow definition.
        
        Args:
            detailed: Whether to include detailed node information
            
        Returns:
            Dictionary with workflow information
        """
        info = {
            'name': self.name,
            'version': self._version,
            'description': self._description,
            'status': self._build_status.value,
            'states_count': len(self.states),
            'start_state': self.start_state,
            'states': {}
        }
        
        for state_name, state in self.states.items():
            info['states'][state_name] = state.inspect(include_node=detailed)
        
        return info


class StateMachineWorkflow:
    """
    A workflow that operates as a state machine.
    
    The workflow executes nodes as it transitions through states,
    with transitions determined by node outputs and conditions.
    """
    
    def __init__(self, name: str, states: Dict[str, StateDefinition], 
                start_state: str, version: str = "1.0.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self.states = states
        self.start_state = start_state
        self.current_state_name = None
        self.context = {}  # Stores state outputs and workflow context
        self.status = WorkflowState.READY
        self.execution_path = []  # Track the sequence of states visited
    
    def execute(self, initial_context: Dict[str, Any] = None):
        """
        Execute the workflow from the start state until a terminal state or no transition.
        
        Args:
            initial_context: Initial context values
            
        Returns:
            The final context dictionary
        """
        if initial_context:
            self.context.update(initial_context)
        
        self.status = WorkflowState.RUNNING
        self.current_state_name = self.start_state
        self.execution_path = []  # Reset execution path
        
        while self.status == WorkflowState.RUNNING and self.current_state_name:
            current_state = self.states[self.current_state_name]
            
            # Record this state in the execution path
            self.execution_path.append(self.current_state_name)
            
            # Execute the node for this state
            self._execute_node(current_state.node)
            
            # Check if this is a terminal state
            if current_state.is_terminal:
                self.status = WorkflowState.COMPLETED
                break
            
            # Determine the next state
            next_state = current_state.get_next_state(self.context)
            
            if next_state:
                print(f"Transitioning from '{self.current_state_name}' to '{next_state}'")
                self.current_state_name = next_state
            else:
                print(f"No valid transition from state '{self.current_state_name}'")
                self.status = WorkflowState.FAILED
                break
        
        return self.context
    
    def _execute_node(self, node):
        """
        Execute a node and update the context with its outputs.
        
        Args:
            node: The node to execute
        """
        print(f"Executing node: {node.name}")
        
        # Get inputs for this node based on its InputSchema
        node_inputs = self._get_inputs_for_node(node)
        
        # Run the node with the inputs
        node.run(**node_inputs)
        
        # Update context with node outputs
        self.context[self.current_state_name] = node.get_outputs()
    
    def _get_inputs_for_node(self, node) -> Dict[str, Any]:
        """
        Get inputs for a node based on its InputSchema.
        
        Automatically maps available context values to required inputs.
        
        Args:
            node: The node to get inputs for
            
        Returns:
            Dictionary of input values for the node
        """
        inputs = {}
        
        # Get the node's required and optional input keys
        required_keys = node.input_schema.required_keys
        optional_keys = node.input_schema.optional_keys
        all_input_keys = set(required_keys + optional_keys)
        
        # For each input key, try to find a matching output in the context
        for input_key in all_input_keys:
            for state_name, outputs in self.context.items():
                if input_key in outputs:
                    inputs[input_key] = outputs[input_key]
                    break
        
        # Check if all required inputs are available
        missing_inputs = [key for key in required_keys if key not in inputs]
        if missing_inputs:
            print(f"Warning: Missing required inputs for node {node.name}: {missing_inputs}")
        
        return inputs
    
    def get_status(self) -> WorkflowState:
        """
        Get the current status of the workflow.
        
        Returns:
            The current workflow state
        """
        return self.status
    
    def __str__(self) -> str:
        """String representation of the workflow."""
        return f"{self.name} v{self.version} ({len(self.states)} states, status: {self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed representation of the workflow."""
        return f"StateMachineWorkflow({self.name}, {len(self.states)} states, status: {self.status.value})"
        
    #
    # Workflow Execution Inspection Methods
    #
    
    def get_states(self) -> List[str]:
        """Get all state names in the workflow."""
        return list(self.states.keys())
    
    def get_node(self, state_name: str):
        """Get the node object for a specific state."""
        if state_name not in self.states:
            raise ValueError(f"State '{state_name}' not found in workflow")
        
        return self.states[state_name].node
    
    def get_input_schema(self, state_name: str) -> Dict[str, List[str]]:
        """Get the input schema for a specific state's node."""
        node = self.get_node(state_name)
        return {
            'required': node.input_schema.required_keys,
            'optional': node.input_schema.optional_keys
        }
    
    def get_output_schema(self, state_name: str) -> List[str]:
        """Get the output schema for a specific state's node."""
        node = self.get_node(state_name)
        return node.output_schema.keys
    
    def get_execution_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get execution results for all states that have been executed.
        
        Returns:
            Dictionary mapping state names to their execution results
        """
        return self.context
    
    def get_state_result(self, state_name: str) -> Dict[str, Any]:
        """
        Get execution results for a specific state.
        
        Args:
            state_name: Name of the state
            
        Returns:
            Dictionary with the state's execution results
        """
        if state_name not in self.context:
            return {}
        
        return self.context[state_name]
    
    def get_output_value(self, state_name: str, output_key: str) -> Any:
        """
        Get a specific output value from a state.
        
        Args:
            state_name: Name of the state
            output_key: Key of the output value to retrieve
            
        Returns:
            The output value, or None if not found
        """
        state_result = self.get_state_result(state_name)
        return state_result.get(output_key)
    
    def get_execution_path(self) -> List[str]:
        """
        Get the sequence of states visited during execution.
        
        Returns:
            List of state names in execution order
        """
        return self.execution_path
    
    def print_execution_summary(self):
        """Print a summary of the workflow execution."""
        print(f"\n=== Execution Summary for {self.name} v{self.version} ===")
        print(f"Status: {self.status.value}")
        
        if self.execution_path:
            print(f"Execution path: {' → '.join(self.execution_path)}")
        
        if not self.context:
            print("No execution data available.")
            return
        
        print("\nState execution results:")
        for state_name, results in self.context.items():
            print(f"  - {state_name}:")
            for key, value in results.items():
                # Truncate long values for display
                if isinstance(value, list) and len(value) > 3:
                    print(f"    {key}: List with {len(value)} items")
                    for i, item in enumerate(value[:3]):
                        print(f"      [{i}] {_format_value(item)}")
                    print(f"      ... {len(value) - 3} more items")
                else:
                    print(f"    {key}: {_format_value(value)}")
            print()
    
    def get_workflow_data(self) -> Dict[str, Any]:
        """
        Get a complete overview of the workflow, including structure and execution data.
        
        Returns:
            Dictionary with workflow information
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'status': self.status.value,
            'start_state': self.start_state,
            'current_state': self.current_state_name,
            'states': self.get_states(),
            'execution_path': self.execution_path,
            'execution_results': self.get_execution_results()
        }
    
    def print_workflow_structure(self):
        """Print a formatted overview of the workflow structure."""
        print(f"\n=== {self.name} v{self.version} ===")
        print(f"Description: {self.description}")
        print(f"Status: {self.status.value}")
        print("\nStates:")
        
        for state_name, state in self.states.items():
            is_start = "(START)" if state.is_start else ""
            is_terminal = "(TERMINAL)" if state.is_terminal else ""
            is_current = "(CURRENT)" if state_name == self.current_state_name else ""
            status = f"{is_start} {is_terminal} {is_current}".strip()
            
            # Check if this state has been executed
            executed = state_name in self.context
            executed_str = "[EXECUTED]" if executed else ""
            
            print(f"  - {state_name} [{state.node.__class__.__name__}] {status} {executed_str}")
            
            # Show transitions
            if state.transitions:
                print("    Transitions:")
                for transition in state.transitions:
                    condition = " (conditional)" if transition.condition else ""
                    print(f"      → {transition.to_state}{condition}")
            
            # Show execution status
            if executed:
                print("    Execution:")
                result_keys = list(self.context[state_name].keys())
                print(f"    Output keys: {', '.join(result_keys)}")
            
            print()  # Empty line between states
    
    def visualize(self):
        """
        Generate a text-based visualization of the workflow.
        
        Returns:
            String containing a text visualization of the workflow
        """
        lines = [f"Workflow: {self.name} v{self.version}"]
        lines.append("Status: " + self.status.value)
        lines.append("")
        
        # Find the start state
        if self.start_state:
            lines.append(f"START → {self.start_state}")
        
        # Visualize each state and its transitions
        for state_name, state in self.states.items():
            executed = state_name in self.context
            prefix = "* " if executed else "  "
            
            if state.is_terminal:
                lines.append(f"{prefix}{state_name} → END")
            else:
                for transition in state.transitions:
                    condition = f" [{transition.condition}]" if transition.condition else ""
                    lines.append(f"{prefix}{state_name} →{condition} {transition.to_state}")
        
        return "\n".join(lines)