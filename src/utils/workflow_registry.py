import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.core.workflow import StateMachineWorkflow


class WorkflowRegistry:
    """
    Registry for workflow definitions.
    Allows for saving, loading, and discovering workflow templates.
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure only one registry exists."""
        if cls._instance is None:
            cls._instance = super(WorkflowRegistry, cls).__new__(cls)
            cls._instance._workflows = {}
            cls._instance._categories = {}
            cls._instance._storage_dir = os.environ.get("WORKFLOW_STORAGE_DIR", "workflows")
        return cls._instance

    def register_workflow(self, workflow_def: Dict[str, Any], category: str = "General") -> str:
        """
        Register a workflow definition with the registry.

        Args:
            workflow_def: The workflow definition dictionary
            category: Optional category to organize workflows

        Returns:
            ID of the registered workflow
        """
        # Generate an ID if not provided
        workflow_id = workflow_def.get('id', str(uuid.uuid4()))
        workflow_def['id'] = workflow_id

        # Add metadata
        workflow_def['registered_at'] = datetime.now().isoformat()

        # Store the workflow
        self._workflows[workflow_id] = workflow_def

        # Add to category
        if category not in self._categories:
            self._categories[category] = []

        if workflow_id not in self._categories[category]:
            self._categories[category].append(workflow_id)

        return workflow_id

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a workflow definition by ID.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Workflow definition dictionary, or None if not found
        """
        return self._workflows.get(workflow_id)

    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete a workflow from the registry.

        Args:
            workflow_id: ID of the workflow

        Returns:
            True if deleted, False if not found
        """
        if workflow_id not in self._workflows:
            return False

        # Remove from workflows
        del self._workflows[workflow_id]

        # Remove from categories
        for category in self._categories:
            if workflow_id in self._categories[category]:
                self._categories[category].remove(workflow_id)

        return True

    def get_all_workflows(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered workflow definitions.

        Returns:
            Dictionary mapping workflow IDs to definitions
        """
        return self._workflows.copy()

    def get_categories(self) -> List[str]:
        """
        Get all workflow categories.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def get_workflows_in_category(self, category: str) -> List[str]:
        """
        Get all workflow IDs in a category.

        Args:
            category: Category name

        Returns:
            List of workflow IDs in the category
        """
        return self._categories.get(category, [])

    def save_to_disk(self, workflow_id: str = None) -> None:
        """
        Save workflows to disk.

        Args:
            workflow_id: Optional ID of specific workflow to save,
                         or None to save all workflows
        """
        os.makedirs(self._storage_dir, exist_ok=True)

        if workflow_id:
            # Save a specific workflow
            if workflow_id in self._workflows:
                workflow_def = self._workflows[workflow_id]
                filename = os.path.join(self._storage_dir, f"{workflow_id}.json")
                with open(filename, 'w') as f:
                    json.dump(workflow_def, f, indent=2)
        else:
            # Save all workflows
            for wf_id, workflow_def in self._workflows.items():
                filename = os.path.join(self._storage_dir, f"{wf_id}.json")
                with open(filename, 'w') as f:
                    json.dump(workflow_def, f, indent=2)

    def load_from_disk(self, workflow_id: str = None) -> int:
        """
        Load workflows from disk.

        Args:
            workflow_id: Optional ID of specific workflow to load,
                         or None to load all workflows in the storage directory

        Returns:
            Number of workflows loaded
        """
        if not os.path.exists(self._storage_dir):
            return 0

        count = 0

        if workflow_id:
            # Load a specific workflow
            filename = os.path.join(self._storage_dir, f"{workflow_id}.json")
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    workflow_def = json.load(f)
                    self.register_workflow(workflow_def)
                    count = 1
        else:
            # Load all workflows in the directory
            for filename in os.listdir(self._storage_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(self._storage_dir, filename), 'r') as f:
                            workflow_def = json.load(f)
                            self.register_workflow(workflow_def)
                            count += 1
                    except Exception as e:
                        print(f"Error loading workflow from {filename}: {str(e)}")

        return count

    def print_catalog(self) -> None:
        """Print a catalog of all registered workflows by category."""
        print("\n=== Workflow Catalog ===\n")

        for category in sorted(self._categories.keys()):
            print(f"Category: {category}")

            for workflow_id in sorted(self._categories[category]):
                workflow_def = self._workflows[workflow_id]
                name = workflow_def.get('name', 'Unnamed')
                version = workflow_def.get('version', '1.0.0')
                description = workflow_def.get('description', 'No description')
                print(f"  - {name} v{version} ({workflow_id}): {description}")

            print()  # Empty line between categories

    def serialize_workflow(self, workflow: StateMachineWorkflow) -> Dict[str, Any]:
        """
        Serialize a workflow instance to a definition dictionary.

        Args:
            workflow: StateMachineWorkflow instance

        Returns:
            Serialized workflow definition
        """
        # Basic workflow metadata
        workflow_def = {
            'id': getattr(workflow, 'id', str(uuid.uuid4())),
            'name': workflow.name,
            'version': workflow.version,
            'description': workflow.description,
            'start_state': workflow.start_state,
            'states': {}
        }

        # Serialize each state
        for state_name, state in workflow.states.items():
            state_def = {
                'is_start': state.is_start,
                'is_terminal': state.is_terminal,
                'node_type': state.node.__class__.__name__,
                'node_config': {
                    'name': state.node.name,
                    'description': state.node.description,
                    'execute_mode': state.node.execute_mode,
                    'inputs': state.node.get_input_values()
                },
                'transitions': []
            }

            # Serialize transitions
            for transition in state.transitions:
                transition_def = {
                    'to_state': transition.to_state,
                    'condition': str(transition.condition) if transition.condition else None
                }
                state_def['transitions'].append(transition_def)

            workflow_def['states'][state_name] = state_def

        return workflow_def

    def set_storage_directory(self, directory: str) -> None:
        """
        Set the storage directory for workflow definitions.

        Args:
            directory: Path to the storage directory
        """
        self._storage_dir = directory
        os.makedirs(self._storage_dir, exist_ok=True)


# Example usage
def register_workflow_from_instance(workflow: StateMachineWorkflow, category: str = "General") -> str:
    """
    Register a workflow instance with the registry.

    Args:
        workflow: StateMachineWorkflow instance
        category: Optional category

    Returns:
        ID of the registered workflow
    """
    registry = WorkflowRegistry()
    workflow_def = registry.serialize_workflow(workflow)
    return registry.register_workflow(workflow_def, category)