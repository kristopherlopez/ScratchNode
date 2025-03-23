import inspect
import importlib
import pkgutil
from typing import Dict, Type, List, Optional
from src.core.node import BaseNode


class NodeRegistry:
    """
    Registry for workflow nodes.
    Provides discovery and lookup capabilities for all node types.
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure only one registry exists."""
        if cls._instance is None:
            cls._instance = super(NodeRegistry, cls).__new__(cls)
            cls._instance._nodes = {}
            cls._instance._categories = {}
        return cls._instance

    def register_node(self, node_class: Type[BaseNode], category: str = "General") -> None:
        """
        Register a node class with the registry.

        Args:
            node_class: The node class to register
            category: Optional category to organize nodes
        """
        if not issubclass(node_class, BaseNode):
            raise TypeError(f"{node_class.__name__} is not a subclass of BaseNode")

        node_name = node_class.__name__
        self._nodes[node_name] = node_class

        # Add to category
        if category not in self._categories:
            self._categories[category] = []

        if node_name not in self._categories[category]:
            self._categories[category].append(node_name)

    def get_node_class(self, node_name: str) -> Optional[Type[BaseNode]]:
        """
        Get a node class by name.

        Args:
            node_name: Name of the node class

        Returns:
            The node class, or None if not found
        """
        return self._nodes.get(node_name)

    def get_all_nodes(self) -> Dict[str, Type[BaseNode]]:
        """
        Get all registered node classes.

        Returns:
            Dictionary mapping node names to node classes
        """
        return self._nodes.copy()

    def get_categories(self) -> List[str]:
        """
        Get all node categories.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def get_nodes_in_category(self, category: str) -> List[str]:
        """
        Get all node names in a category.

        Args:
            category: Category name

        Returns:
            List of node names in the category
        """
        return self._categories.get(category, [])

    def discover_nodes(self, package_name: str = "src.nodes") -> int:
        """
        Automatically discover and register node classes in a package.

        Args:
            package_name: Name of the package to scan for nodes

        Returns:
            Number of nodes discovered
        """
        count = 0
        package = importlib.import_module(package_name)

        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            if is_pkg:
                # Recursively discover nodes in subpackages
                count += self.discover_nodes(module_name)
            else:
                # Import the module and register any node classes
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                                issubclass(obj, BaseNode) and
                                obj.__module__ == module_name and
                                obj != BaseNode):
                            # Determine category from module path
                            parts = module_name.split('.')
                            category = parts[-2] if len(parts) > 2 else "General"

                            # Register the node
                            self.register_node(obj, category)
                            count += 1
                except Exception as e:
                    print(f"Error discovering nodes in {module_name}: {str(e)}")

        return count

    def print_catalog(self) -> None:
        """Print a catalog of all registered nodes by category."""
        print("\n=== Node Catalog ===\n")

        for category in sorted(self._categories.keys()):
            print(f"Category: {category}")

            for node_name in sorted(self._categories[category]):
                node_class = self._nodes[node_name]
                description = node_class.__doc__.split('\n')[0].strip() if node_class.__doc__ else "No description"
                print(f"  - {node_name}: {description}")

            print()  # Empty line between categories

    def get_node_documentation(self, node_name: str) -> str:
        """
        Get detailed documentation for a node.

        Args:
            node_name: Name of the node class

        Returns:
            Formatted documentation string
        """
        node_class = self.get_node_class(node_name)
        if not node_class:
            return f"Node '{node_name}' not found in registry"

        docs = []
        docs.append(f"=== {node_name} ===")

        # Class docstring
        if node_class.__doc__:
            docs.append(node_class.__doc__.strip())
        else:
            docs.append("No documentation available")

        # Try to create an instance to get schema information
        try:
            instance = node_class()
            docs.append("\nInput Schema:")
            if instance.input_schema.required_keys:
                docs.append(f"  Required: {', '.join(instance.input_schema.required_keys)}")
            if instance.input_schema.optional_keys:
                docs.append(f"  Optional: {', '.join(instance.input_schema.optional_keys)}")

            docs.append("\nOutput Schema:")
            if instance.output_schema.keys:
                docs.append(f"  Keys: {', '.join(instance.output_schema.keys)}")
        except Exception as e:
            docs.append(f"\nCould not inspect schema: {str(e)}")

        return "\n".join(docs)


# Example usage
def register_all_nodes():
    """Discover and register all node types in the project."""
    registry = NodeRegistry()
    count = registry.discover_nodes()
    print(f"Discovered {count} node types")
    return registry


# Decorator for easy node registration
def register_node(category: str = "General"):
    """
    Decorator to register a node class with the registry.

    Args:
        category: Category to place the node in

    Example:
        @register_node(category="IO")
        class MyNode(BaseNode):
            ...
    """

    def decorator(cls):
        NodeRegistry().register_node(cls, category)
        return cls

    return decorator