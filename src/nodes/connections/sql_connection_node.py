from contextlib import contextmanager
from core.node import BaseNode, InputSchema, OutputSchema
from typing import Dict, Any, Optional, List, ContextManager, Generator

import sqlalchemy as sa
import urllib.parse

class MSSQLConnectionNode(BaseNode):
    """
    Node for creating and managing MSSQL database connections.
    Handles connection pooling and provides access to database engines.
    
    Args:
        name: Name of the node
        description: Description of the node's purpose
        server: Database server address
        database: Database name
        username: Optional username for SQL authentication
        password: Optional password for SQL authentication
        trusted_connection: Whether to use Windows Authentication
        environment: Optional environment identifier (for predefined connections)
    
    Example:
        # Create from explicit parameters
        conn_node = MSSQLConnectionNode(
            name="Finance DB Connection",
            server="10.3.0.50",
            database="FinanceDB",
            username="db_user",
            password="db_password"
        )
        
        # Create from predefined environment
        conn_node = MSSQLConnectionNode.from_environment(
            name="BIA Connection",
            environment="bia_trusted"
        )
        
        # Get connection for use
        conn_node.run()
        connection = conn_node.get_output('connection')
    """
    
    # Class level connection pool
    _connection_pool: Dict[str, Any] = {}
    
    def __init__(self, name=None, description=None, **inputs):
        """Initialize the connection node."""
        super().__init__(name=name, description=description, **inputs)
        self.engine = None
        self.connection_string = None
        
        # Initialize connection parameters
        self.server = inputs.get('server')
        self.database = inputs.get('database')
        self.username = inputs.get('username')
        self.password = inputs.get('password')
        self.trusted_connection = inputs.get('trusted_connection', False)
        self.environment = inputs.get('environment')
        
        # If we have all required parameters, initialize the connection string
        if self.server and self.database:
            self.connection_string = self._build_connection_string()
    
    def _define_schemas(self):
        """Define input and output schemas for database connections."""
        self.input_schema = InputSchema(
            required_keys=['server', 'database'],
            optional_keys=['username', 'password', 'trusted_connection', 'environment']
        )
        self.output_schema = OutputSchema(
            keys=['connection', 'engine']
        )
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs to create a database connection.
        
        Args:
            inputs: Dictionary containing:
                - server: Database server address
                - database: Database name
                - username: Optional username for SQL auth
                - password: Optional password for SQL auth
                - trusted_connection: Whether to use Windows Authentication
                - environment: Optional environment name
                
        Returns:
            Dictionary containing:
                - connection: MSSQLConnectionNode instance (self)
                - engine: SQLAlchemy engine object
        """
        # Store connection parameters
        self.server = inputs['server']
        self.database = inputs['database']
        self.username = inputs.get('username')
        self.password = inputs.get('password')
        self.trusted_connection = inputs.get('trusted_connection', False)
        self.environment = inputs.get('environment')
        
        # Build connection string
        self.connection_string = self._build_connection_string()
        
        # Get or create engine
        self.engine = self._get_or_create_engine()
        
        return {
            'connection': self,
            'engine': self.engine
        }
    
    def _build_connection_string(self) -> str:
        """
        Build a connection string for MSSQL.
        
        Returns:
            URL-encoded connection string
        """
        if self.trusted_connection:
            return urllib.parse.quote_plus(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes"
            )
        else:
            return urllib.parse.quote_plus(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password}"
            )
    
    def _get_or_create_engine(self) -> sa.engine.Engine:
        """
        Get an existing engine from the pool or create a new one.
        
        Returns:
            SQLAlchemy engine object
        """
        key = self.environment or self.connection_string
        if not self._connection_pool.get(key):
            self._connection_pool[key] = sa.create_engine(
                f"mssql+pyodbc:///?odbc_connect={self.connection_string}"
            )
        return self._connection_pool[key]
    
    def _ensure_engine(self):
        """Ensure the engine is initialized."""
        if self.engine is None:
            if self.connection_string is None:
                # We need to run the node first
                self.run()
            else:
                # We have a connection string but no engine
                self.engine = self._get_or_create_engine()
    
    @contextmanager
    def connect(self) -> Generator:
        """
        Context manager for database connections.
        
        Yields:
            Active database connection
            
        Example:
            with connection_node.connect() as conn:
                result = conn.execute("SELECT * FROM Table")
        """
        # Make sure we have an engine
        self._ensure_engine()
        
        # Now open a connection
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def close(self) -> None:
        """Close the database connection and remove it from the pool."""
        if self.connection_string is None:
            return
            
        key = self.environment or self.connection_string
        if self._connection_pool.get(key):
            self._connection_pool[key].dispose()
            del self._connection_pool[key]
            self.engine = None
    
    @staticmethod
    def get_secret(secret_name: str) -> str:
        """
        Retrieve a secret from the secrets store.
        
        Args:
            secret_name: Name of the secret to retrieve
            
        Returns:
            Secret value
        """
        # This method should be implemented to retrieve secrets from your KeyVault
        # For now, it's a placeholder
        return f"{{SECRET_{secret_name}}}"
    
    @classmethod
    def from_environment(cls, name: str = None, description: str = None, environment: str = None) -> 'MSSQLConnectionNode':
        """
        Create a connection node from a predefined environment.
        
        Args:
            name: Optional name for the node
            description: Optional description for the node
            environment: Name of the predefined environment
            
        Returns:
            Configured MSSQLConnectionNode instance
            
        Raises:
            ValueError: If the environment is unknown
        """
        env_params = {
            "uat": {
                "server": "10.3.0.50",
                "database": "FinanceDB",
                "username": cls.get_secret('DW-DB-USERNAME'),
                "password": cls.get_secret('DW-DB-PASSWORD'),
                "trusted_connection": False
            },
            "cuida": {
                "server": "petsure-infrastructure.database.windows.net",
                "database": "Cuida",
                "username": cls.get_secret('CUIDA-USERNAME'),
                "password": cls.get_secret('CUIDA-PASSWORD'),
                "trusted_connection": False
            },
            "bia_trusted": {
                "server": "PS-UAT-AZS-DWH1",
                "database": "BIA",
                "trusted_connection": True
            }
        }
        
        if not environment or environment not in env_params:
            raise ValueError(f"Unknown environment: {environment}")
        
        # Get environment parameters
        params = env_params[environment]
        
        # Add environment identifier
        params['environment'] = environment
        
        # Create node
        return cls(
            name=name or f"{environment.upper()} Connection",
            description=description or f"Connection to {params['database']} on {params['server']}",
            **params
        )