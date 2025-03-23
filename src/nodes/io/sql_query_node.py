from typing import Dict, Any, Optional, List, Union, Tuple
from sqlalchemy import text
from sqlalchemy.engine.row import Row
from src.core.node import BaseNode, InputSchema, OutputSchema

class MSSQLQueryNode(BaseNode):
    """
    Node for executing SQL queries against MSSQL databases.
    Fully declarative - all inputs are provided at initialization.
    
    Args:
        name: Name of the node
        description: Description of the node's purpose
        execute_mode: Mode of execution ('execute', 'get_one', 'get_many', 'get_all')
        connection: MSSQL connection object
        query: SQL query string
        params: Optional parameters for the query (dict or tuple)
        size: Number of records to fetch when using get_many mode (default: 10)
    
    Example:
        query_node = MSSQLQueryNode(
            name="User Query",
            description="Get active users",
            execute_mode="get_all",
            connection=db_connection,
            query="SELECT * FROM Users WHERE IsActive = 1"
        )
        
        # Execute and get results
        results = query_node.run().get_output('result')
    """
    
    def _define_schemas(self):
        """Define input and output schemas for SQL queries."""
        self.input_schema = InputSchema(
            required_keys=['connection', 'query'],
            optional_keys=['params', 'size']
        )
        self.output_schema = OutputSchema(
            keys=['result']
        )
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the inputs by executing the SQL query based on execute_mode.
        
        Args:
            inputs: Dictionary containing:
                - connection: MSSQL connection object
                - query: SQL query string
                - params: Optional query parameters
                - size: Optional fetch size for get_many mode
                
        Returns:
            Dictionary containing the query result under the 'result' key
            
        Raises:
            ValueError: If no query is provided or params are of invalid type
        """
        connection = inputs['connection']
        query = inputs['query']
        params = inputs.get('params')
        
        if not query:
            raise ValueError("No query provided")
            
        # Determine execution method based on execute_mode
        if self.execute_mode == 'get_all':
            result = self._get_all(connection, query, params)
        elif self.execute_mode == 'get_many':
            size = inputs.get('size', 10)
            result = self._get_many(connection, query, params, size)
        elif self.execute_mode == 'get_one':
            result = self._get_one(connection, query, params)
        elif self.execute_mode == 'execute':
            result = self._execute(connection, query, params)
        else:
            # Default to get_all if no execute_mode is specified
            result = self._get_all(connection, query, params)
            
        return {'result': result}
    
    def _execute(self, connection, query: str, params: Optional[Union[Dict[str, Any], Tuple]] = None):
        """
        Execute a query and return the raw result object.
        
        Args:
            connection: MSSQL connection object
            query: SQL query string
            params: Optional parameters (dict or tuple)
            
        Returns:
            SQLAlchemy result object
            
        Raises:
            ValueError: If params are of invalid type
        """
        with connection.connect() as conn:
            if isinstance(params, dict):
                result = conn.execution_options(no_parameters=True).execute(text(query), params)
            elif isinstance(params, tuple):
                result = conn.execution_options(no_parameters=True).execute(text(query), [params])
            elif params is None:
                result = conn.execution_options(no_parameters=True).execute(text(query))
            else:
                raise ValueError("params must be either a dictionary, a tuple, or None")
            conn.commit()
            return result
    
    def _get_one(self, connection, query: str, params: Optional[Union[Dict[str, Any], Tuple]] = None) -> Optional[Any]:
        """
        Execute a query and return the first result.
        
        Args:
            connection: MSSQL connection object
            query: SQL query string
            params: Optional parameters (dict or tuple)
            
        Returns:
            First column of the first row, or None if no results
        """
        with connection.connect() as conn:
            if params:
                result = conn.execution_options(no_parameters=True).execute(text(query), params).fetchone()
            else:
                result = conn.execution_options(no_parameters=True).execute(text(query)).fetchone()
            conn.commit()
            return result[0] if result else None
    
    def _get_many(self, connection, query: str, params: Optional[Union[Dict[str, Any], Tuple]] = None, 
                 size: int = 10) -> List[Dict[str, Any]]:
        """
        Execute a query and return up to 'size' results as dictionaries.
        
        Args:
            connection: MSSQL connection object
            query: SQL query string
            params: Optional parameters (dict or tuple)
            size: Maximum number of results to return
            
        Returns:
            List of row dictionaries, limited by size
        """
        with connection.connect() as conn:
            if params:
                result = conn.execution_options(no_parameters=True).execute(text(query), params).fetchmany(size)
            else:
                result = conn.execution_options(no_parameters=True).execute(text(query)).fetchmany(size)
            conn.commit()
            return [self._row_to_dict(row) for row in result]
    
    def _get_all(self, connection, query: str, params: Optional[Union[Dict[str, Any], Tuple]] = None) -> List[Dict[str, Any]]:
        """
        Execute a query and return all results as dictionaries.
        
        Args:
            connection: MSSQL connection object
            query: SQL query string
            params: Optional parameters (dict or tuple)
            
        Returns:
            List of all result rows as dictionaries
        """
        with connection.connect() as conn:
            if params:
                result = conn.execution_options(no_parameters=True).execute(text(query), params).fetchall()
            else:
                result = conn.execution_options(no_parameters=True).execute(text(query)).fetchall()
            conn.commit()
            return [self._row_to_dict(row) for row in result]
    
    def _row_to_dict(self, row: Row) -> Dict[str, Any]:
        """
        Convert a SQLAlchemy Row object to a dictionary.
        
        Args:
            row: SQLAlchemy Row object
            
        Returns:
            Dictionary representation of the row
        """
        if isinstance(row, dict):
            return row
        elif hasattr(row, '_asdict'):
            return row._asdict()
        elif hasattr(row, 'keys'):
            return {key: getattr(row, key) for key in row.keys()}
        else:
            return {key: value for key, value in zip(row.keys(), row)}