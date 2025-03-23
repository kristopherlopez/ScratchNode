def _format_value(value, max_length=100):
    """
    Format a value for display, handling different types appropriately.
    
    Args:
        value: Value to format
        max_length: Maximum string length before truncation
        
    Returns:
        Formatted string representation
    """
    if value is None:
        return "None"
    elif isinstance(value, dict):
        keys = list(value.keys())
        if len(keys) > 3:
            return f"Dict with {len(keys)} keys: {', '.join(keys[:3])}..."
        else:
            return f"Dict with keys: {', '.join(keys)}"
    elif isinstance(value, list):
        if len(value) > 3:
            return f"List with {len(value)} items"
        elif not value:
            return "Empty list"
        else:
            return f"List: [{', '.join(str(x)[:20] for x in value)}]"
    elif hasattr(value, '__dict__'):  # Object instance
        return f"{value.__class__.__name__} object"
    else:
        s = str(value)
        if len(s) > max_length:
            return s[:max_length] + "..."
        return s