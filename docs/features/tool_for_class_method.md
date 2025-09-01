# Tool for Class Methods

## Overview

The AgentFly framework now supports defining tools on class methods using the `@tool` decorator. This feature allows you to create tools that are bound to class instances, enabling better organization of related functionality and access to instance state and methods.

## Tool Definition Types

### 1. Standalone Function Tools (Traditional)

Standalone function tools are the traditional way to define tools in AgentFly. These are independent functions that don't require any instance context.

```python
@tool(name="AdditionTool", description="Adds two numbers.")
def add(a: int, b: int = 1) -> int:
    """
    Adds two numbers.

    Args:
        a (int): The first number.
        b (int): The second number which should be a non-negative integer.
    
    Returns:
        int: The sum of a and b.
    """
    return a + b
```

**Characteristics:**
- Independent functions with no instance context
- Can be called directly without any setup
- Suitable for stateless operations
- Automatically registered in the global tool registry

### 2. Class Method Tools (New Feature)

Class method tools are methods within a class that are decorated with `@tool`. These tools have access to the class instance and can use instance variables, methods, and state.

```python
class ImageEditingAgent(BaseAgent):
    def __init__(self, model_name_or_path: str, **kwargs):
        self._image_database = {}
        tools = [self.auto_inpaint_image_tool]
        super().__init__(
            model_name_or_path=model_name_or_path,
            system_prompt=IMAGE_AGENT_SYSTEM_PROMPT,
            tools=tools,
            **kwargs
        )
    
    @tool(
        name="auto_inpaint_image",
        description="Automatically detect objects and inpaint them in one operation."
    )
    async def auto_inpaint_image_tool(
        self,
        image_id: str, 
        detect_prompt: str, 
        prompt: str,
        # ... other parameters
    ) -> str:
        # This method has access to self._image_database
        image = self._get_image(image_id)
        # ... implementation
```

**Characteristics:**
- Bound to class instances
- Have access to `self` and instance state
- Can call other instance methods
- Automatically set up with the correct instance during agent initialization
- Maintain instance context across tool calls

## Key Differences

| Aspect | Standalone Function Tools | Class Method Tools |
|--------|--------------------------|-------------------|
| **Context** | No instance context | Bound to class instance |
| **State Access** | No access to instance variables | Full access to instance state |
| **Method Calls** | Cannot call instance methods | Can call other instance methods |
| **Initialization** | No setup required | Automatically bound during agent init |
| **Use Case** | Stateless operations | Stateful operations with instance data |
| **Registration** | Global tool registry | Instance-specific tool list |

## Implementation Details

### How Class Method Tools Work

1. **Detection**: The `Tool` class automatically detects if a decorated function is a method by checking if the first parameter is `self`:
   ```python
   sig = inspect.signature(func)
   params = list(sig.parameters.keys())
   if params and params[0] == 'self':
       self.is_method = True
   ```

2. **Instance Binding**: During agent initialization, the framework automatically sets the instance for method tools:
   ```python
   tool_methods = []
   for name, method in inspect.getmembers(self):
       if isinstance(method, Tool):
           tool_methods.append(method)
   for tool_method in tool_methods:
       if hasattr(tool_method, 'is_method') and tool_method.is_method:
           tool_method.instance = self
   ```

3. **Execution**: When a method tool is called, it's executed with the bound instance:
   ```python
   if self.is_method:
       if self.instance is None:
           raise ValueError(f"Instance not set for method tool {self.name}")
       if inspect.iscoroutinefunction(self.user_func):
           result = await self.user_func(self.instance, **kwargs)
       else:
           result = self.user_func(self.instance, **kwargs)
   ```

## Usage Examples

### Example 1: Image Processing Agent

```python
class ImageProcessingAgent(BaseAgent):
    def __init__(self, model_name_or_path: str, **kwargs):
        self._image_cache = {}
        self._processing_history = []
        
        # Define tools as instance methods
        tools = [
            self.process_image_tool,
            self.cache_image_tool,
            self.get_history_tool
        ]
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            tools=tools,
            **kwargs
        )
    
    @tool(name="process_image", description="Process an image with various filters")
    async def process_image_tool(self, image_id: str, filter_type: str) -> str:
        # Access instance state
        if image_id not in self._image_cache:
            return "Error: Image not found in cache"
        
        # Call instance methods
        image = self._get_cached_image(image_id)
        processed = self._apply_filter(image, filter_type)
        
        # Update instance state
        self._processing_history.append({
            "image_id": image_id,
            "filter": filter_type,
            "timestamp": time.time()
        })
        
        return f"Image {image_id} processed with {filter_type} filter"
    
    @tool(name="get_history", description="Get processing history")
    async def get_history_tool(self) -> str:
        return json.dumps(self._processing_history)
    
    def _get_cached_image(self, image_id: str):
        return self._image_cache[image_id]
    
    def _apply_filter(self, image, filter_type: str):
        # Implementation details
        pass
```

### Example 2: Database Agent

```python
class DatabaseAgent(BaseAgent):
    def __init__(self, model_name_or_path: str, database_url: str, **kwargs):
        self._db_connection = self._connect_to_db(database_url)
        self._query_cache = {}
        
        tools = [
            self.query_database_tool,
            self.insert_data_tool,
            self.get_cache_stats_tool
        ]
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            tools=tools,
            **kwargs
        )
    
    @tool(name="query_database", description="Execute a SQL query")
    async def query_database_tool(self, sql_query: str, use_cache: bool = True) -> str:
        # Check cache first
        if use_cache and sql_query in self._query_cache:
            return f"Cached result: {self._query_cache[sql_query]}"
        
        # Execute query using instance connection
        result = self._execute_query(sql_query)
        
        # Cache result
        if use_cache:
            self._query_cache[sql_query] = result
        
        return result
    
    @tool(name="insert_data", description="Insert data into database")
    async def insert_data_tool(self, table: str, data: dict) -> str:
        # Use instance database connection
        success = self._insert_record(table, data)
        return f"Data inserted successfully: {success}"
    
    def _connect_to_db(self, url: str):
        # Implementation details
        pass
    
    def _execute_query(self, query: str):
        # Implementation details
        pass
```



## Best Practices

### 1. Tool Organization
- Group related tools within the same class
- Use descriptive names that reflect the tool's purpose
- Keep tools focused on single responsibilities

### 2. Instance State Management
- Initialize instance variables in `__init__`
- Use instance methods for common operations
- Maintain clean separation between tool logic and helper methods

### 3. Error Handling
- Validate inputs within tools
- Handle instance state errors gracefully
- Provide meaningful error messages

### 4. Async Support
- Use `async def` for tools that perform I/O operations
- Ensure proper async/await patterns
- Handle both sync and async method calls

### 5. Tool Registration
- Add tools to the `tools` list in `__init__`
- Ensure all required tools are included
- Consider tool dependencies and order

## Migration from Standalone Tools

If you have existing standalone tools that would benefit from instance context:

1. **Identify candidates**: Look for tools that share common state or helper functions
2. **Create a class**: Wrap related tools in a class
3. **Move state**: Convert global variables to instance variables
4. **Update decorators**: Add `@tool` decorators to class methods
5. **Update agent**: Modify agent initialization to use the new class

## Conclusion

The new tool for class method feature provides a powerful way to organize related tools and maintain state across tool calls. By leveraging class instances, you can create more sophisticated agents that maintain context and share resources efficiently. This feature is particularly useful for agents that need to manage persistent state, such as image processing, database operations, or web scraping tasks.
