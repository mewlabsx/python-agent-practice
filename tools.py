from rag import build_rag_context


def create_retrieve_knowledge_tool(knowledge_base):
    def retrieve_knowledge(query):
        rag_result = build_rag_context(
            query,
            knowledge_base["chunks"],
            knowledge_base["index"],
            top_k=3
        )
        return rag_result["context"]

    return retrieve_knowledge


# 函数本身的定义
def get_project_info(project_name):
    if project_name == "DTM":
        return {
            "name": "DTM",
            "description": "Synthetic distributed task runtime example",
            "status": "Demo fixture"
        }
    elif project_name == "AI":
        return {
            "name": "AI",
            "description": "Synthetic AI project example",
            "status": "Demo fixture"
        }
    else:
        raise ValueError(f"Project '{project_name}' not found")


# 函数本身的定义
def calculator(a, b, operation):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    else:
        raise ValueError("Invalid operation")


def retrieve_knowledge(query):
    rag_result = build_rag_context(
        query,
        knowledge_base["chunks"],
        knowledge_base["index"],
        top_k=3
    )
    return rag_result["context"]


# 给大模型的函数描述
calculator_definition = {
    "name": "calculator",
    "description": "Perform a basic arithmetic operation on two numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "The first number."
            },
            "b": {
                "type": "number",
                "description": "The second number."
            },
            "operation": {
                "type": "string",
                "description": "The arithmetic operation to perform.",
                "enum": ["add", "subtract", "multiply", "divide"]
            }
        },
        "required": ["a", "b", "operation"]
    }
}

# 给大模型的函数描述
get_project_info_definition = {
    "name": "get_project_info",
    "description": "Retrieve information about a specific project.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string"
            }
        },
        "required": ["project_name"]
    }
}


# 工具定义
tools = {
    "calculator": calculator,
    "get_project_info": get_project_info,
    "retrieve_knowledge": retrieve_knowledge
}

# 给大模型的工具定义
openai_tools = [
    {
        "type": "function",
        "function": calculator_definition
    },
    {
        "type": "function",
        "function": get_project_info_definition
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": (
                "Search the project technical knowledge base. "
                "Use this tool when the user asks about project requirements, design, implementation, testing, "
                "verification, or other facts that should be grounded in the knowledge base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The technical knowledge query to retrieve."
                    }
                },
                "required": ["query"]
            }
        }
    }
]
