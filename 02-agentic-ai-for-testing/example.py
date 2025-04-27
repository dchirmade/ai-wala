import os
import autogen
from typing import List, Dict, Any

# --- Configuration ---

# 1. Get OpenRouter API Key (ensure environment variable is set)
# api_key = os.getenv("OPENROUTER_API_KEY")
api_key = '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'
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

# 2. Define the list of 7 FREE model identifiers you want to use.
#    *** CRITICAL: VERIFY these on https://openrouter.ai/models ***
#    *** These are placeholders and MUST be updated! ***
free_model_identifiers = [
    "deepseek/deepseek-r1:free",             # Placeholder 1 (e.g., for Requirements)
    "meta-llama/llama-4-maverick:free",                        # Placeholder 2 (e.g., for Architecture)
    "qwen/qwq-32b:free",                     # Placeholder 3 (e.g., for Backend Dev)
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",# Placeholder 4 (e.g., for Frontend Dev)
    "microsoft/mai-ds-r1:free",         # Placeholder 5 (e.g., for Unit Testing)
    "agentica-org/deepcoder-14b-preview:free",                       # Placeholder 6 (e.g., for UI Automation)
    "mistralai/mistral-nemo:free",              # Placeholder 7 (e.g., for QA/Docs)
    # Ensure you have exactly 7 valid, free model identifiers here
]

if len(free_model_identifiers) != 7:
    raise ValueError(f"Expected 7 model identifiers, but found {len(free_model_identifiers)}. Please update the list.")

# 3. Define the base OpenRouter configuration
openrouter_config_base = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": api_key,
    "price": [0, 0], # Price is often optional
    # Add other common parameters if needed, e.g., "temperature": 0.7
}

# 4. Create LLM Configs for each agent, assigning one model per agent
llm_config_list = [
    {
        "model": model_id,
        **openrouter_config_base # Merge base config
    }
    for model_id in free_model_identifiers
]

# --- Agent Definitions ---

# Define roles for each agent corresponding to SDLC phases
# Assigning roles more aligned with SDLC
agent_roles = [
    "Requirements_Analyst: You define functional and non-functional requirements clearly and concisely. Elicit details needed for development and testing.",
    "Software_Architect: You design the high-level structure and interaction of software components. Provide simple architectural diagrams or descriptions.",
    "Backend_Developer: You focus on server-side logic, database interactions, and API design. Provide pseudocode or explanations of the backend implementation.",
    "Frontend_Developer: You focus on the user interface structure (HTML), presentation (CSS), and client-side behavior (JavaScript). Describe the UI components and interactions.",
    "Unit_Tester: You write and suggest unit tests for individual functions or methods, primarily focusing on backend logic using frameworks like pytest.",
    "UI_Automation_Tester: You create automated test scripts for the user interface using tools like Selenium or Playwright in Python.",
    "QA_Documentation_Specialist: You oversee the quality assurance process, summarize test activities, document results, and ensure clarity in all project artifacts.",
]


if len(agent_roles) != len(llm_config_list):
     raise ValueError("Mismatch between number of roles and number of models/configs.")

# Create Assistant Agents, one for each role and model config
agents: List[autogen.AssistantAgent] = []
for i in range(len(llm_config_list)):
    agent_name = agent_roles[i].split(":")[0] # Extract name like "Requirements_Analyst"
    system_message = agent_roles[i] # Full role description

    agents.append(
        autogen.AssistantAgent(
            name=agent_name,
            system_message=system_message,
            llm_config={"config_list": [llm_config_list[i]]} # Assign one specific config
        )
    )
    print(f"Created Agent: {agent_name} using model: {llm_config_list[i]['model']}")

# Create the User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="Project_Manager", # Changed name for better role fit
    human_input_mode="TERMINATE", # Allows user input to guide or end, terminates on "TERMINATE"
    max_consecutive_auto_reply=15, # Increased slightly for potentially longer flow
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "autogen_sdlc_coding", # Directory for generated code/artifacts
        "use_docker": False,  # Set to True if Docker is available and preferred
    },
    # Updated system message for the coordinator role
    system_message="""You are the Project Manager.
    Your role is to initiate the request and guide the team through the software development lifecycle phases for the given feature.
    Ensure each specialist contributes appropriately: Requirements -> Architecture -> Backend -> Frontend -> Unit Tests -> UI Tests -> QA Summary.
    Keep the discussion focused and moving forward. Ask clarifying questions if needed.
    Summarize the final outputs. Reply TERMINATE when all steps are satisfactorily completed."""
)

# --- Group Chat Setup ---

groupchat = autogen.GroupChat(
    agents=[user_proxy] + agents, # Include Project_Manager and all specialist agents
    messages=[],
    max_round=25 # Increased max rounds for SDLC flow
)

# Use the GroupChatManager to facilitate the conversation
# Using the first model's config for the manager
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": [llm_config_list[0]]}
)

# --- Initiate Chat ---

# Define the initial task, guiding the SDLC flow
initial_task = """
Team, we need to develop and test the login functionality for our hypothetical 'WebAppLogin'.

Here's the basic user flow:
- User enters username (id: 'username') and password (id: 'password').
- User clicks the login button (id: 'login-button').
- On success: Redirect to '/dashboard'.
- On failure: Show error message (class: 'error-message').

Let's proceed through our standard process:
1.  **Requirements_Analyst**: Define the key functional requirements for this login feature.
2.  **Software_Architect**: Propose a simple architectural approach (e.g., client-server interaction).
3.  **Backend_Developer**: Outline the server-side logic/pseudocode for authentication.
4.  **Frontend_Developer**: Describe the necessary HTML elements and basic client-side interaction.
5.  **Unit_Tester**: Suggest 1-2 Python unit test cases for the backend authentication logic.
6.  **UI_Automation_Tester**: Generate a basic Python Selenium script for a successful login UI test.
7.  **QA_Documentation_Specialist**: Oversee the process and prepare a brief summary at the end.

Project_Manager (me) will coordinate. Let's start with Requirements. Requirements_Analyst, please begin.
"""

print("\n--- Starting Autogen SDLC Group Chat ---")
print(f"Initial Task: {initial_task}")
print("-" * 30)

# Initiate the chat with the defined SDLC task
user_proxy.initiate_chat(
    manager,
    message=initial_task,
)

print("\n--- Autogen Group Chat Finished ---")