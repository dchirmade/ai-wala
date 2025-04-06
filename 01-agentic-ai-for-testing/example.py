# Import necessary classes from the autogen library
import autogen
import os
import json # Keep for potential future use
from typing import Dict, List, Optional, Union # Keep for type hinting

# --- Configuration ---
# IMPORTANT: This script requires configuration for OpenRouter and Gemini models.
# 1. Ensure OPENROUTER_API_KEY is set: export OPENROUTER_API_KEY='your_openrouter_key'
# 2. Ensure GOOGLE_API_KEY is set: export GOOGLE_API_KEY='your_gemini_key'
# 3. This version uses Autogen's config system to connect to OpenRouter,
#    treating it as an OpenAI-compatible endpoint via 'base_url'.

# Check for API keys (optional but recommended)
openrouter_key = os.getenv("OPENROUTER_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

if not openrouter_key:
    print("Warning: OPENROUTER_API_KEY environment variable not set.")
if not google_key:
    print("Warning: GOOGLE_API_KEY environment variable not set.")
if not openrouter_key or not google_key:
    print("Script might fail if API keys are missing or Autogen is not configured.")

# --- Autogen Configuration List ---
# Define a config list with configurations for both OpenRouter and Gemini.
# Agents will filter this list based on tags.

# Note: The attempt to pass 'extra_headers' directly caused a ValidationError.
# Removing it to fix the error. Passing custom headers might require
# a different configuration structure (e.g., request_options) or agent customization
# depending on the Autogen version and features.
# openrouter_extra_headers = {
#     # "HTTP-Referer": "<YOUR_SITE_URL>",
#     # "X-Title": "<YOUR_SITE_NAME>",
# }

config_list = [
    {
        "model": "meta-llama/llama-4-maverick:free", # Specify the OpenRouter model - llama-4-maverick
        "api_key": openrouter_key,
        "base_url": "https://openrouter.ai/api/v1", # Crucial for pointing to OpenRouter
        "tags": ["openrouter", "llama-4-maverick"], # Tag for easy filtering
        "price": [0, 0], # Price is often optional
        # "extra_headers": openrouter_extra_headers if openrouter_extra_headers else None, # Removed due to ValidationError
    },
    {
        "model": "gemini-2.0-flash-lite", # Or other Gemini model
        "api_key": google_key,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "tags": ["gemini", "gemini-2.0-flash-lite"], # Tag for easy filtering
        "price": [0, 0], # Price is often optional
        # Add other necessary fields for Gemini if required by your Autogen setup
        # e.g., "api_type": "google", "base_url": "..."

    },
]

# --- LLM Configuration Filters ---

# Config for the Generator Agent (uses OpenRouter via Autogen config)
openrouter_config = {
    "config_list": autogen.filter_config(config_list, {"tags": ["openrouter"]}),
    "cache_seed": 42, # Use caching
    "temperature": 0.7,
    # Note: 'extra_headers' was removed from the config_list entry above
}

# Config for the Reviewer Agent (uses Gemini)
gemini_config = {
    "config_list": autogen.filter_config(config_list, {"tags": ["gemini"]}),
    "cache_seed": 42, # Use caching
    "temperature": 0.5,
}

# Config for the Group Chat Manager (now uses Gemini)
manager_config = gemini_config


# --- Agent Definition ---
# NOTE: The custom OpenRouterAgent class is REMOVED.

# 1. User Proxy Agent (Required for group chat coordination)
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message="Acts as the initiator and coordinator. Terminates the process once the refined list is provided.",
    human_input_mode="NEVER", # Fully automated
    code_execution_config=False, # No code execution
    llm_config=manager_config, # Manager uses Gemini config
    is_termination_msg=lambda x: "FINAL REFINED LIST" in x.get("content", "").upper()
)

# 2. Test Case Generator Agent (Standard AssistantAgent configured for OpenRouter)
generator = autogen.AssistantAgent(
    name="Llama_4_maverick_Software_Tester",
    llm_config=openrouter_config, # Pass the config pointing to OpenRouter
    system_message="""As an AI expert in software testing utilizing the Llama_4_maverick model through OpenRouter,
     your primary responsibility is to generate test cases as a software tester."""
)

# 3. Test Case Reviewer/Refiner Agent (Uses Gemini)
reviewer = autogen.AssistantAgent(
    name="Gemini_Software_Automation_Engineer",
    llm_config=gemini_config, # Uses Gemini config defined earlier
    system_message="""As an AI automation expert for mobile applications powered by Gemini, your task is to implement test cases in Python and Appium.
Please provide a sample implementation for each test case under the title 'FINAL REFINED LIST:'."""
)


# --- Group Chat Setup ---

# Create the list of agents, using the standard AssistantAgent for the generator
agent_list = [user_proxy, generator, reviewer]

# Create the group chat
groupchat = autogen.GroupChat(
    agents=agent_list,
    messages=[],
    max_round=200 # Allow enough rounds for generation and review
)

# Create the chat manager (using Gemini config now)
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=manager_config # Manager uses Gemini LLM to coordinate
)

# --- Task Execution ---

# Define the specific Software Under Test (SUT) and initiate the chat
sut_software_description = "As a user, I would like to utilize the mobile application to adjust the volume of my television."

# Update prompt to refer to the generator agent name (no change needed here)
initial_prompt = f"""The task is to generate and automate test cases for: {sut_software_description}

Workflow:
1.  **Llama_4_maverick_Software_Tester**: Utilize the Deepseek model through the OpenRouter (configured endpoint) to generate a total of two functional test case scenarios: one positive and one negative.
2.  **Gemini_Software_Automation_Engineer**: Please develop sample Python code utilizing the Appium framework.
3.  **Llama_4_maverick_Software_Tester**: Performs code analysis and integrates tests to ensure desired outcomes and actions.
4.  **Gemini_Software_Automation_Engineer**: Please do final review of the proposed test cases and make a final list under the section titled “FINAL REFINED LIST:”.
5.  **User_Proxy**: Terminate the chat once the final refined list is presented.

Begin the process. Llama_4_maverick_Software_Tester, please provide the initial list. Gemini_Software_Automation_Engineer, make sure that you review it on the go."""

print("Initiating multi-agent test case generation (OpenRouter via config) and review (Gemini)...")

# Initiate the chat with the manager
user_proxy.initiate_chat(
    manager,
    message=initial_prompt,
)

print("Multi-agent process finished.")

# --- Output ---
# The conversation between the agents will be printed.
# The generator agent now uses the standard AssistantAgent mechanism,
# configured to communicate with the OpenRouter endpoint.
