from services.agent_service import invoke_agent

info, messages = invoke_agent("Primary study objectives")
print(info)
