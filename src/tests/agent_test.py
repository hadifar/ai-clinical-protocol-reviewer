from services.agent_service import invoke_agent

info, messages = invoke_agent("primary_study_objectives")
print(info)
