# Syllabus — Harness Agent From Scratch

22 notebooks (`ch00`–`ch21`), one chapter each. Each notebook follows the template in `NOTEBOOK_AUTHORING.md`.

| Ch | Notebook | Module(s) | CLI milestone |
|----|----------|-----------|---------------|
| 00 | ch00_introduction | setup | `harness-agent doctor` |
| 01 | ch01_llm_tool_calling | providers | tool call |
| 02 | ch02_messages_and_providers | providers | multi-turn |
| 03 | ch03_agent_loop | agent.py | agent loop |
| 04 | ch04_tool_registry | tools/ | registry |
| 05 | ch05_observations_and_recovery | observations.py | structured results |
| 06 | ch06_session_storage | sessions/ | SQLite + FTS |
| 07 | ch07_prompt_assembly | prompt/ | system prompt |
| 08 | ch08_skills_system | skills/ | progressive disclosure |
| 09 | ch09_memory_and_user_model | memory/ | MEMORY/USER |
| 10 | ch10_closed_learning_loop | learning/ | auto skills |
| 11 | ch11_context_compression | compression/ | long chats |
| 12 | ch12_subagents_and_delegate | delegate.py | subagent |
| 13 | ch13_mcp_integration | mcp/ | MCP tools |
| 14 | ch14_cron_scheduler | cron/ | cron tick |
| 15 | ch15_gateway_and_cli | gateway/, cli/ | chat + gateway |
| 16 | ch16_provider_resolution | providers/, config | /model |
| 17 | ch17_terminal_backends | tools/environments/ | local/docker |
| 18 | ch18_acp_integration | acp/ | acp stdio |
| 19 | ch19_trajectories_and_batch | trajectories/ | export JSONL |
| 20 | ch20_plugins_and_hooks | plugins/ | plugin tool |
| 21 | ch21_full_system_integration | all | capstone |
