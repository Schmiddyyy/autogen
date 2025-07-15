import asyncio
import json
import re
import os
import subprocess
from typing import Dict, List, Optional, Any
import requests

import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager


class MultiAgentSystem:
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.agents = {}
        self.group_chat = None
        self.manager = None
        self.API_URL = "http://localhost:8081/task/index/"
        self.LOG_FILE = "results.log"  # Fixed: was LOGS_DIR
        self.total_tokens = 0  # Initialize token counter
        self.working_directory = None  # Initialize working directory
        self._setup_agents()

    def _setup_agents(self):
        """Set up the specialized agents for SWE-Bench tasks."""

        # Planner Agent - Strategic problem analysis
        self.agents["planner"] = AssistantAgent(
            name="Planner",
            system_message="""You are a Senior Software Architect and Problem Analyst.

                Your responsibilities:
                1. Analyze the problem statement thoroughly
                2. Understand the failing tests and what they expect
                3. Identify the root cause of the issue
                4. Create a minimal, targeted fix strategy
                5. Provide clear implementation guidance

                When you receive a problem:
                - Read the problem description carefully
                - Analyze what the failing tests expect vs. what's currently happening
                - Identify the specific files and functions that need modification
                - Create a step-by-step plan that addresses ONLY the failing tests
                - Avoid over-engineering - make minimal changes
                - Consider edge cases and potential regressions

                Always end your analysis with a clear, actionable plan for the Coder.""",
            llm_config=self.llm_config,
        )


        # Coder Agent - Implementation specialist
        self.agents["coder"] = AssistantAgent(
            name="Coder",
            system_message="""You are a Senior Software Engineer specializing in precise code implementation.

                Your responsibilities:
                1. Implement the Planner's strategy exactly
                2. Make actual file modifications that will show up in git diff
                3. Write clean, maintainable code that matches the project style
                4. Ensure all changes are saved to disk immediately
                5. Focus on minimal, targeted changes

                Implementation rules:
                - Always read existing code first to understand context and style
                - Make the smallest possible change that fixes the issue
                - Preserve all existing functionality
                - Follow the project's coding conventions
                - Save changes immediately after making them
                - Use proper error handling where appropriate
                - Test your changes locally if possible

                CRITICAL: You must actually modify files using file operations, not just suggest changes. Use the Executer Agent to make the Changes in the repo""",
            llm_config=self.llm_config,

        )

        # Tester Agent - Validation specialist
        self.agents["tester"] = AssistantAgent(
            name="Tester",
            system_message="""You are a Senior QA Engineer specializing in test execution and validation.

                Your responsibilities:
                1. Execute the test suite to verify fixes
                2. Validate that failing tests now pass
                3. Ensure existing tests continue to pass
                4. Provide detailed test result analysis
                5. Identify any regressions or issues

                Testing approach:
                - Run the specific FAIL_TO_PASS tests mentioned in the problem
                - Run the PASS_TO_PASS tests to check for regressions
                - Use the appropriate test framework (pytest, unittest, etc.)
                - Provide clear, detailed output of test results
                - If tests fail, provide diagnostic information
                - Suggest additional fixes if needed

                Always provide concrete test results with pass/fail counts.""",
            llm_config=self.llm_config,
        )

        self.agents["executer"] = UserProxyAgent(
            name="Executer",
            code_execution_config={
                "last_n_messages": 3,
                "work_dir": self.working_directory,
                "use_docker": False,
                "timeout": 60,
            },
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            llm_config=self.llm_config,
            )


        # Setup group chat
        self.group_chat = GroupChat(
            agents=list(self.agents.values()),
            messages=[],
            max_round=10,
            speaker_selection_method="auto",
            allow_repeat_speaker=False,
        )

        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )

    def _setup_working_directory(self, repo_dir: str):
        """Set up the working directory for the agents."""
        self.working_directory = repo_dir
        if os.path.exists(repo_dir):
            os.chdir(repo_dir)

    def _get_repository_context(self, repo_dir: str) -> str:
        """Get basic repository context including file structure."""
        try:
            # Get basic file structure
            result = subprocess.run(
                ["find", ".", "-type", "f", "-name", "*.py", "|", "head", "-20"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                shell=True
            )
            file_list = result.stdout.strip()

            # Get recent commits
            git_log = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            recent_commits = git_log.stdout.strip()

            context = f"""
Repository Structure (Python files):
{file_list}

Recent Commits:
{recent_commits}
"""
            return context
        except Exception as e:
            return f"Error getting repository context: {e}"

    def _extract_token_usage(self, chat_history: List[Dict]) -> int:
        """Extract and calculate token usage from the conversation."""
        total_tokens = 0

        for message in chat_history:
            if isinstance(message.get("content", ""), str):
                # Rough estimation: ~4 characters per token for English text
                content_length = len(message["content"])
                tokens = content_length // 4
                total_tokens += tokens

        self.total_tokens = total_tokens
        return total_tokens

    async def solve_problem(self, problem_prompt: str) -> Dict[str, Any]:
        """
        Main method to solve a SWE-Bench problem using the multi-agent system.

        Args:
            problem_prompt: The complete problem description and context

        Returns:
            Dictionary containing results, metrics, and metadata
        """
        try:
            # Extract repository directory from the prompt
            repo_match = re.search(r"Work in the directory: (\S+)", problem_prompt)
            if repo_match:
                repo_name = repo_match.group(1)
                repo_dir = os.path.join(os.getcwd(), repo_name)
                self._setup_working_directory(repo_dir)

            # Gather repository context
            repo_context = ""
            if self.working_directory and os.path.exists(self.working_directory):
                repo_context = self._get_repository_context(self.working_directory)

            # Enhanced prompt with repository context
            enhanced_prompt = f"""{problem_prompt}

Repository Context:
{repo_context}

Instructions for the team:
1. Planner: Start by analyzing the problem and creating a fix strategy
2. Coder: Implement the planned solution with actual file modifications. 
3. Tester: Validate the fix by running the appropriate tests
4. Work collaboratively and communicate clearly about progress and issues
5. Executer: Execute the suggestet changes by the Coder

Begin the problem-solving process now."""

            print("🚀 Starting AutoGen multi-agent collaboration...")

            # Initialize the group chat
            chat_result = self.agents["executer"].initiate_chat(
                self.manager,
                message=enhanced_prompt,
                clear_history=True,
            )

            # Extract results
            chat_history = (
                chat_result.chat_history if hasattr(chat_result, "chat_history") else []
            )
            if not chat_history and hasattr(self.group_chat, "messages"):
                chat_history = self.group_chat.messages

            token_usage = self._extract_token_usage(chat_history)

            # Get the final git diff to see what was changed
            git_diff = ""
            if self.working_directory and os.path.exists(self.working_directory):
                try:
                    diff_result = subprocess.run(
                        ["git", "diff"],
                        cwd=self.working_directory,
                        capture_output=True,
                        text=True,
                    )
                    git_diff = diff_result.stdout
                except Exception as e:
                    git_diff = f"Error getting git diff: {e}"

            # Check if any files were modified
            files_modified = len(git_diff.strip()) > 0

            return {
                "success": True,
                "files_modified": files_modified,
                "git_diff": git_diff,
                "token_usage": token_usage,
                "chat_history": chat_history,
                "working_directory": self.working_directory,
                "agent_count": len(self.agents),
                "conversation_rounds": len(chat_history),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "token_usage": self.total_tokens,
                "working_directory": self.working_directory,
                "git_diff": "",
                "files_modified": False,
            }

    def get_token_usage(self) -> int:
        """Get the total token usage from the last run."""
        return self.total_tokens

    def reset(self):
        """Reset the agent system for a new problem."""
        self.total_tokens = 0
        self.working_directory = None
        if self.group_chat:
            self.group_chat.messages = []

    async def handle_task(self, index):
        """Handle a single task by fetching, solving, and evaluating it."""
        api_url = f"{self.API_URL}{index}"
        print(f"Fetching test case {index} from {api_url}...")
        repo_dir = os.path.join(
            "./workspaces/", f"repo_{index}"
        )  # Use unique repo directory per task
        start_dir = os.getcwd()  # Remember original working directory

        try:
            # Ensure workspaces directory exists
            os.makedirs("./workspaces", exist_ok=True)

            response = requests.get(api_url)
            if response.status_code != 200:
                raise Exception(f"Invalid response: {response.status_code}")

            testcase = response.json()
            prompt = testcase["Problem_statement"]
            git_clone = testcase["git_clone"]
            fail_tests = json.loads(testcase.get("FAIL_TO_PASS", "[]"))
            pass_tests = json.loads(testcase.get("PASS_TO_PASS", "[]"))
            instance_id = testcase["instance_id"]

            parts = git_clone.split("&&")
            clone_part = parts[0].strip()
            checkout_part = parts[-1].strip() if len(parts) > 1 else None

            repo_url = clone_part.split()[2]

            # Remove existing repo directory if it exists
            if os.path.exists(repo_dir):
                subprocess.run(["rm", "-rf", repo_dir], check=True)

            print(f"Cloning repository {repo_url} into {repo_dir}...")
            env = os.environ.copy()
            env["GIT_TERMINAL_PROMPT"] = "0"
            subprocess.run(["git", "clone", repo_url, repo_dir], check=True, env=env)

            if checkout_part:
                commit_hash = checkout_part.split()[-1]
                print(f"Checking out commit: {commit_hash}")
                subprocess.run(
                    ["git", "checkout", commit_hash], cwd=repo_dir, check=True, env=env
                )

            full_prompt = (
                f"You are a team of agents with the following roles:\n"
                f"Work in the directory: repo_{index}. This is a Git repository.\n"
                f"Your goal is to fix the problem described below.\n"
                f"All code changes must be saved to the files, so they appear in `git diff`.\n"
                f"The fix will be verified by running the affected tests.\n\n"
                f"Problem description:\n"
                f"{prompt}\n\n"
                f"Make sure the fix is minimal and only touches what's necessary to resolve the failing tests."
            )

            print("Launching agents...")
            await self.solve_problem(full_prompt)

            token_total = self.extract_last_token_total_from_logs()

            # Call REST service for evaluation
            print(f"Calling SWE-Bench REST service with repo: {repo_dir}")
            test_payload = {
                "instance_id": instance_id,
                "repoDir": f"/repos/repo_{index}",  # mount with docker
                "FAIL_TO_PASS": fail_tests,
                "PASS_TO_PASS": pass_tests,
            }
            res = requests.post("http://localhost:8082/test", json=test_payload)
            res.raise_for_status()
            result_raw = res.json().get("harnessOutput", "{}")
            result_json = json.loads(result_raw)

            if not result_json:
                raise ValueError(
                    "No data in harnessOutput – possible evaluation error or empty result"
                )

            instance_id = next(iter(result_json))
            tests_status = result_json[instance_id]["tests_status"]
            fail_pass_results = tests_status["FAIL_TO_PASS"]
            fail_pass_total = len(fail_pass_results["success"]) + len(
                fail_pass_results["failure"]
            )
            fail_pass_passed = len(fail_pass_results["success"])
            pass_pass_results = tests_status["PASS_TO_PASS"]
            pass_pass_total = len(pass_pass_results["success"]) + len(
                pass_pass_results["failure"]
            )
            pass_pass_passed = len(pass_pass_results["success"])

            # Log results
            os.chdir(start_dir)
            with open(self.LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"\n--- TESTCASE {index} ---\n")
                log.write(
                    f"FAIL_TO_PASS passed: {fail_pass_passed}/{fail_pass_total}\n"
                )
                log.write(
                    f"PASS_TO_PASS passed: {pass_pass_passed}/{pass_pass_total}\n"
                )
                log.write(f"Total Tokens Used: {token_total}\n")
            print(f"Test case {index} completed and logged.")

        except Exception as e:
            os.chdir(start_dir)
            with open(self.LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"\n--- TESTCASE {index} ---\n")
                log.write(f"Error: {e}\n")
            print(f"Error in test case {index}: {e}")

    def extract_last_token_total_from_logs(self):
        """Extract the last token total from logs."""
        log_dir = "logs"

        if not os.path.exists(log_dir):
            return 0

        log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        if not log_files:
            return 0

        log_files.sort(reverse=True)

        latest_log_path = os.path.join(log_dir, log_files[0])
        try:
            with open(latest_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in reversed(lines):
                match = re.search(r"Cumulative Total=(\d+)", line)
                if match:
                    return int(match.group(1))
        except Exception as e:
            print(f"Error reading log file: {e}")

        return 0


async def main():
    """Main function to run the multi-agent system."""
    config = {
        "model": "gemma3:1b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "max_tokens": 4096,
    }

    agent = MultiAgentSystem(llm_config=config)
    for i in range(1, 300):
        await agent.handle_task(i)
        agent.reset()  # Reset for next task


if __name__ == "__main__":
    asyncio.run(main())