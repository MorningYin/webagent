# webgym/models/qwen/conversation_builder.py
from typing import List, Dict, Any
from ..base.conversation_builder import ConversationBuilder

class QwenConversationBuilder(ConversationBuilder):
    """Multi-turn conversation builder for Qwen3-VL"""

    def __init__(self, interaction_mode: str, variant: str = "instruct", prompt_version: str = "vanilla"):
        """
        Initialize Qwen conversation builder

        Args:
            interaction_mode: 'coordinates' or 'set_of_marks'
            variant: 'instruct' or 'thinking'
            prompt_version: 'vanilla' or 'complete'
        """
        self.interaction_mode = interaction_mode
        self.variant = variant.lower()
        self.prompt_version = prompt_version.lower()

        if self.variant not in ['instruct', 'thinking']:
            raise ValueError(f"Invalid variant: {variant}. Must be 'instruct' or 'thinking'")

        if self.prompt_version not in ['vanilla', 'complete', 'mini']:
            raise ValueError(f"Invalid prompt_version: {prompt_version}. Must be 'vanilla', 'complete' or 'mini'")

    def build_conversation(self, task: str, trajectory: List[Dict], current_observation: Dict, **kwargs) -> List[Dict]:
        """
        Build multi-turn conversation history with 2-round sliding window

        Following OSWorld desktop computer-use approach:
        - Keep last 2 rounds as explicit history (with images)
        - Summarize anything older than 2 rounds as text (running_log) in first user message
        - Add current screenshot as final user message
        - Total: 3 images (2 historical + 1 current). running_log carries long-term memory,
          so few recent images suffice for grounding -> ~40% fewer image tokens, faster calls.
          NOTE: window size is a TRAIN=INFERENCE consistency parameter — FIXED for the whole 5k run.

        Args:
            task: Task description
            trajectory: List of trajectory steps
            current_observation: Current observation dict

        Returns:
            List of message dicts (system, user, assistant alternating)
        """
        messages = []
        HISTORY_WINDOW = 2  # Keep last 2 rounds as explicit history (3 images total w/ current); running_log carries the rest

        # 1. System message with tool definition
        messages.append(self._build_system_message())

        num_steps = len(trajectory)

        # Determine which steps go into rolling history (last 4 rounds + current observation = 5 images)
        # Note: A "round" is (observation, response) pair. We need 4 complete rounds + current observation.
        # This gives us 5 observations total: [obs_n-4, obs_n-3, obs_n-2, obs_n-1, obs_current]
        if num_steps <= HISTORY_WINDOW + 1:
            # All steps fit in window (need HISTORY_WINDOW+1 to account for 4 rounds + current)
            rolling_history_start = 0
            older_steps = []
        else:
            # Split: older steps get summarized, last 5 observations (4 rounds + current) get full treatment
            rolling_history_start = num_steps - HISTORY_WINDOW - 1
            older_steps = trajectory[:rolling_history_start]

        # 2. First user message: task + website + summary of older history (if any)
        website = kwargs.get('website', '')
        if website:
            first_user_text = f"Please generate the next action according to the UI screenshot and task.\n\nTask: {task}\n\nInitial website: {website}\n\n"
        else:
            first_user_text = f"Please generate the next action according to the UI screenshot and task.\n\nTask: {task}\n\n"

        # Long-term memory: bounded running log maintained incrementally by the
        # summarizer model (stored on the current trajectory step by the rollout loop).
        running_log = ""
        if trajectory:
            running_log = trajectory[-1].get('running_log') or ""
        if running_log:
            first_user_text += f"Notes so far (your memory of earlier steps):\n{running_log}\n\n"

        first_user_text += "Generate the next action to complete the task."

        # Add first user message (no image yet - will be added with rolling history or current)
        if num_steps == 0:
            # No trajectory yet, just task description
            messages.append({
                "role": "user",
                "content": first_user_text
            })
        else:
            # Will add with first rolling history step
            pass

        # 3. Rolling history (last 4 rounds with images)
        for step_idx in range(rolling_history_start, num_steps):
            observation = trajectory[step_idx].get('observation')
            response = trajectory[step_idx].get('response')

            if observation:
                # Build user message with image
                if step_idx == rolling_history_start:
                    # First rolling history step - include task and summary
                    user_msg = self._build_user_message_with_text(
                        observation=observation,
                        text=first_user_text
                    )
                else:
                    # Subsequent steps - just image, no additional text (matches OSWorld)
                    user_msg = self._build_user_message_image_only(observation)
                messages.append(user_msg)

            if response:
                # Assistant message: raw response text
                assistant_msg = self._build_assistant_message(response)
                messages.append(assistant_msg)

        # 4. Current observation (final user message with current screenshot)
        # trajectory[-1] contains the current observation, so it's already included in the loop above

        return messages

    def _build_system_message(self) -> Dict:
        """Build system message with tool definition and response format"""

        if self.interaction_mode == 'coordinates':
            tool_def = self._get_computer_use_tool_def()
        else:
            tool_def = self._get_set_of_marks_tool_def()

        # Add response format based on prompt_version
        response_format = ""
        if self.prompt_version == "complete":
            # Complete version: includes Progress, Intention, Action, and tool call
            response_format = """

# How to work

Each step, write three things in order:

1) Thought — what you see on the screen right now, and the one objective you're pursuing at this moment: what you're trying to get done and how your next action moves it forward. If the objective you were just on has succeeded or hit a wall, say so before turning to the next one. Think about the situation in front of you — you don't need to plan the whole task ahead or list out future steps.
2) Action — one plain sentence saying what you're about to do.
3) A single <tool_call>...</tool_call> block with only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Work the way a careful researcher does: hold one objective at a time and see it through until it's done or clearly blocked, rather than darting between directions — each objective is a self-contained piece of work with a point and an outcome. Ground every action in what's actually on the screen; when a page is blank, blocked, or throws up a cookie/consent/popup, handle it as part of what you're already doing. Your earlier findings are kept for you under "Notes so far" — lean on it for anything that has scrolled out of view, no need to restate it. Once what you've genuinely seen answers the task, give the final answer."""
        elif self.prompt_version == "mini":
            # Mini version: small-model-optimized — STRONG explicit format rules + few-shot,
            # because smaller models drop the <tool_call>, mis-name it, or answer in plain text.
            response_format = """

# How to respond — follow this format EXACTLY, every single step

Output exactly three parts, in order:
1) Thought: one short sentence — what you see and the single objective right now.
2) Action: one short sentence — what you will do.
3) A <tool_call> block. THIS IS MANDATORY: every response MUST end with one <tool_call> block.

The tool_call is ALWAYS this shape — copy it exactly:
<tool_call>
{"name": "computer_use", "arguments": {"action": "<left_click|type|scroll|wait|go_back|navigate|answer>", ...}}
</tool_call>

RULES YOU MUST NOT BREAK:
- "name" is ALWAYS "computer_use". NEVER use the action as the name. WRONG: {"name":"navigate"}. RIGHT: {"name":"computer_use","arguments":{"action":"navigate","url":"..."}}.
- The action ALWAYS goes inside "arguments" as "action".
- To finish, you MUST submit with the answer action. NEVER write the final answer as plain text — it will not count.
- Never output a response without a <tool_call> block.

EXAMPLES — copy these formats exactly:

navigate:
Thought: I'm on Google; I should go straight to a source site.
Action: Navigate to Wikipedia.
<tool_call>
{"name": "computer_use", "arguments": {"action": "navigate", "url": "https://en.wikipedia.org"}}
</tool_call>

click:
Thought: The search box is near the top; I'll click it.
Action: Click the search box.
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [512, 80]}}
</tool_call>

type:
Thought: I'll search for the person.
Action: Type the query.
<tool_call>
{"name": "computer_use", "arguments": {"action": "type", "coordinate": [512, 80], "text": "John Smith actor birthplace"}}
</tool_call>

scroll:
Thought: The answer may be lower on the page.
Action: Scroll down.
<tool_call>
{"name": "computer_use", "arguments": {"action": "scroll", "direction": "down"}}
</tool_call>

FINAL ANSWER (you MUST do this to finish — do not write the answer as plain text):
Thought: The page confirms the city, so I can answer now.
Action: Submit the final answer.
<tool_call>
{"name": "computer_use", "arguments": {"action": "answer", "text": "Milwaukee, Wisconsin"}}
</tool_call>

Work one objective at a time, ground every action in what's on the screen, and the moment what you've seen answers the task, submit with the answer action. Your earlier notes are under "Notes so far"."""
        else:
            # Vanilla version: minimal format without Thoughts or Memory
            response_format = """

# Response format

Response format for every step:
1) Action: a short sentence describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Action describes the high-level intention of the tool call within a single sentence.
- Do not output anything else outside those two parts."""

        system_content = f"""You are an autonomous web-browsing agent. You're given a task and a live browser, and you work through it yourself — navigating, reading what's on the page, and acting on what you see — until you can answer from what you've actually observed.

# Tools

You drive the browser by calling functions. Their signatures are in <tools></tools>:
<tools>
{tool_def}
</tools>

Return each call as a JSON object with the function name and arguments inside <tool_call></tool_call> tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>{response_format}"""

        return {
            "role": "system",
            "content": system_content
        }

    def _build_user_message(self, task: str, observation: Any, step_idx: int) -> Dict:
        """Build user message with screenshot and query"""

        screenshot_path = observation.image_path

        # First message includes full task
        if task and step_idx == 0:
            text = f"Your task is: {task}\n\nPlease analyze the current screenshot and decide your next action."
        else:
            # Subsequent messages include observation of previous action
            page_title = observation.page_metadata.get('title', 'Page loaded') if hasattr(observation, 'page_metadata') else 'Page loaded'
            text = f"Observation: {page_title}\n\nPlease analyze the current screenshot and decide your next action."

        # Use file:// URL for vLLM (more efficient than base64)
        from webgym.utils import encode_image_to_file_url
        image_url = encode_image_to_file_url(screenshot_path)

        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text}
            ]
        }

    def _build_user_message_text_only(self, task: str, observation: Any, step_idx: int) -> Dict:
        """Build user message with TEXT ONLY (no image) for historical steps"""

        # First message includes full task
        if task and step_idx == 0:
            text = f"Your task is: {task}\n\nPlease analyze the screenshot and decide your next action."
        else:
            # Subsequent messages include observation of previous action
            page_title = observation.page_metadata.get('title', 'Page loaded') if hasattr(observation, 'page_metadata') else 'Page loaded'
            text = f"Observation: {page_title}"

        return {
            "role": "user",
            "content": text  # Text only, no image
        }

    def _build_user_message_with_text(self, observation: Any, text: str) -> Dict:
        """Build user message with both image and custom text"""
        # Handle both dict (current_observation) and object (trajectory observation)
        if isinstance(observation, dict):
            screenshot_path = observation['image_path']
        else:
            screenshot_path = observation.image_path

        # Use file:// URL for vLLM (more efficient than base64)
        from webgym.utils import encode_image_to_file_url
        image_url = encode_image_to_file_url(screenshot_path)

        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text}
            ]
        }

    def _build_user_message_image_only(self, observation: Any) -> Dict:
        """Build user message with only image (no text) - matches OSWorld style"""
        # Handle both dict (current_observation) and object (trajectory observation)
        if isinstance(observation, dict):
            screenshot_path = observation['image_path']
        else:
            screenshot_path = observation.image_path

        # Use file:// URL for vLLM (more efficient than base64)
        from webgym.utils import encode_image_to_file_url
        image_url = encode_image_to_file_url(screenshot_path)

        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }

    def _summarize_older_history(self, older_steps: List[Dict]) -> str:
        """
        Create compact text summary of steps older than 4-round window.
        Focus on actions taken, avoid stale visual details or coordinates.
        Matches OSWorld "Previous actions:" format.
        """
        if not older_steps:
            return ""

        # Bound the summary to the most recent SUMMARY_WINDOW older steps so the text
        # payload stays ~constant regardless of trajectory length. Long-horizon facts
        # are carried by the model's own Memory field, not by ever-growing summaries.
        SUMMARY_WINDOW = 10
        n_omitted = max(0, len(older_steps) - SUMMARY_WINDOW)
        summary_lines = []
        for i, step in enumerate(older_steps[n_omitted:], start=n_omitted):
            observation = step.get('observation')
            action = step.get('action')
            response = step.get('response')

            if action and response:
                # Get webpage name from observation
                webpage_name = "Unknown page"
                if observation and hasattr(observation, 'page_metadata'):
                    webpage_name = observation.page_metadata.get('title', 'Unknown page')

                # Get action description from response (the "Action:" field)
                action_desc = response.answering_tokens.get('action', '')

                # Fallback: if no action description, construct from action details
                if not action_desc:
                    action_key = action.action.get('key', 'unknown')
                    action_args = action.action.get('arguments', {})

                    # Create more informative fallback description
                    if action_key == 'click' and 'element_id' in action_args:
                        action_desc = f"Click element {action_args['element_id']}"
                    elif action_key == 'type':
                        if 'element_id' in action_args:
                            text = action_args.get('content', '')[:30]  # First 30 chars
                            action_desc = f"Type '{text}' into element {action_args['element_id']}"
                        elif 'text' in action_args:
                            text = action_args.get('text', '')[:30]
                            action_desc = f"Type '{text}'"
                    elif action_key == 'scroll':
                        direction = action_args.get('direction', 'down')
                        action_desc = f"Scroll {direction}"
                    elif action_key == 'answer':
                        content = action_args.get('content', '')[:50]
                        action_desc = f"Answer: {content}"
                    else:
                        action_desc = f"{action_key.capitalize()} action"

                # Get action effect (observation) - only tells if page changed or not
                observation_text = response.answering_tokens.get('observation', '')

                # Extract whether page changed (observation is generated by comparing screenshots)
                if observation_text:
                    if 'did not change' in observation_text:
                        effect = "page unchanged"
                    elif 'changed' in observation_text:
                        effect = "page changed"
                    else:
                        effect = "executed"
                else:
                    effect = "executed"

                # Create rich summary: Webpage, Action, Effect
                summary_lines.append(f"Step {i+1} on [{webpage_name}]: {action_desc} → {effect}")

        if not summary_lines:
            return "No previous actions."

        prefix = (f"(... {n_omitted} earlier steps omitted; their key results are retained in Memory ...)\n"
                  if n_omitted > 0 else "")
        return prefix + "\n".join(summary_lines)

    def _build_assistant_message(self, response: Any) -> Dict:
        """Build assistant message from response

        For thinking variant, only answer tokens (without thinking) are included
        in conversation history. This prevents the model from seeing its previous
        thinking tokens.
        """

        # Get full response (thinking + answer tokens for thinking variant)
        raw_response = response.raw_response

        # For thinking variant, exclude thinking tokens from conversation history
        # Only include answer tokens so model doesn't see previous thinking
        if self.variant == 'thinking':
            import re
            # Remove <think>...</think> blocks to get answer tokens only
            answer_tokens = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
            # Also handle case where thinking doesn't have opening tag (auto-start)
            answer_tokens = re.sub(r'^.*?</think>\s*', '', answer_tokens, flags=re.DOTALL)
            answer_tokens = answer_tokens.strip()
        else:
            # For non-thinking variants, all tokens are answer tokens
            answer_tokens = raw_response

        return {
            "role": "assistant",
            "content": answer_tokens
        }

    def _get_computer_use_tool_def(self) -> str:
        """Tool definition for coordinates mode"""
        return """{
  "name": "computer_use",
  "description": "Use a mouse and keyboard to interact with a computer. The screen's resolution is 1000x1000.\\n* You do not have access to download files or play videos.\\n* Focus on web browsing and navigation tasks only.",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "description": "The action to perform:\\n* `left_click`: Click the left mouse button at the specified coordinates.\\n* `type`: Type text at the specified coordinates. The system will automatically click at the coordinates, type the text, and press Enter.\\n* `scroll`: Scroll the page in the specified direction (up or down).\\n* `wait`: Wait for the specified number of seconds for changes to occur.\\n* `go_back`: Go back to the previous page in browser history.\\n* `navigate`: Navigate directly to a specific website URL. The URL must start with https://. CRITICAL: If you see reCAPTCHA or any CAPTCHA challenge on the screen, DO NOT attempt to solve it. Instead, immediately use the navigate action to go to a different relevant website to complete your task. Search engines (Bing, Google, DuckDuckGo) will block you with reCAPTCHA - bypass them by navigating directly to websites like wikipedia.org, news sites, company websites, government sites, etc. NEVER navigate to google.com.\\n* `answer`: Provide the final answer to complete the task.",
        "enum": ["left_click", "type", "scroll", "wait", "go_back", "navigate", "answer"]
      },
      "coordinate": {
        "type": "array",
        "description": "[x, y] coordinates (0-1000 range). Required for left_click and type actions. For type action, specify WHERE to type (e.g., coordinates of input field).",
        "items": {
          "type": "integer",
          "minimum": 0,
          "maximum": 1000
        },
        "minItems": 2,
        "maxItems": 2
      },
      "text": {
        "type": "string",
        "description": "Text to type or answer. Required for type and answer actions. Note: For type action, the system will automatically click at the coordinates, type the text, and press Enter - no need to click separately before typing."
      },
      "direction": {
        "type": "string",
        "enum": ["up", "down"],
        "description": "Scroll direction. Required for scroll action."
      },
      "time": {
        "type": "number",
        "description": "Seconds to wait. Required for wait action."
      },
      "url": {
        "type": "string",
        "description": "URL to navigate to. Required for navigate action. Must start with https://. When you encounter reCAPTCHA, use this to navigate away to a different website instead of trying to solve the CAPTCHA. Navigate to relevant websites that can help complete the task. Avoid google.com (will block you). Examples: wikipedia.org, news sites, company websites, government sites, etc. IMPORTANT: If a website fails to load (you see a 'Navigation failed' message), try the URL with www. added/removed: if the URL is 'https://example.com', try 'https://www.example.com'; if the URL is 'https://www.example.com', try 'https://example.com'."
      }
    },
    "required": ["action"]
  }
}"""

    def _get_set_of_marks_tool_def(self) -> str:
        """Tool definition for set_of_marks mode"""
        return """{
  "name": "web_interaction",
  "description": "Interact with web elements using their numerical labels shown on the screenshot.\\n* You do not have access to download files or play videos.\\n* Focus on web browsing and navigation tasks only.",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "description": "The action to perform:\\n* `click`: Click on the element with the specified numerical label.\\n* `type`: Type text into the element with the specified numerical label. The system will automatically click the element, type the text, and press Enter.\\n* `hover`: Hover the mouse over the element with the specified numerical label.\\n* `scroll`: Scroll the page in the specified direction (up or down).\\n* `wait`: Wait for the specified number of seconds for changes to occur.\\n* `go_back`: Go back to the previous page in browser history.\\n* `navigate`: Navigate directly to a specific website URL. The URL must start with https://. CRITICAL: If you see reCAPTCHA or any CAPTCHA challenge on the screen, DO NOT attempt to solve it. Instead, immediately use the navigate action to go to a different relevant website to complete your task. Search engines (Bing, Google, DuckDuckGo) will block you with reCAPTCHA - bypass them by navigating directly to websites like wikipedia.org, news sites, company websites, government sites, etc. NEVER navigate to google.com.\\n* `answer`: Provide the final answer to complete the task.",
        "enum": ["click", "type", "hover", "scroll", "wait", "go_back", "navigate", "answer"]
      },
      "element_id": {
        "type": "string",
        "description": "Numerical label of the element. Required for click, type, hover actions. For type action, specify WHICH element to type into."
      },
      "text": {
        "type": "string",
        "description": "Text to type or answer. Required for type and answer actions. Note: For type action, the system will automatically click the element, type the text, and press Enter - no need to click separately before typing."
      },
      "direction": {
        "type": "string",
        "enum": ["up", "down"],
        "description": "Scroll direction. Required for scroll action."
      },
      "time": {
        "type": "number",
        "description": "Seconds to wait. Required for wait action."
      },
      "url": {
        "type": "string",
        "description": "URL to navigate to. Required for navigate action. Must start with https://. When you encounter reCAPTCHA, use this to navigate away to a different website instead of trying to solve the CAPTCHA. Navigate to relevant websites that can help complete the task. Avoid google.com (will block you). Examples: wikipedia.org, news sites, company websites, government sites, etc. IMPORTANT: If a website fails to load (you see a 'Navigation failed' message), try the URL with www. added/removed: if the URL is 'https://example.com', try 'https://www.example.com'; if the URL is 'https://www.example.com', try 'https://example.com'."
      }
    },
    "required": ["action"]
  }
}"""

    def summarize_trajectory(self, trajectory: List[Dict]) -> str:
        """Summarize trajectory for evaluation - creates a simple text summary of actions and observations"""
        summary_lines = []

        for i, step in enumerate(trajectory):
            action = step.get('action')
            response = step.get('response')

            if action and response:
                action_str = action.action_string if action.action_string else ''
                observation = response.answering_tokens.get('observation', '')
                summary_lines.append(f"Step {i}: {action_str} | Observation: {observation}")

        return "\n".join(summary_lines) if summary_lines else "No trajectory available"

    def get_conversation_type(self) -> str:
        return "multi-turn"
