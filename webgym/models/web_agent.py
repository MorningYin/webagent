# webgym/models/web_agent.py
import os
import threading
import time
from typing import List, Dict, Any, Tuple, Optional
from webgym.data.components import Action, Response
from webgym.context import ContextManager
from webgym.data.response_decomposer import decompose_raw_response, get_action_string
from webgym.utils import BlocklistManager
import httpx


class WebAgent:
    """
    Web automation agent that uses context management for action generation.
    Uses vLLM for model inference.
    """

    def __init__(self, policy_config, context_config, model_config, save_path, vllm_server_url: str,
                 openai_config: Optional[Dict] = None, operation_timeout: int = 120,
                 vllm_timeout: int = 120, max_retries: int = 1, max_vllm_sessions: int = 32,
                 verbose: bool = True):
        """
        Initialize WebAgent.

        Args:
            policy_config: Policy configuration
            context_config: Context configuration
            model_config: Model configuration (e.g., {"model_type": "qwen3vl", "variant": "instruct"})
            save_path: Path to save models
            vllm_server_url: URL for vLLM server (required)
            openai_config: OpenAI configuration for evaluation (optional, creates internal Evaluator)
            operation_timeout: General operation timeout in seconds
            vllm_timeout: Timeout specifically for vLLM requests in seconds
            max_vllm_sessions: Maximum concurrent vLLM sessions (default: 32)
        """
        self.policy_config = policy_config
        self.context_config = context_config
        self.model_config = model_config
        self.save_path = save_path
        self.vllm_server_url = vllm_server_url
        self.openai_config = openai_config
        self.operation_timeout = operation_timeout
        self.vllm_timeout = vllm_timeout
        self.max_retries = max_retries
        self.max_vllm_sessions = max_vllm_sessions

        # ----------------------------------------------------------------
        # Optional closed-source / relay policy backend (OpenAI-compatible),
        # with a robust MULTI-STATION router (LiteLLM-style: weighted round-robin
        # + health circuit-breaker + automatic failover + cooldown recovery).
        # Activated ONLY when policy_config.api is set; else the original vLLM
        # path below runs unchanged.
        #
        #   policy_config:
        #     api:
        #       model: "gpt-5.4"          # default model (per-endpoint can override)
        #       max_retries: 8            # transient retries per call (across endpoints)
        #       backoff_cap_sec: 60
        #       endpoints:                # <-- multiple stations; budgets combined
        #         - {name: yunwu,  base_url: "https://yunwu.ai/v1", api_key_env_var: POLICY_API_KEY_YUNWU,  weight: 1}
        #         - {name: newapi, base_url: "https://<host>/v1",   api_key_env_var: POLICY_API_KEY_NEWAPI, weight: 1}
        #     # (legacy single-station form base_url/model/api_key_env_var still works)
        # ----------------------------------------------------------------
        policy_api = getattr(policy_config, 'api', None)
        self.policy_api = policy_api
        self.policy_endpoints = self._build_policy_endpoints(policy_api, verbose) if policy_api else []
        self.use_policy_api = len(self.policy_endpoints) > 0
        if self.use_policy_api:
            self.policy_api_max_retries = int(getattr(policy_api, 'max_retries', 8))
            self.policy_api_backoff_cap = float(getattr(policy_api, 'backoff_cap_sec', 60))
            self._policy_router_lock = threading.Lock()
            self._policy_rr = 0  # round-robin cursor

        # Configure vLLM HTTP client with connection pooling.
        # verify=False: the policy/eval/summarizer all hit user-controlled relay gateways
        # (中转站) whose Let's Encrypt certs get renewed periodically and sometimes ship an
        # INCOMPLETE chain (server sends only the leaf -> "unable to verify first certificate").
        # That broke the entire run once (all stations SSL-failed at once -> tasks died after
        # 40 sweeps). The API key already authenticates us; skipping chain verification on these
        # trusted middlemen makes cert-renewal breakage a non-event. Robustness > chain check here.
        self.vllm_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=2400,
                max_keepalive_connections=1200,
                keepalive_expiry=300.0
            ),
            # Talking to the LOCAL LiteLLM proxy: connect is instant; read must cover the proxy's
            # internal retries (num_retries * request_timeout ≈ 3*45 = 135s) so we don't time out
            # while the proxy is still failing over across backends.
            timeout=httpx.Timeout(180.0, connect=8.0),
            http2=False,
            verify=False
        )

        # Simple semaphore to limit concurrent requests
        self.vllm_semaphore = threading.Semaphore(max_vllm_sessions)

        # Suppress verbose httpx logging
        import logging
        logging.getLogger("httpx").setLevel(logging.WARNING)

        if verbose:
            print(f"✅ vLLM httpx client configured:")
            print(f"   - Max connections: 2400")
            print(f"   - Keepalive connections: 1200")
            print(f"   - Keepalive expiry: 300s")
            print(f"   - Connect timeout: 30s")
            print(f"   - Request timeout: {vllm_timeout}s")
            print(f"   - Concurrent requests: {max_vllm_sessions}")

        # Pre-warm connection pool for large session counts (vLLM only; relays have no /health)
        if max_vllm_sessions > 10 and not self.use_policy_api:
            self._prewarm_connection_pool(verbose=verbose)

        # These will be set by concrete implementations
        self.model = None
        self.train_model = None

        # Use checkpoint_path from policy_config if available
        checkpoint_name = getattr(policy_config, 'checkpoint_path', 'model.pt')
        if checkpoint_name and checkpoint_name.strip():
            checkpoint_full_path = os.path.join(save_path, checkpoint_name)
            if os.path.exists(checkpoint_full_path):
                self.updated_model_path = checkpoint_full_path
            else:
                self.updated_model_path = None
        else:
            self.updated_model_path = None

        # Initialize ContextManager with model_config
        self.context_manager = ContextManager(self.context_config, self.model_config, verbose)
        self.parser = self.context_manager.get_parser()

        # For backward compatibility - expose conversation builder
        self.prompt_constructor = self.context_manager.get_model_interface().conversation_builder

        # Initialize blocklist manager
        self.blocklist_manager = BlocklistManager(save_path)

        # Setup Evaluator if OpenAI config provided (for backward compatibility)
        self.evaluator = None
        if openai_config:
            from .evaluator import Evaluator
            self.evaluator = Evaluator(
                openai_config=openai_config,
                conversation_builder=self.prompt_constructor,
                max_retries=max_retries,
                verbose=verbose,
                route_chat=(self._route_chat if self.use_policy_api else None)
            )

        # Rolling running-log summarizer (gpt-5.4-mini via openai_config). Maintains a
        # bounded long-term "memory" injected into every step's input. Self-contained,
        # robust: any failure falls back to the previous log (never crashes a step).
        self.running_log_client = None
        self.running_log_model = None
        if openai_config:
            try:
                from openai import OpenAI
                key_env = openai_config.get('openai_api_key_env_var', 'OPENAI_API_KEY')
                if key_env in os.environ:
                    self.running_log_client = OpenAI(
                        api_key=os.environ[key_env],
                        base_url=openai_config.get('base_url'))
                    self.running_log_model = openai_config.get('model')
            except Exception as e:
                if verbose:
                    print(f"⚠️ running-log summarizer not initialized: {e}")

    _SUMMARIZER_SYSTEM = """You keep a running research log for a web-browsing agent working through a long, multi-step task. This log is the agent's memory of everything that has scrolled out of its recent screenshots, so keep it faithful to what matters — and naturally compact, since the agent re-reads it on every step.

You're given the current log and the latest step (the action taken, the page it led to, whether the page changed, and the agent's own note). Fold that step into the log and return the updated version — integrate it, don't just tack on a line.

Keep what would hurt to forget: concrete task-relevant facts (names, numbers, dates, prices, URLs, candidate answers), what's been confirmed or ruled out, dead-ends not worth retrying, and where useful things were found. Let go of what no longer earns its place: superseded page states, routine navigation that led nowhere, repetition; a run of failed attempts can collapse into one line.

Write it as terse notes for the agent's own use — no preamble, no headings, just the log itself."""

    def update_running_log(self, prev_log: str, step: Dict) -> str:
        """Fold one completed trajectory step into the bounded running log, via the SHARED
        account pool (_route_chat). Robust: returns prev_log on any failure."""
        prev_log = prev_log or ""
        if not self.use_policy_api:
            return prev_log
        try:
            at = getattr(step.get('response'), 'answering_tokens', {}) or {}
            obs = step.get('observation')
            try:
                page = (obs.page_metadata or {}).get('title', '') if obs is not None else ''
            except Exception:
                page = ''
            eff = at.get('observation', '') or ''
            changed = 'yes' if ('changed' in eff and 'did not' not in eff) else 'no'
            action_desc = (at.get('action') or '')[:200]
            note = (at.get('thought') or '')[:300]
            user = (f"Current log:\n{prev_log or '(empty — this is the first step)'}\n\n"
                    f"New step:\n- Action: {action_desc}\n- Page after action: {page}\n"
                    f"- Page changed: {changed}\n- Agent's note this step: {note}\n\nUpdated log:")
            out = self._route_chat(
                [{"role": "system", "content": self._SUMMARIZER_SYSTEM},
                 {"role": "user", "content": user}],
                max_tokens=2000, temperature=0.3, top_p=1.0)
            out = (out or "").strip()
            return out if out else prev_log
        except Exception:
            return prev_log

    def _get_combined_action_and_observation(self, trajectory: List[Dict], screenshot_path: str,
                                                page_metadata: Dict, ac_tree: str) -> Tuple[Action, str, Response]:
        """Get action, observation summary, and full response object"""

        # Now trajectory should always have at least one element with current observation
        task_name = trajectory[-1]['observation'].task.task_name
        task_website = trajectory[-1]['observation'].task.website

        # Build current observation dict for the conversation builder
        # Read submit from reward if available, otherwise default to False
        submit_flag = False
        if trajectory and trajectory[-1].get('reward') and hasattr(trajectory[-1]['reward'], 'submit'):
            submit_flag = trajectory[-1]['reward'].submit

        current_observation = {
            'image_path': screenshot_path,
            'ac_tree': ac_tree,
            'page_metadata': page_metadata,
            'task': task_name,
            'submit': submit_flag
        }

        # Build conversation using new ContextManager API
        messages = self.context_manager.build_conversation(
            task=task_name,
            trajectory=trajectory,
            current_observation=current_observation,
            website=task_website
        )

        # Ensure messages is a list
        if isinstance(messages, dict):
            messages = [messages]

        # Get response from vLLM
        raw_response = self._make_vllm_request_direct(messages)

        # Parse response using model-specific parser
        parsed_response = self.context_manager.parse_response(raw_response)

        # Defensive check: ensure parsed_response is a dict
        if not isinstance(parsed_response, dict):
            raise Exception(f"parse_response returned {type(parsed_response)} instead of dict. Raw response: {str(raw_response)[:200]}")

        # Extract action using model-specific extractor
        action_info = self.context_manager.extract_action(parsed_response)

        # Get action string for backward compatibility
        action_string = parsed_response.get('action', '') or str(action_info)

        # Create Action object
        action = Action(
            action=action_info,
            action_string=action_string
        )

        # Create Response object with model interface for proper parsing
        # Convert base64 images in prompt to file paths before storing
        from webgym.utils import convert_messages_to_path_format
        raw_prompt_with_paths = convert_messages_to_path_format(messages, trajectory, current_observation)

        response_obj = decompose_raw_response(
            raw_response,
            model_interface=self.context_manager.get_model_interface(),
            raw_prompt=raw_prompt_with_paths
        )

        return action, response_obj

    def get_action_and_observation_sync(self, trajectory: List[Dict], screenshot_path: str,
                                            page_metadata: Dict, step_data: Dict = None) -> Tuple[Action, str, Response]:
        """
        Get action, observation summary, and response object using combined approach
        """

        # Extract AC tree from step_data if provided
        ac_tree = step_data.get('ac_tree', '') if step_data else ''

        # Get action, observation summary, and response object
        action, response_obj = self._get_combined_action_and_observation(
            trajectory, screenshot_path, page_metadata, ac_tree
        )

        # Make sure response_obj is not None
        if response_obj is None:
            print("WARNING: Response object is None in get_action_and_observation_sync")

        return action, response_obj

    def parse_action_to_browser_command(self, action: Action) -> Dict:
        """Convert action to browser command using context manager"""
        return self.context_manager.parse_action_to_browser_command(action)

    # ================================================================
    # CONNECTION MANAGEMENT
    # ================================================================

    def __enter__(self):
        """No-op context manager entry for compatibility"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up client on exit"""
        if hasattr(self, 'vllm_client'):
            self.vllm_client.close()

    def _prewarm_connection_pool(self, num_connections: int = None, verbose: bool = True):
        """Pre-create HTTP connections to avoid cold start

        Args:
            num_connections: Number of connections to pre-warm (default: min(max_vllm_sessions, 100))
            verbose: Whether to print progress messages
        """
        import concurrent.futures

        num_connections = num_connections or min(self.max_vllm_sessions, 100)

        if verbose:
            print(f"🔥 Pre-warming {num_connections} vLLM connections...")

        def warmup_connection():
            """Simple health check to establish connection"""
            try:
                response = self.vllm_client.get(
                    f"{self.vllm_server_url}/health",
                    timeout=5.0
                )
                return response.status_code == 200
            except:
                return False

        # Run warmup in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_connections) as executor:
            results = list(executor.map(lambda _: warmup_connection(), range(num_connections)))

        successful = sum(results)
        if verbose:
            print(f"✅ Pre-warmed {successful}/{num_connections} connections")

    # ================================================================
    # VLLM REQUEST HANDLING
    # ================================================================

    def _make_vllm_request_direct(self, messages: List[Dict]):
        """Make direct vLLM request using httpx - let vLLM do the batching"""
        # Route to closed-source / relay API when configured (opt-in, vLLM path untouched otherwise)
        if self.use_policy_api:
            return self._make_policy_api_request(messages)

        model_name = getattr(self.policy_config, 'base_model', 'unknown')

        payload = {
            "model": self.updated_model_path if self.updated_model_path else model_name,
            "messages": messages,
            "temperature": getattr(self.policy_config, 'temperature', 0.7),
            "top_p": getattr(self.policy_config, 'top_p', 1e-5),
            "top_k": getattr(self.policy_config, 'top_k', 20),
            "max_tokens": getattr(self.policy_config, 'max_new_tokens', 512),
            "logprobs": 1,
            "echo": False,
        }

        # Use semaphore to avoid overwhelming vLLM
        with self.vllm_semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.vllm_client.post(
                        f"{self.vllm_server_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )

                    if response.status_code == 200:
                        data = response.json()
                        choice = data["choices"][0]
                        message = choice["message"]
                        content = message["content"]

                        # Check for reasoning_content (from vLLM reasoning parser)
                        if "reasoning_content" in message and message["reasoning_content"]:
                            reasoning = message["reasoning_content"]
                            content = f"<think>\n{reasoning}\n</think>\n\n{content}"

                        return content
                    else:
                        error_msg = f"vLLM server returned status {response.status_code}: {response.text[:200]}"
                        if attempt == self.max_retries:
                            raise Exception(error_msg)
                        else:
                            print(f"   🔄 vLLM request attempt {attempt + 1} failed: {error_msg}")
                            time.sleep(1 + attempt)

                except Exception as e:
                    if attempt == self.max_retries:
                        import traceback
                        print(f"vLLM request failed after {self.max_retries + 1} attempts: {e}")
                        traceback.print_exc()
                        raise Exception(f"vLLM server request failed: {e}")
                    else:
                        error_msg = str(e)
                        print(f"   🔄 vLLM request attempt {attempt + 1} failed: {error_msg[:100]}")
                        time.sleep(1 + attempt)

    def _inline_images_as_base64(self, messages: List[Dict]) -> List[Dict]:
        """Return a copy of messages with any file:// image_url converted to a
        base64 data URL, so a remote API (no local FS access) can read the images.
        Non-image content and already-base64/http URLs are passed through untouched.
        """
        from webgym.utils import encode_image_to_base64

        out = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                out.append(msg)
                continue
            new_content = []
            for item in content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("file://"):
                        local_path = url[len("file://"):]
                        item = {"type": "image_url",
                                "image_url": {"url": encode_image_to_base64(local_path)}}
                new_content.append(item)
            out.append({**msg, "content": new_content})
        return out

    # ===================== Multi-station policy router =====================
    # LiteLLM-style: weighted round-robin + health circuit-breaker + failover.

    @staticmethod
    def _normalize_chat_url(base_url):
        base = str(base_url).rstrip('/')
        if base.endswith('/chat/completions'):
            return base
        if base.endswith('/v1'):
            return base + '/chat/completions'
        return base + '/v1/chat/completions'

    def _build_policy_endpoints(self, policy_api, verbose=True):
        """Parse policy_config.api into endpoint dicts w/ health state.
        Accepts api.endpoints (list) or legacy single api.base_url."""
        def g(o, k, d=None):
            if hasattr(o, 'get'):
                try: return o.get(k, d)
                except Exception: pass
            return getattr(o, k, d)

        default_model = g(policy_api, 'model')
        raw = g(policy_api, 'endpoints')
        specs = list(raw) if raw else ([policy_api] if g(policy_api, 'base_url') else [])

        eps, skipped = [], []
        for i, s in enumerate(specs):
            # base_url can be a literal or read from an env var (keeps private relay
            # hosts out of version control, like the API keys).
            base_url_env = g(s, 'base_url_env_var')
            base_url = os.environ.get(base_url_env) if base_url_env else g(s, 'base_url')
            if not base_url:
                skipped.append(f"{g(s,'name',base_url_env or '?')} (no base_url / env {base_url_env} unset)")
                continue
            key_env = g(s, 'api_key_env_var', 'POLICY_API_KEY')
            if key_env not in os.environ:
                skipped.append(f"{g(s,'name',base_url)} (env {key_env} unset)")
                continue
            model = g(s, 'model') or default_model
            if not model:
                skipped.append(f"{g(s,'name',base_url)} (no model)")
                continue
            eps.append({
                'name': g(s, 'name') or f"ep{i}",
                'url': self._normalize_chat_url(base_url),
                'model': model,
                'key': os.environ[key_env],
                'weight': max(1, int(g(s, 'weight', 1))),
                'fallback': bool(g(s, 'fallback', False)),  # used only when no primary is healthy
                'dead_until': 0.0,    # cooldown end (time.time)
                'fails': 0,           # consecutive failures -> exponential cooldown (reset on success)
            })
        if specs and not eps:
            raise ValueError(
                "policy_config.api configured but no usable endpoint "
                f"(missing keys/models): {skipped}")
        if verbose:
            print("🌐 Policy router endpoints:")
            for e in eps:
                print(f"   - {e['name']}: {e['url']} model={e['model']} weight={e['weight']}")
            if skipped:
                print(f"   (skipped: {skipped})")
        return eps

    def _route_chat(self, messages, max_tokens=512, temperature=0.7, top_p=1.0):
        """Send ONE chat-completion to the local LiteLLM proxy. The proxy owns ALL backend
        selection: load-balancing across the relay gateways, latency-aware routing,
        per-deployment cooldown, retry and failover. We just POST and lightly retry the
        (local) proxy itself. Shared by policy / evaluator / running_log summarizer, so every
        gpt-5.4 call goes through the same pool. `messages` ready (images inlined). Returns str.
        """
        ep = self.policy_endpoints[0]   # single endpoint now: the local LiteLLM proxy
        payload = {"model": ep['model'], "messages": messages,
                   "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {ep['key']}"}
        max_retries = self.policy_api_max_retries
        cap = self.policy_api_backoff_cap
        last = "no attempt"
        with self.vllm_semaphore:
            for attempt in range(max_retries):
                try:
                    r = self.vllm_client.post(ep['url'], json=payload, headers=headers)
                    if r.status_code == 200:
                        try:
                            m = r.json()["choices"][0]["message"]
                            content = m.get("content") or ""
                            if m.get("reasoning_content"):
                                content = f"<think>\n{m['reasoning_content']}\n</think>\n\n{content}"
                            if content.strip():
                                return content
                            last = "empty 200"
                        except (ValueError, KeyError, IndexError, TypeError) as e:
                            last = f"malformed 200: {e}"
                    else:
                        body = r.text[:200]
                        low = body.lower()
                        # per-request content moderation -> deterministic, don't burn retries
                        if any(k in low for k in ("content_policy", "内容审计", "审计命中", "风险规则", "content policy", "sensitive")):
                            raise Exception(f"policy content rejected {r.status_code}: {body[:120]}")
                        last = f"{r.status_code} {body[:140]}"
                except httpx.HTTPError as e:
                    # the LOCAL proxy is unreachable (rare) -> brief wait & retry
                    last = f"proxy network: {str(e)[:120]}"
                time.sleep(min(cap, 1.5 * (attempt + 1)))
            raise Exception(f"policy proxy failed after {max_retries} attempts (last: {last})")

    def _make_policy_api_request(self, messages: List[Dict]):
        """Policy step: inline file:// images to base64, then route via the shared pool."""
        return self._route_chat(
            self._inline_images_as_base64(messages),
            max_tokens=getattr(self.policy_config, 'max_new_tokens', 512),
            temperature=getattr(self.policy_config, 'temperature', 0.7),
            top_p=getattr(self.policy_config, 'top_p', 1.0))

    def _build_http_messages(self, system_prompt: str, user_prompt: str,
                           screenshot_path: str, trajectory: list = None):
        """
        Build messages for HTTP requests with current image only.
        """
        from webgym.utils import encode_image_to_base64

        # Only add system message if system_prompt is not empty
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        user_content = [{"type": "text", "text": user_prompt}]

        # Add current screenshot only
        current_image_url = encode_image_to_base64(screenshot_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": current_image_url}
        })

        messages.append({"role": "user", "content": user_content})
        return messages
