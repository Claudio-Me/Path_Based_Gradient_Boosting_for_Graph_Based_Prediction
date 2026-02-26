"""Process-based timeout mechanism for long-running evaluations."""
from multiprocessing import Process, Queue
from typing import Any, Callable, Optional, Tuple


class TimeoutException(Exception):
    """Raised when a function exceeds its timeout."""
    pass


def _run_in_process(q: Queue, func: Callable, args: tuple, kwargs: dict):
    """Worker function that runs in a separate process."""
    try:
        result = func(*args, **(kwargs or {}))
        q.put(("OK", result))
    except Exception as e:
        q.put(("ERR", f"{type(e).__name__}: {e}"))


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout_sec: int = 72000
) -> Tuple[Any, bool, Optional[str]]:
    """
    Execute function in separate process with hard timeout.

    This uses process-based isolation to ensure a hard timeout that cannot
    be caught or suppressed by the target function or sklearn internals.

    Args:
        func: Function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        timeout_sec: Timeout in seconds (default: 72000 = 20 hours)

    Returns:
        Tuple of (result, timed_out, error_message):
        - On success: (result, False, None)
        - On timeout: (None, True, "TIMEOUT")
        - On error: (None, False, error_string)
    """
    q = Queue()
    p = Process(target=_run_in_process, args=(q, func, args, kwargs or {}))
    p.start()
    # timeout_sec=0 means no timeout (wait indefinitely)
    p.join(timeout_sec if timeout_sec else None)

    if p.is_alive():
        # Process exceeded timeout - terminate it
        p.terminate()
        p.join()
        return None, True, "TIMEOUT"

    if q.empty():
        # Process terminated but produced no result
        return None, False, "Process terminated without result"

    status, payload = q.get()
    if status == "OK":
        return payload, False, None
    else:
        # payload is error string
        return None, False, payload
