from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncIterator

from core.log_config import logger
from uni_api.admission import get_request_admission_lease


BACKGROUND_STREAM_CLEANUP_TASKS: set[asyncio.Task[Any]] = set()
try:
    BACKGROUND_STREAM_CLEANUP_MAX_TASKS = max(
        1,
        int(os.getenv("BACKGROUND_STREAM_CLEANUP_MAX_TASKS", "8")),
    )
except (TypeError, ValueError):
    BACKGROUND_STREAM_CLEANUP_MAX_TASKS = 8
try:
    STREAM_CLEANUP_TIMEOUT_SECONDS = max(
        0.001,
        float(os.getenv("STREAM_CLEANUP_TIMEOUT_SECONDS", "1.0")),
    )
except (TypeError, ValueError):
    STREAM_CLEANUP_TIMEOUT_SECONDS = 1.0


def background_stream_cleanup_snapshot() -> dict[str, int]:
    tasks = list(BACKGROUND_STREAM_CLEANUP_TASKS)
    return {
        "pending": sum(1 for task in tasks if not task.done()),
        "done": sum(1 for task in tasks if task.done()),
        "total": len(tasks),
    }


async def wait_background_stream_cleanup_tasks(timeout: float = 5.0) -> dict[str, int]:
    tasks = [task for task in list(BACKGROUND_STREAM_CLEANUP_TASKS) if not task.done()]
    if not tasks:
        return background_stream_cleanup_snapshot()

    done, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout))
    cancelled_done: set[asyncio.Task[Any]] = set()
    for task in pending:
        task.cancel()
    if pending:
        cancelled_done, still_pending = await asyncio.wait(
            pending,
            timeout=min(1.0, max(0.0, timeout)),
        )
        done.update(cancelled_done)
        logger.warning(
            "Cancelled detached stream cleanup tasks during shutdown: done=%d still_pending=%d",
            len(cancelled_done),
            len(still_pending),
        )

    BACKGROUND_STREAM_CLEANUP_TASKS.difference_update(done)
    BACKGROUND_STREAM_CLEANUP_TASKS.difference_update(cancelled_done)
    snapshot = background_stream_cleanup_snapshot()
    snapshot["completed_during_wait"] = len(done)
    snapshot["cancelled_during_wait"] = len(pending)
    return snapshot


def drain_current_task_cancellation() -> None:
    current_task = asyncio.current_task()
    uncancel = getattr(current_task, "uncancel", None)
    if callable(uncancel):
        while current_task is not None and current_task.cancelling():
            uncancel()


def track_background_stream_cleanup_task(task: asyncio.Task[Any], *, label: str) -> bool:
    if len(BACKGROUND_STREAM_CLEANUP_TASKS) >= BACKGROUND_STREAM_CLEANUP_MAX_TASKS:
        logger.error(
            "Detached stream cleanup capacity exhausted; label=%s capacity=%d",
            label,
            BACKGROUND_STREAM_CLEANUP_MAX_TASKS,
        )
        return False
    BACKGROUND_STREAM_CLEANUP_TASKS.add(task)

    def cleanup_done(done: asyncio.Task[Any]) -> None:
        BACKGROUND_STREAM_CLEANUP_TASKS.discard(done)
        if done.cancelled():
            logger.warning("%s cleanup task was cancelled after detach", label)
            return
        try:
            done.result()
        except BaseException as exc:
            logger.warning(
                "%s cleanup failed after detach",
                label,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    task.add_done_callback(cleanup_done)
    return True


async def _await_ownership_transaction(
    task: asyncio.Task[Any],
) -> tuple[Any | None, asyncio.CancelledError | None, BaseException | None]:
    """Shield a handoff transaction and report cancellation only afterward."""

    pending_cancel: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            pending_cancel = pending_cancel or exc
    if task.cancelled():
        return None, pending_cancel, asyncio.CancelledError()
    try:
        return task.result(), pending_cancel, None
    except BaseException as exc:
        return None, pending_cancel, exc


async def _detach_or_finish_bounded(
    task: asyncio.Task[Any],
    *,
    label: str,
    transport_isolated: bool = False,
) -> bool:
    request_lease = get_request_admission_lease()

    async def finish_without_releasing_ownership() -> None:
        while not task.done():
            drain_current_task_cancellation()
            try:
                await asyncio.shield(task)
            except (asyncio.CancelledError, GeneratorExit):
                continue
            except BaseException:
                break
        task.result()

    if request_lease is not None and not request_lease.released:
        if transport_isolated:
            async def register_isolated_cleanup() -> bool:
                deferral = await request_lease.defer_memory_release()
                supervisor: asyncio.Task[None] | None = None
                registered = False
                try:
                    async def supervise_with_memory_owner() -> None:
                        try:
                            await finish_without_releasing_ownership()
                        finally:
                            await deferral.release()

                    # No await occurs between task creation and registration,
                    # so a capacity rejection can cancel a not-yet-started
                    # supervisor without racing its deferral finalizer.
                    supervisor = asyncio.create_task(
                        supervise_with_memory_owner()
                    )
                    registered = track_background_stream_cleanup_task(
                        supervisor,
                        label=label,
                    )
                    if registered:
                        return True
                    supervisor.cancel()
                    try:
                        await supervisor
                    except asyncio.CancelledError:
                        pass
                    return False
                finally:
                    if not registered:
                        await deferral.release()

            registration_task = asyncio.create_task(register_isolated_cleanup())
            registered, pending_cancel, registration_error = (
                await _await_ownership_transaction(registration_task)
            )
            if registration_error is not None:
                exc = registration_error
                logger.warning(
                    "%s could not defer request memory after transport isolation",
                    label,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            elif registered:
                if pending_cancel is not None:
                    raise pending_cancel
                return True

            # A failed/full registration must remain fail-closed.  Run the
            # wait itself as another shielded transaction so an observed
            # cancellation cannot release request ownership first.
            finish_task = asyncio.create_task(finish_without_releasing_ownership())
            _result, finish_cancel, finish_error = (
                await _await_ownership_transaction(finish_task)
            )
            if finish_error is not None and not isinstance(
                finish_error,
                asyncio.CancelledError,
            ):
                logger.warning(
                    "%s cleanup failed while retaining request ownership",
                    label,
                    exc_info=(
                        type(finish_error),
                        finish_error,
                        finish_error.__traceback__,
                    ),
                )
            cancellation = pending_cancel or finish_cancel
            if cancellation is not None:
                raise cancellation
            return False
        # There is no sound generic byte estimate for arbitrary state captured
        # by a non-cooperative aclose coroutine.  Fail closed: retain the real
        # active request and all of its memory ownership until cleanup exits.
        # This can consume at most the bounded active-request envelope and
        # cannot accumulate detached unaccounted objects across new requests.
        try:
            await finish_without_releasing_ownership()
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                logger.warning(
                    "%s cleanup failed while retaining request ownership",
                    label,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
        return False

    if track_background_stream_cleanup_task(task, label=label):
        return True
    # Fail closed once the detached cleanup envelope is full.  At most the
    # already admitted request tasks can wait here, so stuck cleanup work
    # cannot grow without bound across successive requests.
    try:
        await finish_without_releasing_ownership()
    except BaseException as exc:
        if not isinstance(exc, asyncio.CancelledError):
            logger.warning(
                "%s cleanup failed while detached capacity was full",
                label,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
    return False


async def await_stream_cleanup_safely(
    awaitable: Any,
    *,
    label: str,
    timeout_seconds: float | None = None,
    _transport_isolated: bool = False,
) -> bool:
    if awaitable is None or not hasattr(awaitable, "__await__"):
        return True

    drain_current_task_cancellation()
    cleanup_task = asyncio.ensure_future(awaitable)
    timeout = (
        STREAM_CLEANUP_TIMEOUT_SECONDS
        if timeout_seconds is None
        else max(0.001, float(timeout_seconds))
    )
    deadline = asyncio.get_running_loop().time() + timeout
    while not cleanup_task.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            logger.warning(
                "%s cleanup exceeded %.3f seconds; cancelling with bounded detach",
                label,
                timeout,
            )
            cleanup_task.cancel()
            await _detach_or_finish_bounded(
                cleanup_task,
                label=label,
                transport_isolated=_transport_isolated,
            )
            return False
        try:
            await asyncio.wait({cleanup_task}, timeout=remaining)
        except asyncio.CancelledError:
            drain_current_task_cancellation()
            logger.warning(
                "%s cleanup owner was cancelled; cleanup remains bounded by its deadline",
                label,
            )
            continue
        except GeneratorExit as exc:
            logger.warning(
                "%s cleanup was interrupted by generator close; detached cleanup will continue",
                label,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            cleanup_task.cancel()
            await _detach_or_finish_bounded(
                cleanup_task,
                label=label,
                transport_isolated=_transport_isolated,
            )
            return False

    if cleanup_task.cancelled():
        return False
    try:
        cleanup_task.result()
    except BaseException as exc:
        logger.warning(
            "%s cleanup failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False
    return True


async def await_isolated_transport_cleanup_safely(
    awaitable: Any,
    *,
    label: str,
    timeout_seconds: float | None = None,
) -> bool:
    """Bound cleanup after a connection is closed or evicted from its pool.

    Once transport reuse is impossible, the active request slot may return to
    service.  A MemoryReleaseDeferral keeps all known request bytes charged
    until the bounded detached cleanup task truly exits.
    """

    return await await_stream_cleanup_safely(
        awaitable,
        label=label,
        timeout_seconds=timeout_seconds,
        _transport_isolated=True,
    )


async def close_async_iterator_safely(iterator: Any, *, label: str) -> bool:
    aclose = getattr(iterator, "aclose", None)
    if not callable(aclose):
        return True
    try:
        close_result = aclose()
    except RuntimeError as exc:
        logger.debug("%s async iterator close skipped: %s", label, exc)
        return True
    except BaseException as exc:
        logger.warning(
            "%s async iterator close failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False
    return await await_stream_cleanup_safely(close_result, label=label)


async def call_cleanup_safely(cleanup: Any, *, label: str) -> bool:
    try:
        result = cleanup()
    except BaseException as exc:
        logger.warning(
            "%s cleanup failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False
    return await await_stream_cleanup_safely(result, label=label)


async def _wait_cleanup_task_bounded(
    task: asyncio.Task[Any],
    *,
    timeout_seconds: float,
    label: str,
) -> bool:
    deadline = asyncio.get_running_loop().time() + max(
        0.001,
        float(timeout_seconds),
    )
    while not task.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return False
        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            drain_current_task_cancellation()
            logger.warning(
                "%s cleanup owner was cancelled during forced close",
                label,
            )
        except GeneratorExit:
            return False
    return True


def _cleanup_task_result(task: asyncio.Task[Any], *, label: str) -> bool:
    if task.cancelled():
        return False
    try:
        task.result()
    except BaseException as exc:
        logger.warning(
            "%s cleanup failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False
    return True


def _evict_httpcore_pool_request(
    stream: Any,
    *,
    label: str,
) -> tuple[Any | None, list[Any], bool]:
    """Synchronously make a PoolByteStream impossible to reuse.

    Connection close is deliberately returned to the caller.  This lets the
    caller release the per-upstream admission lease immediately after the
    pool entry is gone, before waiting on a potentially non-cooperative socket
    close.
    """

    pool = getattr(stream, "_pool", None)
    pool_request = getattr(stream, "_pool_request", None)
    if pool is None or pool_request is None:
        return None, [], False

    requests = getattr(pool, "_requests", None)
    pool_connections = getattr(pool, "_connections", None)
    connection = getattr(pool_request, "connection", None)
    if not isinstance(requests, list):
        requests = []
    if pool_request not in requests and connection is None:
        return pool, [], False

    try:
        closing: list[Any] = []

        def evict_locked() -> None:
            nonlocal closing
            if pool_request in requests:
                requests.remove(pool_request)
            if isinstance(pool_connections, list) and connection in pool_connections:
                pool_connections.remove(connection)
            # A later PoolByteStream.aclose() must not try to remove the same
            # request a second time.  If an aclose is already inside its
            # guarded branch it may still report a harmless ValueError; the
            # pool and connection are nevertheless already unreachable.
            if hasattr(stream, "_closed"):
                stream._closed = True
            clear_connection = getattr(pool_request, "clear_connection", None)
            if callable(clear_connection):
                clear_connection()
            else:
                try:
                    pool_request.connection = None
                except BaseException:
                    pass
            assign_requests = getattr(pool, "_assign_requests_to_connections", None)
            if callable(assign_requests):
                closing = list(assign_requests())

        lock = getattr(pool, "_optional_thread_lock", None)
        if lock is not None:
            with lock:
                evict_locked()
        else:
            evict_locked()

        if connection is not None and all(candidate is not connection for candidate in closing):
            closing.append(connection)
        return pool, closing, True
    except BaseException as exc:
        logger.warning(
            "%s pool eviction failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return pool, [], False


async def _close_evicted_connections_safely(
    pool: Any,
    closing: list[Any],
    *,
    label: str,
) -> bool:
    if not closing:
        return True
    close_connections = getattr(pool, "_close_connections", None)
    if callable(close_connections):
        return await await_isolated_transport_cleanup_safely(
            close_connections(closing),
            label=label,
        )
    cleanup_ok = True
    for connection_to_close in closing:
        aclose = getattr(connection_to_close, "aclose", None)
        if callable(aclose):
            try:
                close_result = aclose()
            except BaseException as exc:
                logger.warning(
                    "%s connection cleanup failed",
                    label,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                cleanup_ok = False
            else:
                cleanup_ok = await await_isolated_transport_cleanup_safely(
                    close_result,
                    label=f"{label} connection",
                ) and cleanup_ok
    return cleanup_ok


async def force_release_httpcore_pool_request_safely(stream: Any, *, label: str) -> bool:
    pool, closing, evicted = _evict_httpcore_pool_request(
        stream,
        label=label,
    )
    if pool is None:
        return True
    if not evicted and not closing:
        return True
    return await _close_evicted_connections_safely(
        pool,
        closing,
        label=label,
    )


async def _abort_bound_response_lease_safely(
    bound_close_owner: Any,
    *,
    label: str,
) -> bool:
    abort_transport = getattr(bound_close_owner, "abort_transport", None)
    if not callable(abort_transport):
        return True
    try:
        await abort_transport()
    except BaseException as exc:
        logger.warning(
            "%s upstream lease abort failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False
    return True


async def force_close_response_httpcore_stream_chain_safely(response: Any | None, *, label: str) -> bool:
    if response is None:
        return True

    response_aclose = getattr(response, "aclose", None)
    bound_close_owner = getattr(response_aclose, "__self__", None)
    stream = getattr(response, "stream", None)
    candidates: list[Any] = []
    current = stream
    seen: set[int] = set()
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)
        candidates.append(current)
        current = getattr(current, "_stream", None)

    close_task: asyncio.Task[Any] | None = None
    if callable(response_aclose):
        try:
            close_result = response_aclose()
        except BaseException as exc:
            logger.warning(
                "%s cooperative close failed to start",
                label,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        else:
            if hasattr(close_result, "__await__"):
                close_task = asyncio.ensure_future(close_result)

    cooperative_failed = False
    if close_task is not None and await _wait_cleanup_task_bounded(
        close_task,
        timeout_seconds=STREAM_CLEANUP_TIMEOUT_SECONDS,
        label=label,
    ):
        if _cleanup_task_result(close_task, label=label):
            return True
        cooperative_failed = True
        close_task = None

    if close_task is not None:
        logger.warning(
            "%s cooperative close exceeded %.3f seconds; evicting transport",
            label,
            STREAM_CLEANUP_TIMEOUT_SECONDS,
        )

    cleanup_ok = not cooperative_failed
    evicted_any = False
    by_pool: dict[int, tuple[Any, list[Any]]] = {}
    for candidate in candidates:
        pool, closing, evicted = _evict_httpcore_pool_request(
            candidate,
            label=label,
        )
        evicted_any = evicted_any or evicted
        if pool is None or not closing:
            continue
        pool_key = id(pool)
        if pool_key not in by_pool:
            by_pool[pool_key] = (pool, [])
        collected = by_pool[pool_key][1]
        for connection in closing:
            if all(existing is not connection for existing in collected):
                collected.append(connection)

    if evicted_any:
        cleanup_ok = await _abort_bound_response_lease_safely(
            bound_close_owner,
            label=label,
        ) and cleanup_ok

    if close_task is not None and not close_task.done():
        close_task.cancel()

    for pool, closing in by_pool.values():
        cleanup_ok = await _close_evicted_connections_safely(
            pool,
            closing,
            label=f"{label} evicted connection",
        ) and cleanup_ok

    if close_task is not None:
        if not await _wait_cleanup_task_bounded(
            close_task,
            timeout_seconds=STREAM_CLEANUP_TIMEOUT_SECONDS,
            label=label,
        ):
            logger.error(
                "%s remained stuck after transport eviction; retaining request ownership",
                label,
            )
            detached = await _detach_or_finish_bounded(
                close_task,
                label=label,
                transport_isolated=evicted_any,
            )
            return bool(evicted_any and detached)
        cleanup_ok = _cleanup_task_result(close_task, label=label) and cleanup_ok
    elif not evicted_any:
        # There was neither a cooperative close nor an identifiable pool
        # request.  Report failure rather than claiming the lease is safe.
        cleanup_ok = False
    # The return value describes transport safety, not whether every best-
    # effort close coroutine exited cleanly.  An evicted pool request cannot
    # be reused even while its bounded cleanup supervisor is still running.
    return bool(evicted_any or cleanup_ok)


async def yield_from_stream(stream: AsyncIterator[Any], *, label: str) -> AsyncIterator[Any]:
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await close_async_iterator_safely(stream, label=label)
