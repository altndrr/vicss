import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from lightning.pytorch import utilities as pl_utils


def get_logger(name: str = __name__, rank_zero_only: bool = False) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger.

    Args:
        name: The name of the logger, defaults to ``__name__``.
        rank_zero_only: If True, the logger will only log on the process with rank 0.
    """

    def rank_prefixed_log(func: Callable) -> Callable:
        """Wrapper to prefix a log message with the its local rank.

        If `rank` is provided in the wrapped functions kwargs, then the log will only occur on
        that rank/process.

        Args:
            func: The function to wrap.
        """

        @wraps(func)
        def inner(*inner_args, rank_to_log: int | None = None, **inner_kwargs) -> Any:
            rank = getattr(pl_utils.rank_zero_only, "rank", None)
            if rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")

            # add the rank to the extra kwargs
            extra = inner_kwargs.pop("extra", {})
            extra.update({"rank": rank})

            if rank_zero_only:
                if rank == 0:
                    return func(msg=inner_args[0], *inner_args[1:], extra=extra, **inner_kwargs)
            elif rank_to_log is None or rank == rank_to_log:
                return func(msg=inner_args[0], *inner_args[1:], extra=extra, **inner_kwargs)
            else:
                return None

        return inner

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank_prefixed_log decorator
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_prefixed_log(getattr(logger, level)))

    return logger
