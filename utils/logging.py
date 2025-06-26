import logging


def setup_logging(level=logging.INFO,
                  fmt="%(asctime)s [%(levelname)s] %(message)s"):
    """Configure logging for the project.

    Parameters
    ----------
    level : int, optional
        Logging level, by default ``logging.INFO``.
    fmt : str, optional
        Log message format, by default ``"%(asctime)s [%(levelname)s] %(message)s"``.
    """
    logging.basicConfig(level=level, format=fmt)

