# -*- coding: utf-8 -*-


class GPLearnerException(Exception):
    """Base class for all graph pattern learner exceptions."""
    pass


class GPLearnerAbortException(GPLearnerException):
    """Exception base class to be raised to abort whole GP Learner."""
    pass


class GPLearnerTestPatternFoundException(GPLearnerAbortException):
    """Exception to be thrown when a test pattern is found.

    This exception is used in eval mode to enable early termination as soon as
    a test pattern (to be searched) ends up in the results of one generation.
    Early termination here allows not wasting additional time for the full
    run to complete. It skips out of the code immediately, re-raising the
    exception for external (eval) handling.
    """
    pass


class QueryException(GPLearnerException):
    """Base class for exceptions thrown in query execution."""
    pass


class IncompleteQueryException(QueryException):
    """Tried to eval a query without source and target var."""
    pass


class MultiQueryException(QueryException):
    """Exception thrown from _multi_query code."""
    pass


class MultiQueryClosedException(MultiQueryException):
    """Exception thrown on _multi_query call if previous call failed.

    Mostly to be thrown ."""
    pass


