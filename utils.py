# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from functools import wraps
from itertools import izip_longest
from timeit import default_timer as timer
import traceback

import numpy as np
import rdflib
import rdflib.exceptions
from rdflib import BNode
from rdflib import Literal
from rdflib import URIRef
from rdflib import Variable
from rdflib import namespace
from rdflib.namespace import NamespaceManager
import scoop
import six


# TODO: maybe automagically get these from http://prefix.cc ?
# TODO: make this configurable
_nsm = NamespaceManager(rdflib.Graph())
_nsm.bind('owl', namespace.OWL)
_nsm.bind('xsd', namespace.XSD)
_nsm.bind('foaf', namespace.FOAF)
_nsm.bind('skos', namespace.SKOS)
_nsm.bind('doap', namespace.DOAP)
_nsm.bind('dc', namespace.DC)
_nsm.bind('dct', namespace.DCTERMS)
_nsm.bind('void', namespace.VOID)
_nsm.bind('dbpedia', 'http://dbpedia.org/resource/')  # decurification fallback
_nsm.bind('dbr', 'http://dbpedia.org/resource/')  # will curify as this
_nsm.bind('dbc', 'http://dbpedia.org/resource/Category:')
_nsm.bind('dbt', 'http://dbpedia.org/resource/Template:')
_nsm.bind('dbo', 'http://dbpedia.org/ontology/')
_nsm.bind('dbp', 'http://dbpedia.org/property/')
_nsm.bind('fb', 'http://rdf.freebase.com/')
_nsm.bind('wd', 'http://www.wikidata.org/entity/')
_nsm.bind('gold', 'http://purl.org/linguistics/gold/')
_nsm.bind('prov', 'http://www.w3.org/ns/prov#')
_nsm.bind('schema', 'http://schema.org/')


class URIShortener(object):
    """Wrapper around curify and decurify that remembers used prefixes."""
    def __init__(self, nsm=None, prefixes=None):
        if nsm is None:
            nsm = _nsm
        self.nsm = nsm
        self.prefixes = {}
        self.set_prefixes(prefixes)

    def curify(self, identifier):
        res, prefix, ns_n3 = curify(identifier, self.nsm, return_used=True)
        if prefix:
            self.prefixes[prefix] = ns_n3
        return res

    def decurify(self, n3_str):
        return decurify(n3_str, self.nsm)

    def set_prefixes(self, prefixes):
        if prefixes:
            assert isinstance(prefixes, dict)
            for pr, ns_n3 in prefixes.items():
                self.nsm.bind(pr, rdflib.util.from_n3(ns_n3), replace=True)


def curify(identifier, nsm=None, return_used=False):
    """Returns dbr:Berlin like CURIEs where possible, n3() otherwise.

    Maybe a bit of a misnomer as the result can also be a n3 representation of
    the URI if it can't be converted into a CURIE (e.g. because it contains ()).

    Most useful when trying to insert URIRefs into SPARQL queries without
    wasting a lot of space.

    >>> curify(URIRef('http://dbpedia.org/resource/Berlin'))
    u'dbr:Berlin'

    >>> curify(URIRef('http://dbpedia.org/resource/Category:Trees'))
    u'dbc:Trees'

    :param identifier: an rdflib.URIRef.
    :param nsm: A rdflib NameSpaceManager, _nsm if None.
    :param return_used: If True also return the used prefix and namespace.
    :return: by default returns a string, either consisting of the CURIE or the
        n3() of identifier. If return_used==True returns (<str>, prefix, ns_n3).
    """
    if nsm is None:
        nsm = _nsm
    assert isinstance(identifier, (BNode, Literal, URIRef, Variable)), \
        'not an identifier: %r' % (identifier,)
    if isinstance(identifier, URIRef):
        # noinspection PyBroadException
        try:
            prefix, ns, suffix = nsm.compute_qname(identifier, generate=False)
            res = ':'.join((prefix, suffix))
            if return_used:
                res = (res, prefix, ns.n3())
            return res
        except Exception:  # sadly rdflib raises this overly broad Exception
            pass
    return (identifier.n3(), None, None) if return_used else identifier.n3()


def decurify(n3_str, nsm=None):
    """Returns rdflib terms for CURIE / n3() string representations.

    >>> decurify(u'dbr:Berlin')
    rdflib.term.URIRef(u'http://dbpedia.org/resource/Berlin')

    :param n3_str: string representation.
    :param nsm: NamespaceManager, defaults to _nsm if None.
    :return: rdflib.term.identifier
    """
    assert isinstance(n3_str, six.text_type) and \
        not isinstance(n3_str, rdflib.term.Identifier)
    if nsm is None:
        nsm = _nsm
    if n3_str.startswith('?'):
        return Variable(n3_str)
    return rdflib.util.from_n3(n3_str, nsm=nsm)


def exception_stack_catcher(func):
    """Mainly useful with SCOOP as a workaround as they don't save the trace.

    Auto-logs exceptions in the wrapped function and re-raises them with
    additional attribute '_exc_fmt' which saves the formatted exception trace
    on the worker. In the main process you can check for this attribute in a
    caught exception e (e._exc_fmt) and log it as an error (see
    log_wrapped_exception below).

    As the main process might also invoke code which could raise exceptions and
    their stack traces could be hidden, you can also wrap that functionality
    with this decorator. Once an exception's stack was saved, this decorator
    will not re-log or re-modify the exception. In other words: nesting is
    supported. And not only supported, but each time you use scoop.future.map
    or the like, you should wrap the called function again.
    """
    @wraps(func)
    def exception_stack_wrapper(*args, **kwds):
        try:
            return func(*args, **kwds)
        except BaseException as e:
            if scoop.IS_RUNNING and not hasattr(e, '_exc_fmt'):
                exc_info = sys.exc_info()
                scoop.logger.exception('exception in worker')
                # noinspection PyBroadException
                try:
                    # scoop actually tries to log exception str which can cause
                    # UnicodeDecodeErrors, hence we try to work around that:
                    # see https://github.com/soravux/scoop/pull/24
                    try:
                        str(e)
                    except UnicodeEncodeError:
                        scoop.logger.warning(
                            're-packing exception for scoop, see'
                            'https://github.com/soravux/scoop/pull/24'
                        )
                        e_msg = repr(e.message)
                        six.reraise(type(e), e_msg, exc_info[2])
                    else:
                        raise
                except BaseException as err:
                    # append the stack as field to the re-raised exception
                    err._exc_fmt = 'error in worker:\n%s' % (
                        ''.join(traceback.format_exception(*exc_info)))
            raise
    return exception_stack_wrapper


def log_wrapped_exception(logger, e):
    # see exception_stack_catcher decorator
    if hasattr(e, '_exc_fmt'):
        # noinspection PyProtectedMember
        logger.error(e._exc_fmt)
    else:
        logger.exception(repr(e))


def log_all_exceptions(logger):
    """Decorator to log all local and wrapped worker exceptions to given logger.

    Useful to wrap your main function in. See log_wrapped_exception and
    exception_stack_catcher above.
    """
    def outer(func):
        @wraps(func)
        def inner(*args, **kwds):
            try:
                return exception_stack_catcher(func)(*args, **kwds)
            except Exception as err:
                log_wrapped_exception(logger, err)
                raise
        return inner
    return outer


def sample_from_list(l, probs, max_n=None):
    """Sample list according to probs.

    This method draws up to max_n items from l using the given list of probs as
    sample probabilities. max_n defaults to len(l) if not specified. Items with
    probability 0 are never sampled, so if less than max_n probabilities are > 0
    only those items are returned.

    :param l: list from which to draw items.
    :param probs: List of probabilities to draw items. Normalized by sum(probs).
    :param max_n: Optional. If given restricts max length of result, otherwise
        defaults to len(l).
    :return: list of items sampled according to probs with max length of max_n.
    """
    assert len(l) == len(probs), 'given list l and probs must have same length'
    if max_n is None:
        max_n = len(l)
    sum_probs = sum(probs)
    if sum_probs == 0:
        return []
    probs_ = np.array(probs) / sum_probs
    # we draw max n or |probs_ > 0|
    # noinspection PyTypeChecker
    n = min(max_n, np.sum(probs_ > 0))
    # use idx approach as direct passing to np.random.choice would convert
    # items of l into str
    # noinspection PyUnresolvedReferences
    res = [
        l[idx] for idx in np.random.choice(len(l), n, replace=False, p=probs_)
    ]
    return res


def sparql_json_result_bindings_to_rdflib(res_bindings):
    """Converts a result's bindings to RDFlib terms.

    Converts a results' bindings as retrieved in res["results"]["bindings"]
    by SPARQLWrapper with a sparql select query into the corresponding
    list with rdflib terms, e.g., Literal, URIref, BNode.
    BNodes won't be mixed between iterated calls of this function even if
    they happen to have the same "value". Internally the given value is mapped
    to a random value, which is remembered in _one and the same_ call of this
    function only.
    """
    _bnodes = {}  # makes sure we don't confuse BNodes from different results

    def dict_to_rdflib(d):
        """Maps a dict to the corresponding rdflib term.

        Follows the syntax in http://www.w3.org/TR/rdf-sparql-json-res/ .
        """
        if d is None:
            return None

        t = d["type"]
        v = d["value"]

        if t == "uri":
            return URIRef(v)

        if t == "bnode":
            if v not in _bnodes:
                # v is not used as BNode value on purpose (multiple calls should
                # not have the same value)
                _bnodes[v] = BNode()
            return _bnodes[v]

        l = d.get("xml:lang", None)
        if t == "literal":
            return Literal(v, lang=l)

        if t == "typed-literal":
            # will raise type error if lang and datatype set
            return Literal(v, lang=l, datatype=d["datatype"])

        raise rdflib.exceptions.ParserError(
            "Invalid sparql json result according to "
            "http://www.w3.org/TR/rdf-sparql-json-res/: {0}".format(d))

    res_bindings_rdflib = []
    for row in res_bindings:
        tmp = {}
        for var_name, value in row.items():
            tmp[Variable(var_name)] = dict_to_rdflib(value)
        res_bindings_rdflib.append(tmp)

    return res_bindings_rdflib
