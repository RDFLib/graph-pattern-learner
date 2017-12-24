#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import Iterable
from collections import OrderedDict
from collections import Sequence
from collections import defaultdict
from collections import namedtuple
from copy import deepcopy
from itertools import chain
import logging
import random
import string
import textwrap

import deap
import deap.base
import networkx as nx
import rdflib
import rdflib.compare
import rdflib.term
from rdflib import BNode
from rdflib import Graph
from rdflib import RDF
from rdflib import URIRef
from rdflib import Variable
import six

from utils import URIShortener

logger = logging.getLogger(__name__)


RANDOM_VAR_LEN = 5  # so in total we have 62**5=916132832 different random vars
RANDOM_VAR_PREFIX = 'vr'
SOURCE_VAR = Variable('source')
TARGET_VAR = Variable('target')
ASK_VAR = Variable('ask')
COUNT_VAR = Variable('count')


def gen_random_var():
    return Variable(RANDOM_VAR_PREFIX + ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(RANDOM_VAR_LEN)
    ))


def replace_vars_with_random_vars(triples, exclude=(SOURCE_VAR, TARGET_VAR)):
    rv = defaultdict(gen_random_var)
    return [
        tuple([
            rv[ti] if isinstance(ti, Variable) and ti not in exclude else ti
            for ti in t
        ])
        for t in triples
    ]


def to_nx_graph(gp):
    g = nx.Graph([(s, o) for s, p, o in gp])
    return g


def to_nx_digraph(gp):
    dg = nx.DiGraph([(s, o) for s, p, o in gp])
    return dg


def to_nx_graph_via_edge_nodes(gp):
    """Models each triple (s, p, o) as two edges via edge node (s, p), (p, o).

    Useful to check if gp is "connected" via edges if that is considered being
    connected.

    Graph theoretically this is losing information if any other triple uses p.
    E.g., (s, p, o) and (x, p, y) will end up as (s, p), (p, o), (x, p), (p, y)
    and we can't reconstruct the original triples. If you need that have a look
    at the bipartite hypergraph equivalent below.
    """
    g = nx.Graph([(s, p) for s, p, o in gp] + [(p, o) for s, p, o in gp])
    return g


def to_nx_graph_as_bipartite_hypergraph_equivalent(gp):
    g = nx.Graph([(i, (s, p, o)) for s, p, o in gp for i in (s, p, o)])
    return g


def canonicalize_gp_to_rdf_graph(gp, fixed_vars=None):
    assert isinstance(gp, Iterable), "gp not iterable: %r" % gp
    if fixed_vars is None:
        fixed_vars = set()

    triple_bnodes = set()
    g = Graph()
    for t in gp:
        triple_bnode = BNode()
        assert triple_bnode not in triple_bnodes, \
            "%r triple_bnode %r not meant to be in triple_bnodes %r" % (
                gp, triple_bnode, triple_bnodes)
        trip = []
        for i in t:
            if isinstance(i, Variable):
                if i in fixed_vars:
                    trip.append(URIRef('urn:gp_learner:fixed_var:%s' % i))
                else:
                    trip.append(BNode(i))
            else:
                trip.append(i)
        s, p, o = trip
        g.add((triple_bnode, RDF['type'], RDF['Statement']))
        g.add((triple_bnode, RDF['subject'], s))
        g.add((triple_bnode, RDF['predicate'], p))
        g.add((triple_bnode, RDF['object'], o))
        triple_bnodes.add(triple_bnode)
    return g


def canonicalize_rdf_cg_to_gp(cg):
    cgp = []
    for triple_bnode in cg.subjects(RDF['type'], RDF['Statement']):
        assert isinstance(triple_bnode, BNode), \
            "expected BNode, got %r in %r" % (triple_bnode, list(cg))
        t = [
            cg.value(triple_bnode, p)
            for p in [RDF['subject'], RDF['predicate'], RDF['object']]
        ]
        trip = []
        for i in t:
            if isinstance(i, BNode):
                trip.append(Variable(i))
            else:
                if isinstance(i, URIRef) and \
                        i.startswith('urn:gp_learner:fixed_var:'):
                    trip.append(Variable(i[25:]))
                else:
                    trip.append(i)
        t = tuple(trip)
        cgp.append(t)
    return sorted(cgp)


def canonicalize_sparql_bgp(gp, fixed_vars=None):
    """Returns a canonical basic graph pattern (BGP) with canonical var names.

    :param gp: a GraphPattern in form of a list of triples with Variables.
    :param fixed_vars: A set of variables that should not be canonicalized.
    :return: A canonical GraphPattern as list with Variables renamed.

    >>> U = URIRef
    >>> V = Variable
    >>> gp1 = [
    ...     (V('blub'), V('bar'), U('blae')),
    ...     (V('foo'), V('bar'), U('bla')),
    ...     (V('foo'), U('poo'), U('blub')),
    ... ]
    >>> cgp = canonicalize_sparql_bgp(gp1)
    >>> len(cgp)
    3
    >>> v_blub = cgp[[t[2] for t in cgp].index(U('blae'))][0]
    >>> v_bar = cgp[[t[2] for t in cgp].index(U('blae'))][1]
    >>> v_foo = cgp[[t[2] for t in cgp].index(U('bla'))][0]
    >>> expected = [
    ...     (v_blub, v_bar, U('blae')),
    ...     (v_foo, v_bar, U('bla')),
    ...     (v_foo, U('poo'), U('blub'))
    ... ]
    >>> cgp == expected
    True

    To show that this is variable name and order independent we shuffle gp1 and
    rename its vars:
    >>> gp2 = [
    ...     (V('foonkyname'), V('baaar'), U('bla')),
    ...     (V('foonkyname'), U('poo'), U('blub')),
    ...     (V('funkyname'), V('baaar'), U('blae')),
    ... ]
    >>> cgp == canonicalize_sparql_bgp(gp2)
    True

    """
    g = canonicalize_gp_to_rdf_graph(gp, fixed_vars)
    cg = rdflib.compare.to_canonical_graph(g)
    cgp = canonicalize_rdf_cg_to_gp(cg)
    return cgp


def canonicalize(gp, shorten_varnames=True):
    """Returns a canonical basic graph pattern (BGP) with canonical var names.

    :param gp: a GraphPattern in form of a list of triples with Variables
    :param shorten_varnames: If True (default) long sha256 based var-names will
        be renamed to short enumerated ones.
    :return: A canonical GraphPattern with Variables renamed.

    >>> U = URIRef
    >>> V = Variable
    >>> gp1 = GraphPattern([
    ...     (V('blub'), V('bar'), U('blae')),
    ...     (V('foo'), V('bar'), U('bla')),
    ...     (SOURCE_VAR, V('poo'), TARGET_VAR),
    ... ])
    >>> cgp = canonicalize(gp1)
    >>> v_poo = cgp[[t[2] for t in cgp].index(TARGET_VAR)][1]
    >>> v_foo = cgp[[t[2] for t in cgp].index(U('bla'))][0]
    >>> v_bar = cgp[[t[2] for t in cgp].index(U('bla'))][1]
    >>> v_blub = cgp[[t[2] for t in cgp].index(U('blae'))][0]
    >>> expected = GraphPattern([
    ...     (SOURCE_VAR, v_poo, TARGET_VAR),
    ...     (v_foo, v_bar, U('bla')),
    ...     (v_blub, v_bar, U('blae')),
    ... ])
    >>> cgp == expected
    True

    And again in a different order:
    >>> gp2 = GraphPattern([
    ...     (SOURCE_VAR, V('bla'), TARGET_VAR),
    ...     (V('blub'), V('bli'), U('bla')),
    ...     (V('bluub'), V('bli'), U('blae')),
    ... ])
    >>> cgp == canonicalize(gp2)
    True

    """
    assert isinstance(gp, GraphPattern)
    cbgp = canonicalize_sparql_bgp(gp, fixed_vars={SOURCE_VAR, TARGET_VAR})
    mapping = {}
    if shorten_varnames:
        vars_ = set(chain.from_iterable(cbgp))
        vars_ = sorted([
            v for v in vars_ if isinstance(v, Variable) and v.startswith('cb')
        ])
        for i, v in enumerate(vars_):
            mapping[v] = Variable('vcb%d' % i)
    cgp = GraphPattern(cbgp, mapping=mapping)

    if not (
        len(gp) == len(cbgp) == len(cgp)
        and len(gp.nodes) == len(cgp.nodes)
        and len(gp.edges) == len(cgp.edges)
        and sorted(gp.identifier_counts().values()) ==
            sorted(cgp.identifier_counts().values())
    ):
        # canonicalization should never change any of the features above, but it
        # did before (e.g., https://github.com/RDFLib/rdflib/issues/494 ).
        # this is a last resort safety-net
        logger.warning(
            'GraphPattern canonicalization failed, returning original:\n%r\n'
            'Canonicalized RDF Graph:\n%r\n'
            'Canonicalized Graph Pattern:\n%r\n',
            gp, cbgp, cgp
        )
        return gp

    return cgp


class GPFitness(deap.base.Fitness):
    """Fitness of a GraphPattern.

    This is a specialised DEAP Fitness object, following all their rules, but
    adding some shortcuts to easily access the multi-dimensional components.
    """

    # see gp_learner.evaluate for how these are calculated and more info
    components = (
        (1., "remains"),  # remaining precision sum in this "run"
        (1., "score"),  # trust (1-timeout) * overfitting * gain
        (1., "gain"),  # gained precision sum over remains of gtps
        (1., "f_measure"),  # f1_measure
        (-1., "avg_reslens"),  # given a ?source how many ?targets on avg?
        (1., "gt_matches"),  # how many gtps match?
        (-1., "patlen"),  # triple count
        (-1., "patvars"),  # var count
        (-1., "timeout"),  # did a soft(.5) / hard(1.) timeout occur?
        (-1., "qtime"),  # query time in seconds
    )
    weights, description_list = zip(*components)
    description = "(%s)" % ", ".join(description_list)

    def getValues(self):
        return self._values

    def setValues(self, values):
        super(GPFitness, self).setValues(values)
        self._values = GPFitnessTuple(*values)

    def delValues(self):
        super(GPFitness, self).delValues()
        self._values = ()

    values = property(
        getValues, setValues, delValues,
        "Fitness values. Use directly ``individual.fitness.values = values`` "
        "in order to set the fitness and ``del individual.fitness.values`` "
        "in order to clear (invalidate) the fitness. The (unweighted) fitness "
        "can be directly accessed via ``individual.fitness.values``."
    )

    def __init__(self, values=()):
        self._values = ()
        super(GPFitness, self).__init__(values)

    def __deepcopy__(self, memo):
        copy_ = super(GPFitness, self).__deepcopy__(memo)
        copy_._values = self._values
        return copy_

    def format_fitness(self):
        if self.valid:
            return '(%s)' % ', '.join(
                [('%.4f' % x).rstrip('0').rstrip('.') for x in self.values])
        else:
            return '(not evaluated yet)'


GPFitnessTuple = namedtuple('GPFitnessTuple', GPFitness.description_list)
GPFitnessTuple.__new__.__defaults__ = tuple([0] * len(GPFitness.weights))


class GraphPattern(tuple):
    """A GraphPattern is mostly a tuple of triples with Variables in them.

    There are two special variables: SOURCE_VAR and TARGET_VAR that are used for
    source and target nodes. A GraphPattern provides all standard tuple
    operations, plus methods to generate a SPARQL representation / queries and
    a replace method which generates a new GraphPattern based on the current one
    just with the given identifiers replaced.
    """

    # tuple sub-classes don't seem to support slots, to be optimized if memory
    # consumption gets too high:
    # __slots__ = (
    #     'vars_in_graph',
    #     'fitness',
    # )

    def __new__(
            cls,
            triples,
            source_node=None,
            target_node=None,
            mapping=None,
    ):
        """Creates a new GraphPattern.

        Args:
            triples: an rdflib.Graph, GraphPattern or iterable to be copied
            source_node: the original source node of this pattern. used for
                provenance of the pattern and also implicitly adds a mapping
                source_node: Variable('source') to the mapping mapping,
                which replaces the source_node with a variable.
            target_node: analog to source_node
            mapping: a dictionary of rdflib.terms.Identifiers to others (most
                notably rdflib.Variables) or None. The given mappings are
                applied during creation for the new GraphPattern.
        """
        assert mapping is None or isinstance(mapping, dict), \
            'mapping should be a dict: %r' % mapping
        assert isinstance(triples, Iterable), \
            "triples not iterable: %r" % triples
        triples = set(triples)
        assert not triples or isinstance(next(iter(triples)), tuple)
        mapping = mapping.copy() if mapping else {}
        if source_node is not None and source_node not in mapping:
            mapping[source_node] = SOURCE_VAR
        if target_node is not None and target_node not in mapping:
            mapping[target_node] = TARGET_VAR
        return tuple.__new__(cls, sorted({
            tuple([mapping[ti] if ti in mapping else ti for ti in t])
            for t in triples
        }))

    # noinspection PyUnusedLocal
    def __init__(
            self,
            triples,
            source_node=None,
            target_node=None,
            mapping=None
    ):
        """See __new__ for docs!"""
        tuple.__init__(self)
        self.vars_in_graph = set(
            i for t in self for i in t if isinstance(i, Variable)
        )
        self.fitness = GPFitness()
        self.matching_node_pairs = []
        self.gtp_precisions = OrderedDict()
        self._uri_shortener = URIShortener()

    def replace(self, mapping):
        """Replace Identifiers in pattern with others according to mapping.

        :param mapping: A dictionary mapping rdflib.term.Identifiers to
            be replaced with others, most probably rdflib.Variables.
        :return: A new graph pattern based on the current one with all
            occurrences of any of the mapping's keys replaced with their
            corresponding values.
        """
        return GraphPattern(self, mapping=mapping)

    def only_with(self, identifiers):
        """Return a new pattern of triples if they include any of identifiers.

        :param identifiers: set like
        :return: A new graph pattern based on the current one with all triples
            that at least contain one of identifiers
        """
        assert identifiers
        return GraphPattern(
            [(s, p, o)
             for s, p, o in self
             if o in identifiers or s in identifiers or p in identifiers]
        )

    def exclude(self, identifiers):
        """Return a new pattern of triples including none of identifiers.

        :param identifiers: set like
        :return: A new graph pattern based on the current one with all triples
            that contain none of the identifiers
        """
        assert identifiers
        return GraphPattern(
            [(s, p, o)
             for s, p, o in self
             if p not in identifiers and
                s not in identifiers and
                o not in identifiers
             ]
        )

    def complete(self):
        _vars = self.vars_in_graph
        return SOURCE_VAR in _vars and TARGET_VAR in _vars

    def identifier_counts(self, exclude_vars=False, vars_only=False):
        """Returns a Counter of identifiers in this graph pattern.

        :param exclude_vars: If True will exclude Variables from result.
        :param vars_only: Only return counts for vars.
        :return: Counter of all identifiers in this graph pattern.
        """
        assert not(exclude_vars and vars_only)
        ids = Counter([i for t in self for i in t])
        if exclude_vars:
            for i in self.vars_in_graph:
                del ids[i]
        if vars_only:
            for i in list(ids):
                if not isinstance(i, Variable):
                    del ids[i]
        return ids

    def var_counts(self):
        return self.identifier_counts(vars_only=True)

    @property
    def nodes(self):
        return {n for t in self for n in t[::2]}

    @property
    def edges(self):
        return {p for _, p, _ in self}

    def node_vars(self):
        return {n for n in self.nodes if isinstance(n, Variable)}

    def edge_vars(self):
        return {p for p in self.edges if isinstance(p, Variable)}

    def triples_by_identifier(self, identifiers=None, positions=None):
        if identifiers is None:
            identifiers = self.identifier_counts()
        if positions is None:
            positions = range(3)
        res = {i: set() for i in identifiers}
        for t in self:
            for i in positions:
                identifier = t[i]
                if identifier in identifiers:
                    res[identifier].add(t)
        return res

    def triples_by_nodes(self, nodes=None):
        if nodes is None:
            nodes = self.nodes
        return self.triples_by_identifier(nodes, [0, 2])

    def triples_by_edges(self, edges=None):
        if edges is None:
            edges = self.edges
        return self.triples_by_identifier(edges, [1])

    def curify(self, identifier):
        return self._uri_shortener.curify(identifier)

    def decurify(self, n3_str):
        return self._uri_shortener.decurify(n3_str)

    @property
    def prefixes(self):
        return self._uri_shortener.prefixes

    def _sparql_prefix(self, q):
        """Prefixes query q with the necessary prefix clauses.

        Call after all curification, otherwise prefix dict might not be filled.
        """
        return ''.join(
            'PREFIX %s: %s\n' % (pr, ns_n3)
            for pr, ns_n3 in self.prefixes.items()
        ) + q

    def to_sparql_select_query(
            self,
            projection=None,
            distinct=False,
            count=None,
            bind=None,
            values=None,
            limit=None,
    ):
        """Generates a SPARQL select query from the graph pattern.

        Examples:
        >>> p = rdflib.Variable('p')
        >>> q = rdflib.Variable('q')
        >>> x = rdflib.Variable('x')
        >>> dbr = rdflib.Namespace('http://dbpedia.org/resource/')
        >>> dbo = rdflib.Namespace('http://dbpedia.org/ontology/')
        >>> wikilink = dbo['wikiPageWikiLink']

        >>> gp = GraphPattern((
        ...     (SOURCE_VAR, p, q),
        ...     (q, wikilink, TARGET_VAR),
        ... ))
        >>> print(gp.to_sparql_select_query())
        SELECT ?p ?q ?source ?target WHERE {
         ?q <http://dbpedia.org/ontology/wikiPageWikiLink> ?target .
         ?source ?p ?q .
        }
        <BLANKLINE>

        >>> print(gp.to_sparql_select_query(
        ...     projection=(SOURCE_VAR, p),
        ...     distinct=True,
        ...     count=(COUNT_VAR, q),
        ...     bind={SOURCE_VAR: dbr['Test'], TARGET_VAR: q, x: dbr['X']}
        ... ))
        PREFIX dbr: <http://dbpedia.org/resource/>
        SELECT ?source ?p COUNT(DISTINCT ?q) as ?count WHERE {
         ?q <http://dbpedia.org/ontology/wikiPageWikiLink> ?target .
         ?source ?p ?q .
         FILTER(
          ?source=dbr:Test &&
          ?target=?q
         )
        }
        <BLANKLINE>

        >>> gtps = [
        ...     (dbr['Berlin'], dbr['Germany']),
        ...     (dbr['Amnesia'], dbr['Memory']),
        ...     (dbr['Paris'], dbr['France']),
        ...     (dbr['Rome'], dbr['Egypt']),
        ... ]
        >>> values = {(SOURCE_VAR, TARGET_VAR): gtps}
        >>> print(gp.to_sparql_select_query(values=values, limit=10))
        PREFIX dbr: <http://dbpedia.org/resource/>
        SELECT ?p ?q ?source ?target WHERE {
         VALUES (?source ?target) {
          (dbr:Berlin dbr:Germany)
          (dbr:Amnesia dbr:Memory)
          (dbr:Paris dbr:France)
          (dbr:Rome dbr:Egypt)
         }
         ?q <http://dbpedia.org/ontology/wikiPageWikiLink> ?target .
         ?source ?p ?q .
        }
        LIMIT 10
        <BLANKLINE>

        Args:
            projection: which variables to select on, by default all vars.
            distinct: if set to True and count isn't specified, the select
                query is preceded by the DISTINCT keyword. If count is specified
                DISTINCT is inserted in the count(DISTINCT ...) expression.
            count: a list. The first element is assumed to be the Variable to
                count into (as). The remainder of the tuple should be Variables
                or Strings which are inserted in the count(...) function.
            bind: a dict mapping variables to uris or other variables.
                Note that SPARQL BIND doesn't seem to be what we want here, so
                this is replaced with a FILTER expression. Also note that the
                constructed FILTER expression will only contain variables that
                are also in self.vars_in_graph(), as weird things happen
                otherwise.
            values: a dict mapping a variable tuple to a list of binding tuples,
                e.g. {(v1, v2): [(uri1, uri2), (uri3, uri4), ...]}
            limit: integer to limit the result size
        """
        assert self.vars_in_graph, \
            "tried to get sparql for pattern without vars: %s" % (self,)
        assert projection is None or isinstance(projection, Iterable)
        assert count is None or isinstance(count, Sequence)

        if projection is None:
            projection = sorted([v for v in self.vars_in_graph])
        assert projection or count

        res = "SELECT %(dist)s%(proj)s%(count)s WHERE {\n%(qpp)s}\n%(lim)s" % {
            'dist': 'DISTINCT ' if distinct and not count else '',
            'proj': ' '.join([v.n3() for v in projection]),
            'count': (' COUNT(%s%s) as %s' % (
                'DISTINCT ' if distinct else '',
                ' '.join([
                    c.n3() if isinstance(c, Variable) else str(c)
                    for c in count[1:]
                ]),
                count[0].n3()
            )) if count else '',
            'qpp': self._sparql_query_pattern_part(
                bind=bind,
                values=values,
                indent=' ',
            ),
            'lim': ('LIMIT %d\n' % limit) if limit is not None else '',
        }
        res = textwrap.dedent(res)
        return self._sparql_prefix(res)

    def to_sparql_ask_query(
            self,
            bind=None,
            values=None,
    ):
        return self._sparql_prefix(
            'ASK {\n%s}\n' % self._sparql_query_pattern_part(
                bind=bind,
                values=values,
            )
        )

    def _sparql_query_pattern_part(
            self,
            bind=None,
            values=None,
            indent=' ',
    ):
        assert bind is None or isinstance(bind, dict)
        assert values is None or (
            isinstance(values, dict) and
            isinstance(next(six.iterkeys(values)), Iterable) and
            isinstance(next(six.itervalues(values)), Iterable)
        )

        res = ''
        if values:
            res = indent + self._sparql_values_part(values, indent) + '\n'
        res += indent + self._sparql_triples_part(indent) + '\n'
        if bind:
            res += '%sFILTER(\n%s\n%s)\n' % (
                indent,
                ' &&\n'.join([
                    '%s %s=%s' % (indent, k.n3(), self.curify(v))
                    for k, v in sorted(bind.items())
                    if k in self.vars_in_graph
                ]),
                indent,
            )
        return res

    def _sparql_triples_part(self, indent=''):
        tres = []
        for s, p, o in self:
            tres.append('%s %s %s .' % (s.n3(), p.n3(), o.n3()))
        return ('\n' + indent).join(tres)

    def _sparql_values_part(self, values, indent=''):
        """Returns a SPARQL VALUES block as used in other methods.

        Values are curified by default, as it can be thousands and drastically
        reduces resulting query sizes.

        :param values: Dictionary of value to list of value instances. Both can
            and most likely will be tuples.
        :param indent: Indentation for all lines "in between".
        :return: Values block.
        """
        res = ''
        for vars_, value_tuple_list in values.items():
            vars_str = ' '.join([v.n3() for v in vars_])
            value_tuple_list_str = '\n'.join([
                '%s (%s)' % (indent, ' '.join([self.curify(v) for v in vt]))
                for vt in value_tuple_list
            ])
            res += 'VALUES (%s) {\n%s\n%s}' % (
                vars_str, value_tuple_list_str, indent)
        return res

    def to_combined_ask_count_query(self, values):
        """A combined query for a complete gp that does ask and counts in one.

        Example:
        >>> p = rdflib.Variable('p')
        >>> dbr = rdflib.Namespace('http://dbpedia.org/resource/')
        >>> dbo = rdflib.Namespace('http://dbpedia.org/ontology/')
        >>> wikilink = dbo['wikiPageWikiLink']
        >>> schema = rdflib.Namespace('http://schema.org/')
        >>> gtps = [
        ...     (dbr['Berlin'], dbr['Germany']),
        ...     (dbr['Amnesia'], dbr['Memory']),
        ...     (dbr['Paris'], dbr['France']),
        ...     (dbr['Rome'], dbr['Egypt']),
        ... ]
        >>> values = {(SOURCE_VAR, TARGET_VAR): gtps}

        >>> gp = GraphPattern([
        ...     (SOURCE_VAR, p, dbo['PopulatedPlace']),
        ...     (SOURCE_VAR, wikilink, TARGET_VAR),
        ...     (TARGET_VAR, p, schema['Country']),
        ... ])
        >>> print(gp.to_combined_ask_count_query(values))
        PREFIX dbr: <http://dbpedia.org/resource/>
        SELECT ?source ?target ?ask ?count WHERE {
         VALUES (?source ?target) {
          (dbr:Berlin dbr:Germany)
          (dbr:Amnesia dbr:Memory)
          (dbr:Paris dbr:France)
          (dbr:Rome dbr:Egypt)
         }
         BIND(EXISTS{
            ?source ?p <http://dbpedia.org/ontology/PopulatedPlace> .
            ?source <http://dbpedia.org/ontology/wikiPageWikiLink> ?target .
            ?target ?p <http://schema.org/Country> .
         } AS ?ask)
         OPTIONAL {
          {
           SELECT ?source COUNT(DISTINCT ?target) as ?count WHERE {
            ?source ?p <http://dbpedia.org/ontology/PopulatedPlace> .
            ?source <http://dbpedia.org/ontology/wikiPageWikiLink> ?target .
            ?target ?p <http://schema.org/Country> .
           }
          }
         }
        }
        <BLANKLINE>
        """
        vars_ = (SOURCE_VAR, TARGET_VAR, ASK_VAR, COUNT_VAR)
        res = """\
            SELECT %(proj)s WHERE {
             %(values)s
             BIND(EXISTS{
                %(triples)s
             } AS %(ask)s)
             OPTIONAL {
              {
               SELECT %(source)s COUNT(DISTINCT %(target)s) as %(count)s WHERE {
                %(triples)s
               }
              }
             }
            }
        """ % {
            'proj': ' '.join([v.n3() for v in vars_]),
            'values': self._sparql_values_part(values, indent='             '),
            'triples': self._sparql_triples_part(indent='                '),
            'ask': ASK_VAR.n3(),
            'source': SOURCE_VAR.n3(),
            'target': TARGET_VAR.n3(),
            'count': COUNT_VAR.n3(),
        }
        return self._sparql_prefix(textwrap.dedent(res))

    def to_count_var_over_values_query(self, var, vars_, values, limit):
        """Counts possible fulfilling substitutions for var.

        Example:
        >>> dbr = rdflib.Namespace('http://dbpedia.org/resource/')
        >>> dbo = rdflib.Namespace('http://dbpedia.org/ontology/')
        >>> wikilink = dbo['wikiPageWikiLink']
        >>> e = rdflib.Variable('edge')
        >>> v = rdflib.Variable('var')
        >>> gtps = [
        ...     (dbr['Adolescence'], dbr['Youth']),
        ...     (dbr['Adult'], dbr['Child']),
        ...     (dbr['Angel'], dbr['Heaven']),
        ...     (dbr['Arithmetic'], dbr['Mathematics']),
        ... ]
        >>> values = {(SOURCE_VAR, TARGET_VAR): gtps}
        >>> vars_ = (SOURCE_VAR, TARGET_VAR)

        >>> gp = GraphPattern((
        ...     (SOURCE_VAR, e, TARGET_VAR),
        ...     (v, wikilink, TARGET_VAR),
        ... ))
        >>> print(gp.to_count_var_over_values_query(v, vars_, values, 10))
        PREFIX dbr: <http://dbpedia.org/resource/>
        SELECT ?var COUNT(*) as ?count WHERE {
         VALUES (?source ?target) {
          (dbr:Adolescence dbr:Youth)
          (dbr:Adult dbr:Child)
          (dbr:Angel dbr:Heaven)
          (dbr:Arithmetic dbr:Mathematics)
         }
         {
          SELECT DISTINCT ?source ?target ?var WHERE {
           ?source ?edge ?target .
           ?var <http://dbpedia.org/ontology/wikiPageWikiLink> ?target .
          }
         }
        }
        ORDER BY DESC(?count)
        LIMIT 10
        <BLANKLINE>

        :param var: Variable to count over.
        :param vars_: List of vars to fix values for (e.g. ?source, ?target).
        :param values: List of value lists for vars_.
        :param limit: Limit for result size.
        :return: Query String.
        """
        res = """\
            SELECT %(var)s COUNT(*) as %(count)s WHERE {
             %(values)s
             {
              SELECT DISTINCT %(proj)s %(var)s WHERE {
               %(triples)s
              }
             }
            }
            ORDER BY DESC(%(count)s)
            LIMIT %(limit)d
        """ % {
            'var': var.n3(),
            'count': COUNT_VAR.n3(),
            'values': self._sparql_values_part(values, indent='             '),
            'proj': ' '.join([v.n3() for v in vars_]),
            'triples': self._sparql_triples_part('               '),
            'limit': limit,
        }
        return self._sparql_prefix(textwrap.dedent(res))

    def to_dict(self):
        return {
            'fitness': self.fitness.values if self.fitness.valid else (),
            'fitness_weighted': self.fitness.wvalues,
            'fitness_description': self.fitness.description_list,
            'sparql': self.to_sparql_select_query(),
            'graph_triples': [[self.curify(i) for i in t] for t in self],
            'matching_node_pairs': [
                [n.n3() for n in np_] for np_ in self.matching_node_pairs
            ],
            'gtp_precisions': [
                ((s.n3(), t.n3()), p)
                for (s, t), p in self.gtp_precisions.items()
            ],
            'prefixes': self.prefixes,
        }

    @staticmethod
    def from_dict(d):
        uri_shortener = URIShortener(prefixes=d.get('prefixes'))
        graph_triples = [
            tuple([uri_shortener.decurify(i) for i in t])
            for t in d['graph_triples']
        ]
        gp = GraphPattern(graph_triples)
        gp._uri_shortener = uri_shortener
        fitness = d.get('fitness')
        if fitness:
            gp.fitness.values = fitness
        matching_node_pairs = d.get('matching_node_pairs', [])
        gp.matching_node_pairs = [
            [gp.decurify(n) for n in np_]
            for np_ in matching_node_pairs
        ]
        gtp_precisions = d.get('gtp_precisions', [])
        gp.gtp_precisions = OrderedDict([
            ((gp.decurify(s), gp.decurify(t)),
             p)
            for (s, t), p in gtp_precisions
        ])
        return gp

    def get_gtps_precision_vector(self, gtps):
        return tuple([
            self.gtp_precisions[gtp] if gtp in self.gtp_precisions else 0
            for gtp in gtps
        ])

    def copy(self):
        return deepcopy(self)

    def is_connected(self, via_edges=False):
        if via_edges:
            g = to_nx_graph_via_edge_nodes(self)
        else:
            g = to_nx_graph(self)
        return nx.is_connected(g)

    def is_edge_connected_only(self):
        return self.is_connected(via_edges=True) and not self.is_connected()

    def node_edge_joint(self):
        return self.nodes & self.edges

    def eccentricity(self, v=None):
        g = to_nx_graph(self)
        return nx.eccentricity(g, v)

    def diameter(self):
        g = to_nx_graph(self)
        return nx.diameter(g)

    def __add__(self, other):
        assert isinstance(other, Iterable), \
            "self: %s, other not iterable: %r" % (self, other)
        if __debug__ and not isinstance(other, GraphPattern):
            try:
                it = iter(other)
                peek = next(it)
                assert isinstance(peek, tuple), \
                    "self: %sother first element not a tuple %r, other: %r" % (
                        self, peek, other
                    )
                other = chain((peek,), it)
            except StopIteration:
                pass
        return GraphPattern(chain(self, other))

    def __sub__(self, other):
        return GraphPattern(set(self) - set(other))

    def flip_edge(self, edge_idx):
        assert edge_idx < len(self), \
            "edge_idx %d out of bounds: %s" % (edge_idx, self)
        e = self[edge_idx]
        return GraphPattern(self[:edge_idx] + (e[::-1],) + self[edge_idx + 1:])

    def __repr__(self):
        return 'GraphPattern(' + super(GraphPattern, self).__repr__() + ')'

    def __str__(self):
        return 'GraphPattern:\n' + self._sparql_query_pattern_part()


class GraphPatternStats(object):
    """Stats object for graph patterns.

    A GraphPatternStats object is meant to be iteratively filled via the
    add_graph_pattern method. After adding, the counts can be pruned with
    prune_counts to reduce the memory consumption.

    A GraphPatternStats object provides the following stats attributes:

    Identifier based (treats graph patterns as (unordered) bag of identifiers):
    - identifier_gt_pair_count:
        For each identifier counts how many different ground truth pairs it
        occurred with. Each identifier is counted only once per gt pair. The
        counts will count each nodes' identifiers as often as the node appears
        in the ground truth. In case of dbr:Horse for example this will
        count all it's co-occurring predicates and neighbor nodes multiple
        times.
    - identifier_gt_node_count:
        For each identifier counts how many different nodes from the ground
        truth it occurred with. Each identifier is counted only once per gt
        node. This is different from the above as ground truth nodes can occur
        in multiple ground truth pairs. In case of dbr:Horse for example
        this doesn't multi-count all identifiers occurring with it.
    - identifier_gt_node_sum:
        For each identifier counts how many different nodes from the ground
        truth it occurred with. Each identifier is counted as often as it
        co-occurs with each gt node. For example dbpo:wikiPageWikiLink will
        typically be counted hundreds of times per ground truth node.
    """

    def __init__(self):
        # counts how many different ground truth pairs an identifier occurs with
        self.identifier_gt_pair_count = Counter()

        # counts how many different ground truth nodes an identifier occurs with
        self.identifier_gt_node_count = Counter()

        # sums up all occurrences of this identifier without double counting
        # same nodes in ground truth
        self.identifier_gt_node_sum = Counter()

        # nodes already included in stats
        self.nodes = set()

        # ground truth pairs already included in stats
        self.gt_pairs = set()

    def add_graph_pattern(self, gp, stimulus, response):
        assert isinstance(gp, GraphPattern), "%r not a GraphPattern" % gp
        gtp = (stimulus, response)
        assert gtp not in self.gt_pairs, \
            "gtp %r not in gt_pairs: %r" % (gtp, self.gt_pairs)
        self.gt_pairs.add(gtp)
        identifiers = gp.identifier_counts(exclude_vars=True)
        self.identifier_gt_pair_count.update(identifiers.keys())

        # if the stimulus or response was added to these stats before only add
        # the remaining triples to the counts. if neither was added before
        # identifiers remains as above.
        if stimulus in self.nodes:
            # only add triples with ?target
            identifiers = gp.only_with([TARGET_VAR]).identifier_counts(True)
        elif response in self.nodes:
            # only add triples with ?source
            identifiers = gp.only_with([SOURCE_VAR]).identifier_counts(True)
        self.nodes.add(stimulus)
        self.nodes.add(response)
        self.identifier_gt_node_sum.update(identifiers)
        self.identifier_gt_node_count.update(identifiers.keys())

    def min_identifier_gt_pair_occurrences(self, gp):
        identifiers = gp.identifier_counts(True)
        return min(
            [self.identifier_gt_pair_count.get(i, 0) for i in identifiers])

    def min_identifier_gt_node_occurrences(self, gp):
        identifiers = gp.identifier_counts(True)
        return min(
            [self.identifier_gt_node_count.get(i, 0) for i in identifiers])

    def rate_graph_pattern(self, gp):
        # TODO: rate wrt. what?
        predicates = [p for s, p, o in gp if not isinstance(p, Variable)]
        res = [
            self.identifier_gt_node_count.get(p, 0) /
            (self.identifier_gt_node_sum.get(p, 0) + 1)
            for p in predicates
        ]
        return res


    def prune_counts(self, below=2):
        lns = len(self.identifier_gt_node_sum)
        ln = len(self.identifier_gt_node_count)
        lp = len(self.identifier_gt_pair_count)
        self.identifier_gt_node_sum = Counter({
            k: v
            for k, v in six.iteritems(self.identifier_gt_node_sum)
            if v >= below
        })
        self.identifier_gt_node_count = Counter({
            k: v
            for k, v in six.iteritems(self.identifier_gt_node_count)
            if v >= below
        })
        self.identifier_gt_pair_count = Counter({
            k: v
            for k, v in six.iteritems(self.identifier_gt_pair_count)
            if v >= below
        })
        delta_node_sum = lns - len(self.identifier_gt_node_sum)
        delta_node_count = ln - len(self.identifier_gt_node_count)
        delta_pair_count = lp - len(self.identifier_gt_pair_count)
        return delta_node_sum, delta_node_count, delta_pair_count

    def __str__(self):
        return '%s: pairs: %d, nodes: %d, Identifier counts:\n' \
            'Pairs: %s\nNodes: %s' % (
                self.__class__.__name__, len(self.gt_pairs), len(self.nodes),
                self.identifier_gt_pair_count, self.identifier_gt_node_count
            )
