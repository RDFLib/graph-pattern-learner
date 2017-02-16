#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
import config

logger = logging.getLogger(__name__)


RANDOM_VAR_LEN = 5  # so in total we have 62**5=916132832 different random vars
RANDOM_VAR_PREFIX = 'vr'
SOURCE_VAR = Variable('source')
TARGET_VAR = Variable('target')
ASK_VAR = Variable('ask')
COUNT_VAR = Variable('count')
EDGE_VAR_COUNT = Variable('edge_var_count')
NODE_VAR_SUM = Variable('node_var_sum')


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
        s, p, o = [
            BNode(i) if isinstance(i, Variable) and i not in fixed_vars else i
            for i in t
        ]
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
        t = tuple([Variable(i) if isinstance(i, BNode) else i for i in t])
        cgp.append(t)
    return sorted(cgp)


def canonicalize_sparql_bgp(gp, fixed_vars=None):
    """Returns a canonical basic graph pattern (BGP) with canonical var names.

    :param gp: a GraphPattern in form of a list of triples with Variables
    :param fixed_vars:
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
    if len(gp) != len(cgp):
        # bug in lib: rdflib.compare.to_canonical_graph(g) sometimes collapses
        # distinct bnodes
        # see https://github.com/RDFLib/rdflib/issues/494
        # if this happens, one triple from the original pattern is missing,
        # which means that the lengths are different
        logger.warning(
            'GraphPattern canonicalization failed, returning original:\n%r\n'
            'SPARQL BGP RDF Reification-Graph:\n%r\n'
            'Canonicalized RDF Graph:\n%r\n'
            'Canonicalized Graph Pattern:\n%r\n',
            gp, list(g), list(cg), cgp
        )
        return gp
    return cgp


def canonicalize(gp, shorten_varnames=True):
    """Returns a canonical basic graph pattern (BGP) with canonical var names.

    :param gp: a GraphPattern in form of a list of triples with Variables
    :param shorten_varnames: If True (default) long sha256 based var-names will
        be renamed to short enumerated ones.
    :return: A canonical GraphPattern with Variables renamed.

    >>> U = URIRef
    >>> V = Variable
    >>> gp1 = [
    ...     (V('blub'), V('bar'), U('blae')),
    ...     (V('foo'), V('bar'), U('bla')),
    ...     (SOURCE_VAR, V('poo'), TARGET_VAR),
    ... ]
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
    >>> gp2 = [
    ...     (SOURCE_VAR, V('bla'), TARGET_VAR),
    ...     (V('blub'), V('bli'), U('bla')),
    ...     (V('bluub'), V('bli'), U('blae')),
    ... ]
    >>> cgp == canonicalize(gp2)
    True

    """
    cgp = canonicalize_sparql_bgp(gp, fixed_vars={SOURCE_VAR, TARGET_VAR})
    mapping = {}
    if shorten_varnames:
        vars_ = set(chain.from_iterable(cgp))
        vars_ = sorted([
            v for v in vars_ if isinstance(v, Variable) and v.startswith('cb')
        ])
        for i, v in enumerate(vars_):
            mapping[v] = Variable('vcb%d' % i)
    return GraphPattern(cgp, mapping=mapping)


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
        """Generates a SPARQL query from the graph pattern.

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
        res = 'SELECT '
        if distinct and not count:
            res += 'DISTINCT '
        res += '%s ' % ' '.join([v.n3() for v in projection])
        if count:
            res += 'COUNT('
            if distinct:
                res += 'DISTINCT '
            res += '%s' % ' '.join([
                c.n3() if isinstance(c, Variable) else str(c)
                for c in count[1:]
            ])
            res += ') as %s ' % count[0].n3()
        res += 'WHERE {\n'
        res += self._sparql_query_pattern_part(
            bind=bind,
            values=values,
            indent=' ',
        )
        res += '}\n'
        if limit is not None:
            res += 'LIMIT %d' % limit
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

        if values is None:
            values = {}
        res = self._sparql_values_part(values, indent)
        tres = []
        for s, p, o in self:
            tres.append('%s %s %s .' % (s.n3(), p.n3(), o.n3()))
        res += indent + ('\n' + indent).join(tres) + '\n'
        if bind:
            res += '%sFILTER(\n' % indent
            filters = ' &&\n'.join([
                '%s %s=%s' % (indent, k.n3(), self.curify(v))
                for k, v in bind.items()
                if k in self.vars_in_graph
            ])
            res += filters + '\n%s)\n' % indent
        return res

    def _sparql_values_part(self, values, indent=' '):
        res = ''
        for vars_, value_tuple_list in values.items():
            vars_str = ' '.join([v.n3() for v in vars_])
            value_tuple_list_str = '\n'.join([
                '%s (%s)' % (indent, ' '.join([self.curify(v) for v in vt]))
                for vt in value_tuple_list
            ])
            res += '%sVALUES (%s) {\n' % (indent, vars_str)
            res += value_tuple_list_str + '\n'
            res += '%s}\n' % indent
        return res

    def to_combined_ask_count_query(self, values):
        """A combined query for a complete gp that does ask and counts in one.

        Meant to perform a query like this:
         SELECT ?source ?target ?ask ?count WHERE {
          VALUES (?source ?target) {
           (dbr:Berlin dbr:Germany)
           (dbr:Amnesia dbr:Memory)
           (dbr:Paris dbr:France)
           (dbr:Rome dbr:Egypt)
           ... long list ...
          }
          BIND(EXISTS{
             ?source dbo:wikiPageWikiLink ?target .
             ?source a dbo:PopulatedPlace .
             ?target a schema:Country .
          } AS ?ask)
          OPTIONAL {
           {
            SELECT ?source COUNT(DISTINCT ?target) as ?count WHERE {
             ?source dbo:wikiPageWikiLink ?target .
             ?source a dbo:PopulatedPlace .
             ?target a schema:Country .
            }
           }
          }
         }
        """
        vars_ = (SOURCE_VAR, TARGET_VAR, ASK_VAR, COUNT_VAR)
        res = 'SELECT ' + ' '.join([v.n3() for v in vars_]) + ' WHERE {\n'
        res += self._sparql_values_part(values)

        # BIND part (ASK)
        res += ' BIND(EXISTS{\n'
        tres = []
        for s, p, o in self:
            tres.append('%s %s %s .' % (s.n3(), p.n3(), o.n3()))
        indent = ' ' * 4
        triples = indent + ('\n' + indent).join(tres) + '\n'
        res += triples
        res += ' } AS %s)\n' % ASK_VAR.n3()

        # Subquery part (COUNT)
        res += ' OPTIONAL {\n' \
               '  {\n' \
               '   SELECT %s COUNT(DISTINCT %s) as %s WHERE {\n' % (
                   SOURCE_VAR.n3(),
                   TARGET_VAR.n3(),
                   COUNT_VAR.n3(),
               )
        res += triples
        res += '   }\n' \
               '  }\n' \
               ' }\n' \
               '}\n'
        return self._sparql_prefix(res)

    def to_count_var_over_values_query(self, var, vars_, values, limit):
        """Counts possible fulfilling substitutions for var.

        Meant to perform a query like this:
         SELECT ?var count(*) as ?count WHERE {
          VALUES (?source ?target) {
           (dbr:Adolescence dbr:Youth)
           (dbr:Adult dbr:Child)
           (dbr:Angel dbr:Heaven)
           (dbr:Arithmetic dbr:Mathematics)
          }
          {
           SELECT DISTINCT ?source ?target ?var WHERE {
            ?source ?edge ?target .
            ?var dbo:wikiPageWikiLink ?target .
           }
          }
         }
         ORDER BY desc(?count)
         LIMIT 10

        :param var: Variable to count over.
        :param vars_: List of vars to fix values for (e.g. ?source, ?target).
        :param values: List of value lists for vars_.
        :param limit: Limit for result size.
        :return: Query String.
        """
        res = 'SELECT %s COUNT(*) as %s WHERE {\n' % (var.n3(), COUNT_VAR.n3())
        res += self._sparql_values_part(values)

        res += ' {\n' \
               '  SELECT DISTINCT %s %s WHERE {\n' % (
                   ' '.join([v.n3() for v in vars_]),
                   var.n3(),
               )

        # triples part
        tres = []
        for s, p, o in self:
            tres.append('%s %s %s .' % (s.n3(), p.n3(), o.n3()))
        indent = ' ' * 3
        triples = indent + ('\n' + indent).join(tres) + '\n'
        res += triples

        res += '  }\n' \
               ' }\n' \
               '}\n'
        res += 'ORDER BY DESC(%s)\n' % COUNT_VAR.n3()
        res += 'LIMIT %d\n' % limit
        return self._sparql_prefix(res)

    def to_find_edge_var_for_narrow_path_query(
            self, edge_var, node_var, vars_, values, limit_res,
            filter_node_count=config.MUTPB_DN_FILTER_NODE_COUNT,
            filter_edge_count=config.MUTPB_DN_FILTER_EDGE_COUNT,
    ):
        """Counts possible substitutions for edge_var to get a narrow path

        Meant to perform a query like this:
        SELECT *
        {
          {
            SELECT
              ?edge_var
              (COUNT(*) AS ?edge_var_count)
              (MAX(?node_var_count) AS ?max_node_count)
              (COUNT(*)/AVG(?node_var_count) as ?prio_var)
            {
              SELECT DISTINCT
                ?source ?target ?edge_var (COUNT(?node_var) AS ?node_var_count)
              {
                VALUES (?source ?target) {
                  (dbr:Adolescence dbr:Youth)
                  (dbr:Adult dbr:Child)
                  (dbr:Angel dbr:Heaven)
                  (dbr:Arithmetic dbr:Mathematics)
                }
                ?node_var ?edge_var ?source .
                ?source dbo:wikiPageWikiLink ?target .
              }
            }
            GROUP BY ?edge_var
            ORDER BY DESC(?edge_var_count)
          }
          FILTER(?max_node_count < 10 && ?edge_var_count > 1)
        }
        ORDER BY DESC(?prio_var)
        LIMIT 32

        :param edge_var: Edge variable to find substitution for.
        :param node_var: Node variable to count.
        :param vars_: List of vars to fix values for (e.g. ?source, ?target).
        :param values: List of value lists for vars_.
        :param filter_node_count: Filter on node count of edge variable.
        :param filter_edge_count: Filter for edge count of triples.
        :param limit_res : limit result size
        :return: Query String.
        """

        res = 'SELECT * WHERE {\n'
        res += ' {\n'\
               '  SELECT %s (SUM (?node_var_count) AS %s) (COUNT(%s) AS %s) ' \
               '(MAX(?node_var_count) AS ?max_node_count) WHERE {\n' % (
                     edge_var.n3(),
                     NODE_VAR_SUM.n3(),
                     ' && '.join([v.n3() for v in vars_]),
                     EDGE_VAR_COUNT.n3(), )
        res += '    SELECT DISTINCT %s %s (COUNT(%s) AS ?node_var_count) ' \
               'WHERE {\n   ' % (' '.join([v.n3() for v in vars_]),
                                 edge_var.n3(), node_var.n3(), )
        res += self._sparql_values_part(values)

        # triples part
        tres = []
        for s, p, o in self:
            tres.append('%s %s %s .' % (s.n3(), p.n3(), o.n3()))
        indent = ' ' * 3
        triples = indent + ('\n' + indent).join(tres) + '\n'
        res += triples
        res += '    }\n'\
               '   }\n'
        res += '   GROUP BY %s\n' % edge_var.n3()
        res += '  }\n'
        res += '  FILTER(?max_node_count < %d && %s > %d)\n' \
               % (filter_node_count, EDGE_VAR_COUNT.n3(),
                  filter_edge_count)
        res += '}\n'
        res += 'ORDER BY ASC(%s)\n' % NODE_VAR_SUM.n3()
        res += 'LIMIT %d' % limit_res
        return self._sparql_prefix(res)

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

    def mixed_node_edge_vars(self):
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
