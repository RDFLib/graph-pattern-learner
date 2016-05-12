# coding=utf-8
from graph_pattern import canonicalize
from graph_pattern import GPFitness
from graph_pattern import GPFitnessTuple
from graph_pattern import GraphPattern
from graph_pattern import GraphPatternStats
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from rdflib import Graph
from rdflib import Literal
from rdflib import URIRef
from rdflib import Variable


def test_graph_pattern_fitness():
    e = GPFitness()
    assert not e.valid
    assert e.format_fitness() == '(not evaluated yet)'
    assert e.wvalues == (), e.wvalues
    assert e.values == ()

    values = (655, 0.4048, 0.4048, 0.0089, 7.5, 3, 3, 2, 0, 0.1936)
    e.values = values
    assert e.valid
    assert e.format_fitness() == str(values)
    assert e.values == values
    assert e.wvalues == tuple([
        x * y for x, y in zip(GPFitness.weights, values)])

    del e.values
    assert not e.valid
    assert e.format_fitness() == '(not evaluated yet)'
    assert e.wvalues == (), e.wvalues

    f = GPFitness(values)
    assert f.valid
    assert f.format_fitness() == str(values)
    assert f.values == values
    assert f.wvalues == tuple([
        x * y for x, y in zip(GPFitness.weights, values)])

    assert f.values.score == 0.4048

    t = GPFitnessTuple(remains=655)
    values = (655, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    f = GPFitness(t)
    assert f.valid
    assert f.format_fitness() == str(values)
    assert f.values == values
    assert f.values.remains == 655
    assert f.wvalues == tuple([
        x * y for x, y in zip(GPFitness.weights, values)])

    t = GPFitnessTuple(score=100)
    values = (0, 100, 0, 0, 0, 0, 0, 0, 0, 0)
    f = GPFitness(t)
    assert f.valid
    assert f.format_fitness() == str(values)
    assert f.values == values
    assert f.values.score == 100
    assert f.wvalues == tuple([
        x * y for x, y in zip(GPFitness.weights, values)])

    del f.values
    assert not f.valid
    assert f.format_fitness() == '(not evaluated yet)'
    assert f.wvalues == (), f.wvalues


def test_graph_pattern():
    g = Graph()
    g.add((URIRef('foo'), URIRef('bar'), Literal('bla')))
    g.add((URIRef('foo'), URIRef('baa'), Literal('bla')))
    g.add((URIRef('faa'), URIRef('boo'), Literal('blub')))

    gp = GraphPattern(g)
    gp = gp.replace({
        URIRef('foo'): Variable('a'),
        Literal('bla'): Variable('l'),
    })
    sparql = gp.to_sparql_select_query()
    expected = 'SELECT ?a ?l WHERE {\n' \
        ' ?a <baa> ?l .\n' \
        ' ?a <bar> ?l .\n' \
        ' <faa> <boo> "blub" .\n' \
        '}\n'
    assert sparql == expected, "expected: %s\ngot: %s" % (expected, sparql)
    sparql = gp.to_sparql_select_query()
    assert sparql == expected, "expected: %s\ngot: %s" % (expected, sparql)

    gp2 = gp.replace({URIRef('baa'): Variable('b')})
    sparql = gp2.to_sparql_select_query(
        bind={Variable('a'): URIRef('bound')}
    )
    expected = 'SELECT ?a ?b ?l WHERE {\n' \
        ' ?a ?b ?l .\n' \
        ' ?a <bar> ?l .\n' \
        ' <faa> <boo> "blub" .\n' \
        ' FILTER(\n' \
        '  ?a=<bound>\n' \
        ' )\n' \
        '}\n'
    assert sparql == expected, "expected: %s\ngot: %s" % (expected, sparql)

    gp3 = GraphPattern(g, source_node=URIRef('foo'), target_node=Literal('bla'))
    expected = 'SELECT ?source ?target WHERE {\n' \
        ' ?source <baa> ?target .\n' \
        ' ?source <bar> ?target .\n' \
        ' <faa> <boo> "blub" .\n' \
        '}\n'
    sparql = gp3.to_sparql_select_query()
    assert sparql == expected, "expected: %s\ngot: %s" % (expected, sparql)

    gp4 = gp3.only_with([TARGET_VAR])
    expected = 'SELECT ?source ?target WHERE {\n' \
        ' ?source <baa> ?target .\n' \
        ' ?source <bar> ?target .\n' \
        '}\n'
    sparql = gp4.to_sparql_select_query()
    assert sparql == expected, "expected: %s\ngot: %s" % (expected, sparql)
    gp4_red = gp4.replace({URIRef('baa'): URIRef('bar')})
    assert len(gp4) > len(gp4_red), \
        "double edge should've been reduced: %s" % (gp4_red,)

    gp5 = gp3.only_with([URIRef('bar')])
    expected = 'SELECT ?source ?target WHERE {\n' \
        ' ?source <bar> ?target .\n' \
        '}\n'
    sparql = gp5.to_sparql_select_query()
    assert sparql == expected, "expected: %s\ngot: %s" % (expected, sparql)

    gp6 = gp + gp2
    expected = 'SELECT ?a ?b ?l WHERE {\n' \
        ' ?a ?b ?l .\n' \
        ' ?a <baa> ?l .\n' \
        ' ?a <bar> ?l .\n' \
        ' <faa> <boo> "blub" .\n' \
        '}\n'
    sparql = gp6.to_sparql_select_query()
    assert sparql == expected, "expected: %s\ngot: %s" % (expected, sparql)

    gp7 = gp - gp2
    expected = 'SELECT ?a ?l WHERE {\n' \
        ' ?a <baa> ?l .\n' \
        '}\n'
    sparql = gp7.to_sparql_select_query()
    assert sparql == expected, "expected: %s\ngot: %s" % (expected, sparql)

    gp8 = gp + ((TARGET_VAR, TARGET_VAR, TARGET_VAR),)
    expected = 'SELECT ?a ?l ?target WHERE {\n' \
        ' ?a <baa> ?l .\n' \
        ' ?a <bar> ?l .\n' \
        ' ?target ?target ?target .\n' \
        ' <faa> <boo> "blub" .\n' \
        '}\n'
    sparql = gp8.to_sparql_select_query()
    assert sparql == expected, "expected: %s\ngot: %s" % (expected, sparql)

    gp9 = gp - gp
    assert not bool(gp9), 'gp9 was not empty'
    gp9 = gp - list(gp)
    assert not bool(gp9), 'gp9 - list(gp9) was not empty'

    # test triples by identifier:
    tbi = gp8.triples_by_identifier()
    expected = {
        Variable('a'): {
            (Variable('a'), URIRef('baa'), Variable('l')),
            (Variable('a'), URIRef('bar'), Variable('l')),
        },
        Variable('l'): {
            (Variable('a'), URIRef('baa'), Variable('l')),
            (Variable('a'), URIRef('bar'), Variable('l')),
        },
        URIRef('baa'): {
            (Variable('a'), URIRef('baa'), Variable('l')),
        },
        URIRef('bar'): {
            (Variable('a'), URIRef('bar'), Variable('l')),
        },
        Variable('target'): {
            (Variable('target'), Variable('target'), Variable('target')),
        },
        URIRef('faa'): {
            (URIRef('faa'), URIRef('boo'), Literal('blub')),
        },
        URIRef('boo'): {
            (URIRef('faa'), URIRef('boo'), Literal('blub')),
        },
        Literal('blub'): {
            (URIRef('faa'), URIRef('boo'), Literal('blub')),
        },
    }
    assert tbi == expected, 'triples_by_identifier %s != %s' % (tbi, expected)
    tbn = gp8.triples_by_nodes({
        Variable('a'), Variable('target'), URIRef('notthere'), URIRef('faa'),
        URIRef('boo')
    })
    expected = {
        Variable('a'): {
            (Variable('a'), URIRef('baa'), Variable('l')),
            (Variable('a'), URIRef('bar'), Variable('l')),
        },
        Variable('target'): {
            (Variable('target'), Variable('target'), Variable('target')),
        },
        URIRef('faa'): {
            (URIRef('faa'), URIRef('boo'), Literal('blub')),
        },
        URIRef('notthere'): set(), URIRef('boo'): set(),
    }
    assert tbn == expected, 'triples_by_nodes %s != %s' % (tbn, expected)
    tbe = gp8.triples_by_edges({
        URIRef('baa'), Variable('a'), Variable('?target')
    })
    expected = {
        URIRef('baa'): {
            (Variable('a'), URIRef('baa'), Variable('l')),
        },
        Variable('target'): {
            (Variable('target'), Variable('target'), Variable('target')),
        },
        Variable('a'): set(),
    }
    assert tbe == expected, 'triples_by_edges %s != %s' % (tbe, expected)


def test_graph_pattern_connectedness():
    # test edge var connections
    gp = GraphPattern([
        (SOURCE_VAR, Variable('p'), Variable('v1')),
        (TARGET_VAR, Variable('p'), Variable('v2')),
    ])
    assert not gp.is_connected(), \
        "shouldn't be connected with nodes only: %s" % (gp,)
    assert gp.is_connected(via_edges=True), \
        "should be connected via edges: %s" % (gp,)
    gp = GraphPattern([
        (SOURCE_VAR, Variable('p'), Variable('v1')),
        (Variable('p'), Variable('v2'), TARGET_VAR),
    ])
    assert not gp.is_connected(), \
        "shouldn't be connected with nodes only: %s" % (gp,)
    assert gp.is_connected(via_edges=True), \
        "should be connected via edges: %s" % (gp,)


def test_graph_pattern_canonicalization():
    # test for bug in lib:
    # rdflib.compare.to_canonical_graph(g) sometimes collapses distinct bnodes
    # see https://github.com/RDFLib/rdflib/issues/494
    # The GraphPattern below causes such a problem, currently we return gp
    # itself instead of a canonical representation of it. We just test the len
    # in case it's fixed in rdflib.
    gp = GraphPattern((
        (SOURCE_VAR, Variable(u'vcb0'), TARGET_VAR),
        (SOURCE_VAR, Variable(u'vrBYUk8'), TARGET_VAR),
        (TARGET_VAR, Variable(u'vrBYUk8'), SOURCE_VAR),
        (TARGET_VAR, Variable(u'vrvGapn'), SOURCE_VAR)))
    cgp = canonicalize(gp)
    assert len(gp) == len(cgp)


def test_graph_pattern_stats():
    gp = GraphPattern(
        (
            (URIRef('bar'), URIRef('pred1'), URIRef('s')),
            (URIRef('foo'), URIRef('pred2'), URIRef('t')),
            (URIRef('s'), URIRef('pred3'), URIRef('t')),
        ),
        source_node=URIRef('s'),
        target_node=URIRef('t'),
    )
    gp1 = GraphPattern(
        (
            (URIRef('bar'), URIRef('pred1'), URIRef('s2')),
            (URIRef('foo'), URIRef('pred2'), URIRef('t')),
            (URIRef('s2'), URIRef('pred3'), URIRef('t')),
            (URIRef('single'), URIRef('pred1'), URIRef('s2')),
        ),
        source_node=URIRef('s2'),
        target_node=URIRef('t'),
    )
    gps = GraphPatternStats()
    gps.add_graph_pattern(gp, URIRef('s'), URIRef('t'))
    identifiers = set(gp.identifier_counts(True))
    assert identifiers == {
        URIRef('bar'),
        URIRef('foo'),
        URIRef('pred1'),
        URIRef('pred2'),
        URIRef('pred3'),
    }, identifiers
    assert identifiers == set(gps.identifier_gt_node_count.keys())
    assert identifiers == set(gps.identifier_gt_pair_count.keys())
    assert set([(i, 1) for i in identifiers]) == set(
        gps.identifier_gt_node_count.items())
    assert set([(i, 1) for i in identifiers]) == set(
        gps.identifier_gt_pair_count.items())

    gps.add_graph_pattern(gp1, URIRef('s2'), URIRef('t'))
    assert set(gps.identifier_gt_node_count.keys()) == \
        identifiers | {URIRef('single')}
    assert set(gps.identifier_gt_pair_count.keys()) == \
        identifiers | {URIRef('single')}
    expected_node = {
        (URIRef('bar'), 2),
        (URIRef('foo'), 1),
        (URIRef('pred1'), 2),
        (URIRef('pred2'), 1),
        (URIRef('pred3'), 2),
        (URIRef('single'), 1),
    }
    res = set(gps.identifier_gt_node_count.items())
    assert expected_node == res, res
    expected_pair = {
        (URIRef('bar'), 2),
        (URIRef('foo'), 2),
        (URIRef('pred1'), 2),
        (URIRef('pred2'), 2),
        (URIRef('pred3'), 2),
        (URIRef('single'), 1),
    }
    res = set(gps.identifier_gt_pair_count.items())
    assert expected_pair == res, res

    tmp = gps.min_identifier_gt_node_occurrences(gp)
    assert tmp == 1, 'tmp: %d\n%s' % (tmp, gps)
    tmp = gps.min_identifier_gt_pair_occurrences(gp)
    assert tmp == 2, 'tmp: %d\n%s' % (tmp, gps)
