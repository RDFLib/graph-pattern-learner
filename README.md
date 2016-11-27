Graph Pattern Learner
=====================

(Work in progress...)

In this repository you find the code for a graph pattern learner. Given a list
of source-target-pairs and a SPARQL endpoint, it will try to learn SPARQL
patterns. Given a source, the learned patterns will try to lead you to the right
target.

The algorithm was first developed on a list of human associations that had been
mapped to DBpedia entities, as can be seen in
[data/gt_associations.csv](./data/gt_associations.csv):

| source                            | target                            |
| --------------------------------- | --------------------------------- |
| http://dbpedia.org/resource/Bacon | http://dbpedia.org/resource/Egg   |
| http://dbpedia.org/resource/Baker | http://dbpedia.org/resource/Bread |
| http://dbpedia.org/resource/Crow  | http://dbpedia.org/resource/Bird  |
| http://dbpedia.org/resource/Elm   | http://dbpedia.org/resource/Tree  |
| http://dbpedia.org/resource/Gull  | http://dbpedia.org/resource/Bird  |
| ...                               | ...                               |

As you can immediately see, associations don't only follow a single pattern. Our
algorithm is designed to be able to deal with this. It will try to learn several
patterns, which in combination model your input list of source-target-pairs. If
your list of source-target-pairs is less complicated, the algorithm will happily
terminate earlier.

You can find more information about the algorithm and learning patterns for
human associations on https://w3id.org/associations . The page also includes
publications, as well as the resulting patterns learned for human associations
from a local DBpedia endpoint including wikilinks.


Installation
------------

For now, the suggested installation method is via git clone (also allows easier
contributions):

    git clone https://github.com/RDFLib/graph-pattern-learner.git
    cd graph-pattern-learner

Afterwards, to setup the virtual environment and install all dependencies in it:

    virtualenv venv &&
    . venv/bin/activate &&
    pip install -r requirements.txt &&
    deactivate


Running the learner
-------------------

Before actually running the evolutionary algorithm, please consider that it will
issue a lot of queries to the endpoint you're specifying. Please don't run this
against public endpoints without asking the providers first. It is likely that
you will disrupt their service or get blacklisted. I suggest running against an
own local endpoint filled with the datasets you're interested in. If you really
want to run this against public endpoints, at least don't run the multi-process
version, but restrict yourself to one process.

Always feel free to reach out for help or feedback via the issue tracker or via
associations at joernhees de. We might even run the learner for you ;)

Before running, make sure to activate the virtual environment:

    . venv/bin/activate

To get a list of all available options run:

    python run.py --help

Don't be scared by the length, most options use sane defaults, but it's nice to
be able to change things once you become more familiar with your data and the
learner.

The options you will definitely be interested are:

    --associations_filename (defaults to ./data/gt_associations.csv)
    --sparql_endpoint (defaults to http://dbpedia.org/sparql)

To run the algorithm you might want to run it like this:

    ./clean_logs.sh
    PYTHONIOENCODING=utf-8 python \
        run.py --associations_filename=... --sparql_endpoint=... \
        2>&1 | tee >(gzip > logs/main.log.gz)

If you want to speed things up you can (and should) run with SCOOP in parallel:

    ./clean_logs.sh
    PYTHONIOENCODING=utf-8 python \
        -m scoop -n8 run.py --associations_filename=... --sparql_endpoint=... \
        2>&1 | tee >(gzip > logs/main.log.gz)

SCOOP will then run the graph pattern learner distributed over 8 cores (-n).

The algorithm will by default randomly split your input list of
source-target-pairs into a training and a test set. If you want to see how well
the learned patterns generalise, you can run:

    ./run_create_bundle.sh ./results/bundle_name sparql_endpoint \
        --associations_filename=...

The script will then first learn patterns, visualise them in
`./results/bundle_name/visualise`, before evaluating predictions on first the
training- and then the test-set.


Contributors
------------
 * Jörn Hees
 * Rouven Bauer (visualise code)
