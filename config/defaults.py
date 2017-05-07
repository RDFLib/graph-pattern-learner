# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

# WARNING: file is parsed from helpers.py and imported in __init__
# All uppercase variable assignments are config options that become configurable
# from commandline. Comments (in same line) will become docs.

DATADIR = 'data'  # dir with default input files
GT_ASSOCIATIONS_FILENAME = path.join(DATADIR, 'gt_associations.csv')
SPLITTING_VARIANTS = ['random', 'target_node_disjoint', 'node_disjoint']
RESDIR = 'results'  # dir to put learner results
SYMLINK_CURRENT_RES_RUN_GEN = 'top_graph_patterns_current.json.gz'  # link name
RES_RUN_PREFIX = 'results_run'
SYMLINK_CURRENT_RES_RUN = 'results_current.json.gz'  # link name
PAUSE_FILE = 'pause.lck'

ERROR_RETRIES = 5  # in case an unexpected error occurs retry? (real world!)

NRUNS = 64  # number of whole coverage runs of the evolutionary algorithm
NRUNS_NO_IMPROVEMENT = 5  # stop if no more coverage patterns found in n runs
MIN_SCORE = 2  # don't consider patterns for coverage with a score below this
MIN_REMAINING_GAIN = 1  # stop if remaining gain is below this
QUICK_STOP = True  # stop after 1st generation that reaches MIN_REMAINING_GAIN

# graph pattern:
MAX_PATTERN_LENGTH = 15  # max number of triples in a pattern
MAX_PATTERN_VARS = 10  # max number of vars in a pattern
MAX_PATTERN_QUERY_SIZE = 2000  # maximum select query length (chars)
MAX_LITERAL_SIZE = 128  # maximum length of Literals in a pattern
F_MEASURE_BETA = 1  # 0.5 for higher focus on precision than recall
PATTERN_P_CONNECTED = False  # include patterns connected via predicate not node
OVERFITTING_PUNISHMENT = 0.25  # multiplier for single ?source or ?target match

# SPARQL query:
SPARQL_ENDPOINT = 'http://localhost:8890/sparql'
BATCH_SIZE = 384  # tested to rarely result in error recursions
QUERY_TIMEOUT_FACTOR = 32  # timeout factor compared to a simplistic query
QUERY_TIMEOUT_MIN = 2  # minimum query timeout in seconds
CACHE_SIZE = 1000  # cache for queries and fit-to-live computations
PREDICTION_RESULT_LIMIT = 1000  # max results per individual GP queries
PREDICTION_IN_PARALLEL = True  # run the queries for a prediction in parallel?

# evolutionary algorithm:
NGEN = 20  # number of generations
NGEN_NO_IMPROVEMENT = 5  # terminate if no better individual found in x gens
POPSIZE = 200  # (target) number of individuals
HOFSIZE = 100  # size of hall of fame
CACHE_SIZE_FIT_TO_LIVE = 128  # cache for fit to live checks
TOURNAMENT_SIZE = 3  # selection tournament size, causing selection pressure
INIT_POP_LEN_ALPHA = 5  # alpha value in a length beta distribution
INIT_POP_LEN_BETA = 30  # beta value in a length beta distribution
INIT_POPPB_FV = 0.9  # probability to fix a variable in init population
INIT_POPPB_FV_N = 5  # allow up to n instantiations for each fixed variable
VARPAT_REINTRO = 10  # number of variable patterns re-introduced each generation
HOFPAT_REINTRO = 10  # number of hall of fame patterns re-introduced each gen
LOGLVL_EVAL = 10  # loglvl for eval logs (10: DEBUG, 20: INFO)
LOGLVL_MUTFV = 10  # loglvl for fix var mutation (10: DEBUG, 20: INFO)
LOGLVL_MUTSP = 10  # loglvl for simplify mutation (10: DEBUG, 20: INFO)
CXPB = 0.5  # cross-over aka mating probability
CXPB_RV = 0.5  # probability that vars in mating delta will be renamed
CXPB_BP = 1.0  # prob to draw each of the triples that are in both parents
CXPB_DP = 0.8  # prob to draw each of the dominant parent's only triples
CXPB_OP = 0.2  # prob to draw each of the other parent's only triples
CX_RETRY = 3  # how often we try to generate a living child
MUTPB = 0.5  # mutation probability
MUTPB_IV = 0.1  # prob to introduce a variable instead of a node or edge
MUTPB_SV = 0.1  # prob to split a variable into two
# TODO: the more variables a pattern gets, the higher the prob to merge?
MUTPB_MV = 0.3  # prob to merge 2 variables
MUTPB_MV_MIX = 0.2  # prob to mix node and edge vars (see PATTERN_P_CONNECTED)
# TODO: the larger a pattern gets the higher the prob to delete triple?
MUTPB_DT = 0.4  # prob to delete a random triple from a pattern
MUTPB_EN = 0.4  # prob to expand a node and add a triple from expansion
MUTPB_EN_OUT_LINK = 0.5  # probability to add an outgoing triple (otherwise in)
MUTPB_AE = 0.2  # prob to try adding an edge between two nodes
MUTPB_ID = 0.05  # prob to increase distance between source and target by 1 hop
MUTPB_FV = 0.4  # prob to fix a variable (SPARQL)
MUTPB_FV_RGTP_SAMPLE_N = 128  # sample <= n remaining GTPs to fix variables for
MUTPB_FV_SAMPLE_MAXN = 32  # max n of instantiations to sample from top k
MUTPB_FV_QUERY_LIMIT = 256  # SPARQL query limit for the top k instantiations
MUTPB_SP = 0.05  # prob to simplify pattern (warning: can restrict exploration)

# for import in helpers and __init__
__all__ = [_v for _v in globals().keys() if _v.isupper()]
