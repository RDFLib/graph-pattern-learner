# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Config for graph_learner.

The actual config values can be found in defaults.py, the rest are helpers which
make sure you can update the config via command line args and that these updates
actually arrive on workers in a distributed (parallel) workflow with SCOOP.

To use this (have a look at run.py) the main process should:
- import this as early as possible
- generate an `argparser` with own args
- complete that argparser by handing it to `arg_parse_config_vars()`
- parse the args
- call `finalize` with the resulting args dict
- only after that import the remaining stuff
- call the main method(s)

The reason for this is that the updates should be made before the workers start
importing this config module or any modules which use config values. Otherwise
those config values would still be default and not as set on command line, which
would lead to confusion as the main process will have different values than the
workers then.

Hence, _only_ the main process should import the config module. Once it calls
finalize all the config variables imported from defaults will be wrapped inside
_config. All workers importing this module afterwards, will automagically
replace these config values with the _config script.

If the restrictions above are followed, the config module can be used as if it
was a regular module (see gp_learner.py for example) and even autocompletion
should be working:

import config
assert config.BATCH_SIZE == XYZ
"""

import logging
_logger = logging.getLogger(__name__)

from .helpers import arg_parse_config_vars
from .helpers import Config
from .helpers import str_to_bool
from .defaults import *

_config = Config()


def finalize(conf):
    _config.finalize(conf)
    _replace_config_vars_with_getters()


_config_vars_are_getters = False


def _replace_config_vars_with_getters():
    global _config_vars_are_getters
    if _config_vars_are_getters:
        return
    g = globals()
    for k in [k for k in g.keys() if k.isupper()]:
        g[k] = getattr(_config, k)
    _config_vars_are_getters = True
    _logger.info(
        'initialized config: %s',
        sorted({k: v for k, v in globals().items() if k.isupper()}.items())
    )


# noinspection PyUnresolvedReferences
def _auto_replace_on_workers():
    try:
        import scoop
        if scoop.IS_RUNNING and not scoop.IS_ORIGIN:  # origin calls finalize
            _replace_config_vars_with_getters()
    except ImportError:
        pass

_auto_replace_on_workers()
