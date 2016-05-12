# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import logging
from os import path
import re

from .defaults import *

logger = logging.getLogger(__name__)


class Config(object):
    """Config wrapper object, distributes updated config to SCOOP workers."""
    def __init__(self):
        self._config = {}

    def finalize(self, conf):
        if self._config:
            raise RuntimeError('config was finalized already')
        self._config.update(conf)
        try:
            import scoop.shared
            if scoop.IS_RUNNING:
                scoop.shared.setConst(gp_learner_config=self._config)
        except ImportError:
            pass

    def __getattr__(self, attr):
        if not self._config:
            try:
                import scoop.shared
                if scoop.IS_RUNNING:
                    self._config = scoop.shared.getConst('gp_learner_config')
                    if not self._config:
                        raise RuntimeError('could not initialize config')
                else:
                    raise RuntimeError(
                        'you forgot to finalize config before accessing it'
                    )
            except ImportError:
                pass
        return self._config[attr]


def arg_parse_config_vars(parser, cfg_vars=None, _cfg_var_doc=OrderedDict()):
    """Config helper to be called from main to update given arg parser."""
    if not _cfg_var_doc:
        fn = path.join(path.dirname(path.realpath(__file__)), 'defaults.py')
        logger.debug('parsing options from %s' % fn)
        with open(fn) as f:
            _cfg_var_pat = re.compile(
                r'^(?P<var>[A-Z_0-9]+) = (?P<val>.+?)(  # (?P<doc>.*))?$'
            )
            for line in f:
                m = _cfg_var_pat.match(line)
                if m:
                    var = m.group('var')
                    doc = m.group('doc')
                    logger.debug('doc for config option: %s: %s', var, doc)
                    _cfg_var_doc[var] = doc

    if not cfg_vars:
        cfg_vars = list(_cfg_var_doc)
    for var in cfg_vars:
        val = globals()[var]
        logger.debug('config option: %s = %r', var, val)
        type_ = type(val)
        parser.add_argument(
            "--%s" % var,
            help=_cfg_var_doc.get(var),
            action="store",
            type=str_to_bool if type_ == bool else type_,
            default=val,
        )


def str_to_bool(s):
    """Convert str to bool."""
    s = s.lower()
    if s not in {'true', 'false', '1', '0'}:
        raise ValueError('Bool expected, but got %r' % s)
    return s in {'true', '1'}
