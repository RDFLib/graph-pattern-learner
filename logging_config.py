# encoding: utf-8

import os
import glob
import shutil
from datetime import datetime, timedelta
import time
import logging
import logging.config
import logging.handlers
import subprocess

import scoop


class IndentingFormatter(logging.Formatter):
    def format(self, record):
        s = super(IndentingFormatter, self).format(record)
        s = "\n    ".join(s.split('\n'))
        # The following is crucial for library and exception logging
        # Without it log messages can either be unicode or bytestrings. The
        # latter case will cause problems when writing non ASCII chars into the
        # logfiles, as they are opened via `codecs.open`, which will try to
        # encode the given message into 'utf-8' encoding. If it already is a
        # bytestring this will fail with a new exception, resulting in the
        # actual exception not being written to the file.
        # Hence we try interpreting every bytestring we get as utf-8 and replace
        # encoding mistakes instead of throwing errors, so we at least have
        # something in the logs.
        if not isinstance(s, unicode):
            s = s.decode('utf-8', errors='replace')
        return s


class TTSHandler(logging.Handler):
    tts_cool_down = timedelta(seconds=3)  # only "read" one msg per 3 sec at max

    def __init__(self, *args, **kwds):
        super(TTSHandler, self).__init__(*args, **kwds)
        self.last_msg_time = datetime.utcnow() - self.tts_cool_down

    def emit(self, record):
        now = datetime.utcnow()
        if now - self.last_msg_time < self.tts_cool_down:
            # last "read" isn't long enough ago, don't read it...
            return
        self.last_msg_time = now
        msg = self.format(record)
        # only print first and last line of error / exception
        msgs = msg.strip().split('\n')
        if len(msgs) > 1:
            msg = msgs[0].strip()[:32] + '\n' + msgs[-1].strip()[:32]
        else:
            msg = msg.strip()[:32]
        cmd = ['say', msg]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        # wait for the program to finish (makes logging block)
        # p.communicate()


def _gzip_file(fn):
    """Calls gzip blocking, in case of error doesn't throw an exception."""
    return subprocess.call(
        ['gzip', fn],
    )


class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Compresses older files.

    Current file is not compressed as otherwise the gzip buffer hinders watching
    live progress.
    """
    def doRollover(self):
        """Copy of parent, adapted to gzip old files, hardened for gzip fail."""
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = "%s.%02d" % (self.baseFilename, i)
                dfn = "%s.%02d" % (self.baseFilename, i + 1)
                sfns = [sfn + '.gz', sfn]
                dfns = [dfn + '.gz', dfn]
                for sfn, dfn in zip(sfns, dfns):
                    if os.path.exists(sfn):
                        # print "%s -> %s" % (sfn, dfn)
                        if os.path.exists(dfn):
                            os.remove(dfn)
                        os.rename(sfn, dfn)
            dfn = self.baseFilename + ".01"
            dfns = [dfn + '.gz', dfn]
            for dfn in dfns:
                if os.path.exists(dfn):
                    os.remove(dfn)
            # Issue 18940: A file may not have been created if delay is True.
            if os.path.exists(self.baseFilename):
                os.rename(self.baseFilename, dfn)
                _gzip_file(dfn)
        if not self.delay:
            self.stream = self._open()


# set up logging (keep multiprocessing in mind):
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_SUFFIX = '.log'
START_TIME = time.strftime('%Y-%m-%dT%H-%M-%S')
pid = os.getpid()
worker_kind = 'origin'
if scoop.IS_RUNNING:
    # noinspection PyUnresolvedReferences
    if not scoop.IS_ORIGIN:
        worker_kind = 'worker'
else:
    worker_kind = 'single'
filename = os.path.join(LOG_DIR, '%s_%s_%s' % (START_TIME, pid, worker_kind))
format_str = (
    "%(levelname)s %(asctime)-15s PID:%(process)d" + (' (%s) ' % worker_kind) +
    "%(name)s:%(module)s:%(funcName)s() in "
    'File: "%(pathname)s", line %(lineno)d\n'
    "%(message)s"
)
format_str_tts = '%(levelname)s'  # in %(name)s,\n%(message)s'
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'incremental': False,
    'formatters': {
        'indenting_formatter': {
            '()': IndentingFormatter,
            'format': format_str,
        }
    },
    'handlers': {
        'console_handler': {
            'class': 'logging.StreamHandler',
            'level': logging.INFO,
            'formatter': 'indenting_formatter',
            'stream': 'ext://sys.stderr',
        },
        'file_debug_handler': {
            '()': CompressedRotatingFileHandler,
            'level': logging.DEBUG,
            'formatter': 'indenting_formatter',
            'filename': filename + '_debug' + LOG_SUFFIX,
            'maxBytes': 2 * 1024 ** 2,  # 2 MiB
            'backupCount': 32,
            'mode': 'a',
            'encoding': 'utf-8',
        },
        'file_info_handler': {
            '()': CompressedRotatingFileHandler,
            'level': logging.INFO,
            'formatter': 'indenting_formatter',
            'filename': filename + '_info' + LOG_SUFFIX,
            'maxBytes': 2 * 1024 ** 2,  # 2 MiB
            'backupCount': 32,
            'mode': 'a',
            'encoding': 'utf-8',
        },
        'file_warning_handler': {
            'class': 'logging.FileHandler',
            'level': logging.WARNING,
            'formatter': 'indenting_formatter',
            'filename': filename + '_warning' + LOG_SUFFIX,
            'mode': 'w',
            'encoding': 'utf-8',
            'delay': True,
        },
        'file_error_handler': {
            'class': 'logging.FileHandler',
            'level': logging.ERROR,
            'formatter': 'indenting_formatter',
            'filename': filename + '_error' + LOG_SUFFIX,
            'mode': 'w',
            'encoding': 'utf-8',
            'delay': True,
        },
    },
    'root': {
        'handlers': [
            'console_handler',
            'file_debug_handler',
            'file_info_handler',
            'file_warning_handler',
            'file_error_handler',
        ],
        'level': 'DEBUG',
    },
    # 'loggers': {}  # all other loggers except for root
}

if os.name == 'posix' and os.uname()[0] == 'Darwin':
    logging_config['formatters']['tts_formatter'] = {
        'format': format_str_tts
    }
    logging_config['handlers']['tts_handler'] = {
        '()': TTSHandler,
        'level': logging.WARNING,
        'formatter': 'tts_formatter',
    }
    logging_config['root']['handlers'].append('tts_handler')

logging.config.dictConfig(logging_config)
logging.captureWarnings(True)

if scoop.IS_RUNNING:
    for h in list(scoop.logger.handlers):
        scoop.logger.removeHandler(h)


def save_error_logs():
    dir_ = os.path.join(LOG_DIR, time.strftime('error_logs_%Y-%m-%dT%H-%M-%S'))
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    for fn in glob.glob(os.path.join(LOG_DIR, '*.log*')):
        shutil.copyfile(fn, os.path.join(dir_, os.path.basename(fn)))
