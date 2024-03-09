import logging
import logging
import traceback
import sys, os
from pyee import EventEmitter
from termcolor import colored
import requests
import json
import uuid

global log_change_emitter
log_change_emitter = EventEmitter()


BASIC_LOG_FMT = "[%(name)-24s]%(message)s"

global LOG_LEVEL
LOG_LEVEL = logging.INFO


def change_all_log_levels(inst,new_log_level: int, check_function=None):
    """Changes All Log Levels With pyee broadcast before reactor is running
    :param new_log_level: int - changes unit level log level (10-msg,20-debug,30-info,40-warning,50-error,60-crit)
    :param check_function: callable -> bool - (optional) if provided if check_function(unit) is true then the new_log_level is applied
    """
    print(f"changing log levels to {new_log_level}...")
    if isinstance(new_log_level, float):
        new_log_level = int(new_log_level)  # Float Case Is Handled

    assert (
        isinstance(new_log_level, int)
        and new_log_level >= 1
        and new_log_level <= 100
    )

    global LOG_LEVEL
    LOG_LEVEL = new_log_level

    log.info(f"Changing All Logging Units To Level {new_log_level}")
    log_change_emitter.emit("change_level", new_log_level, check_function)
    LoggingMixin.log_level = new_log_level
    inst.log_level = new_log_level


class LoggingMixin(logging.Filter):
    """Class to include easy formatting in subclasses"""

    log_level = LOG_LEVEL
    _log = None

    log_on = True

    log_fmt = "[%(name)-24s]%(message)s"

    slack_webhook_url = None
    # log_silo = False

    change_all_log_lvl = lambda s, *a, **kw: change_all_log_levels(s,*a, **kw)

    @property
    def logger(self):
        if self._log is None:
            inst_log_name = (
                "engforgelog_" + self.identity + "_" + str(uuid.uuid4())
            )
            self._log = logging.getLogger(inst_log_name)
            self._log.setLevel(level=self.__class__.log_level)

            # Apply Filter Info
            self._log.addFilter(self)
            self.installSTDLogger()
            from engforge.env_var import EnvVariable, SLACK_WEBHOOK

            # Hot Patch Class (EnvVar is logging mixin... soo... here we are)
            if LoggingMixin.slack_webhook_url is None:
                # Do this on the fly since we SecretVariable is a log component
                LoggingMixin.slack_webhook_url = SLACK_WEBHOOK

        if not hasattr(self, "_f_change_log"):

            def _change_log(new_level, check_function=None):
                if new_level != self.log_level:
                    if check_function is None or check_function(self):
                        msg = f"changing {self.identity} log level: {self.log_level} -> {new_level}"
                        self.__class__.log_level = new_level
                        self.info(msg)
                        self._log.setLevel(new_level)
                        self.log_level=new_level
                        self.resetLog()

            log_change_emitter.add_listener("change_level", _change_log)

            self._f_change_log = _change_log

        return self._log

    def resetLog(self):
        """reset log"""
        self._log = None
        self.debug(f"reset!")

    def resetSystemLogs(self, reseted=None):
        """resets log on all internal instance LoggingMixins"""
        self.resetLog()
        self.debug(f"reset!")
        if reseted is None:
            reseted = set()
        for k, v in self.__dict__.items():
            if isinstance(v, LoggingMixin) and id(v) not in reseted:
                reseted.add(id(v))
                v.resetSystemLogs(reseted)

    def installSTDLogger(self):
        """We only want std logging to start"""
        sh = logging.StreamHandler(sys.stdout)
        peerlog = logging.Formatter(self.log_fmt)
        sh.setFormatter(peerlog)
        self._log.addHandler(sh)

    def add_fields(self, record):
        """Overwrite this to modify logging fields"""
        pass

    def filter(self, record):
        """This acts as the interface for `logging.Filter`
        Don't overwrite this, use `add_fields` instead."""
        record.name = self.identity.lower()[:24]
        self.add_fields(record)
        return True

    def msg(self, *args):
        """Writes to log... this should be for raw data or something... least priorty"""
        if self.log_on:
            self.logger.log(
                1, self.message_with_identiy(self.extract_message(args), "blue")
            )

    def debug(self, *args):
        """Writes at a low level to the log file... usually this should
        be detailed messages about what exactly is going on"""
        if self.log_on:
            self.logger.debug(
                self.message_with_identiy(self.extract_message(args), "cyan")
            )

    def info(self, *args):
        """Writes to log but with info category, these are important typically
        and inform about progress of process in general"""
        if self.log_on:
            self.logger.info(
                self.message_with_identiy(self.extract_message(args), "white")
            )

    def warning(self, *args):
        """Writes to log as a warning"""
        self.logger.warning(
            self.message_with_identiy(
                "WARN: " + self.extract_message(args), "yellow"
            )
        )

    def error(self, error, msg=""):
        """Writes to log as a error"""

        # fmt = 'ERROR: {msg!r}|{err!r}'

        tb = error.__traceback__
        fmt = "ERROR:{msg}->{err}"

        tb = "\n".join(traceback.format_exception(error, value=error, tb=tb))
        msgfmt = ("\n" + " " * 51 + "|").join(str(msg).split("\n"))

        # tbcl = colored(tb, "red")
        # self.logger.exception( fmt.format(msg=msgfmt,err=tbcl))
        # self.logger.exception( msgfmt)

        m = colored(fmt.format(msg=msgfmt, err=tb), "red")
        self.logger.error(m)

    def critical(self, *args):
        """A routine to communicate to the root of the server network that there is an issue"""
        msg = self.extract_message(args)
        msg = self.message_with_identiy(msg, "magenta")
        self.logger.critical(msg)

        # FIXME: setup slack notificatinos with env var
        self.slack_notification(self.identity.title(), msg)

    def slack_notification(self, category, message):
        from engforge.env_var import SLACK_WEBHOOK, HOSTNAME

        if SLACK_WEBHOOK.var_name in os.environ:
            self.info("getting slack webhook")
            url = SLACK_WEBHOOK.secret
        else:
            return
        stage = HOSTNAME.secret
        headers = {"Content-type": "application/json"}
        data = {
            "text": "{category} on {stage}:\n```{message}```".format(
                category=category.upper(), stage=stage, message=message
            )
        }
        self.info(f"Slack Notification : {url}:{category},{message}")
        slack_note = requests.post(
            url, data=json.dumps(data).encode("ascii"), headers=headers
        )

    def message_with_identiy(self, message: str, color=None):
        """converts to color and string via the termcolor library
        :param message: a string convertable entity
        :param color: a color in [grey,red,green,yellow,blue,magenta,cyan,white]
        """
        if color != None:
            return colored(str(message), color)
        return str(message)

    def extract_message(self, args):
        for arg in args:
            if type(arg) is str:
                return arg
        if self.log_level < 0:
            print(f"no string found for {args}")
        return ""

    @property
    def identity(self):
        return type(self).__name__

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_f_change_log"] = None
        return d


class Log(LoggingMixin):
    pass


log = Log()


# try:
#     logging.getLogger('parso.cache').disabled=True
#     logging.getLogger('parso.cache.pickle').disabled=True
#     logging.getLogger('parso.python.diff').disabled=True
#
# except Exception as e:
#     log.warning(f'could not diable parso {e}')
# def installGELFLogger():
#     '''Installs GELF Logger'''
#     # self.gelf = graypy.GELFTLSHandler(GELF_HOST,GELF_PORT, validate=True,\
#     #                         ca_certs=credfile('graylog-clients-ca.crt'),\
#     #                         certfile = credfile('test-client.crt'),
#     #                         keyfile = credfile('test-client.key')
#     #                         )
#     log = logging.getLogger('')
#     gelf = graypy.GELFUDPHandler(host=GELF_HOST,port=12203, extra_fields=True)
#     log.addHandler(gelf)


# def installSTDLogger(fmt = BASIC_LOG_FMT):
#     '''We only want std logging to start'''
#     log = logging.getLogger('')
#     sh = logging.StreamHandler(sys.stdout)
#     peerlog = logging.Formatter()
#     sh.setFormatter(peerlog)
#     log.addHandler( sh )
#
#
# def set_all_loggers_to(level,set_stdout=False,all_loggers=False):
#     global LOG_LEVEL
#     LOG_LEVEL = level
#
#     if set_stdout: installSTDLogger()
#
#     logging.basicConfig(level = LOG_LEVEL) #basic config
#
#     log = logging.getLogger()
#     log.setLevel(LOG_LEVEL)# Set Root Logger
#
#     log.setLevel(level) #root
#
#     loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
#     for logger in loggers:
#         if logger.__class__.__name__.lower().startswith('engforge'):
#             logger.log(LOG_LEVEL,'setting log level: {}'.format(LOG_LEVEL))
#             logger.setLevel(LOG_LEVEL)
#         elif all_loggers:
#             logger.log(LOG_LEVEL,'setting log level: {}'.format(LOG_LEVEL))
#             logger.setLevel(LOG_LEVEL)
