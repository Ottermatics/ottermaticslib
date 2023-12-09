"""Defines a class called `EnvVariable` that defines an interface for env variables with an option to obscure and convert variables, as well as provide a default option.

A global record of variables is kept for informational purposes in keeping track of progam variables

To prevent storage of env vars in program memory, access to the os env variables is provided on access of the `secret` variable. It is advisable to use the result of this as directly as possible when dealing with actual secrets. 

For example add: `db_driver(DB_HOST.secret,DB_PASSWORD.secret,...)
"""


import os
from ottermatics.logging import LoggingMixin
from typing import Any
import socket
import inspect

global warned
warned = set()  # a nice global variable to hold any warnings


class EnvVariable(LoggingMixin):
    """A method to wrap SECRETS and in application with a way to get the value using self.secret
    Do not store values from self.secret to ensure security

    You can override the secret with _override"""

    var_name: str = None
    type_conv: Any = None
    default: Any = None
    obscure: bool = True
    _override: str
    _secrets = {}  # its class based so like a singleton
    _replaced = set()
    fail_on_missing: bool
    desc: str = None
    _upgrd_warn:bool = False

    def __init__(
        self,
        secret_var_name,
        type_conv=None,
        default=None,
        obscure=False,
        fail_on_missing=False,
        desc: str = None,
        dontovrride=False,
    ):
        """pass arguments to SecretVariable to have it look up information at runtime from envargs, but not store it in memory.
        :param secret_var_name: the enviornmental variable
        :param type_conv: the data from env vars will be converted with this function
        :param default: the value to use if the secret_var_name doesn't exist in enviornmental variables
        :param obscure: default True, will prevent the result being printed by str(self)
        :param fail_on_missing: if the secret env variable is not found, and default is None
        :param desc: a description of the purpose of the variable
        """
        self.var_name = secret_var_name
        self.type_conv = type_conv

        if default is not None:
            self.default = default
        self.obscure = obscure
        # UserString.__init__(self,f'[SECRET:{secret_var_name}]')

        self.fail_on_missing = fail_on_missing

        # record env vars
        if secret_var_name in self.__class__._secrets:
            cur = self.__class__._secrets[secret_var_name]
            if dontovrride:
                pass
            else:
                self.debug(f"replacing {cur}->{self}")
                self._replaced.add(cur)
                self.__class__._secrets[secret_var_name] = self
        else:
            self.__class__._secrets[secret_var_name] = self

        # FIXME: prevent ottermatics var from replacing other module instnace
        # not possible to locate where other instances
        # if secret_var_name in self.__class__._secrets:
        #     cur = self.__class__._secrets[secret_var_name]
        #     if cur != self and self not in self._replaced:
        #         self._replaced.add(cur)
        #         self.info(f'replacing {cur}->{self}')
        #         self.__class__._secrets[secret_var_name] = self
        #     elif self in self._replaced:
        #         self.info(f'skipping replaced readd {self}')
        #         #self.__class__._secrets[secret_var_name] = self
        # else:
        #     self.__class__._secrets[secret_var_name] = self

    def __str__(self):
        if self.obscure:
            return f"{self.obscured_name:<40} = XXXXXX"
        return f"{self.obscured_name:<40} = {self.secret}"

    def __add__(self, other) -> str:
        return str(str.__add__(str(self), other))

    def __radd__(self, other) -> str:
        return str(str.__add__(other, str(self)))

    @property
    def obscured_name(self) -> str:
        if hasattr(self, "_override"):
            return f"SECRETS[OVERRIDE]"
        return f"SECRETS[{self.var_name}]"

    @property
    def secret(self):

        #Check if this secret is the one in the secrets registry
        sec = self.__class__._secrets[self.var_name]
        if sec is not self:

            #Provide warning that the secret is being replaced
            if not self._upgrd_warn:
                self._upgrd_warn = True
                self.info(f'upgrading: {self.var_name} from {id(self)}->{id(sec)}')
                
            #Monkeypatch dictionary
            self.__dict__ = sec.__dict__

        if hasattr(self, "_override"):
            return self._override

        if self.var_name in os.environ:
            secval = os.environ[self.var_name]
        elif self.default is not None:
            if self.var_name not in warned:
                if self.obscure:
                    dflt = "XXXXXXX"
                else:
                    dflt = self.default

                self.info(f"Env Var: {self.var_name} Not Found! Using: {dflt}")
                warned.add(self.var_name)

            secval = self.default
        else:
            if self.fail_on_missing:
                raise FileNotFoundError(
                    f"Could Not Find Env Variable {self.var_name}"
                )
            else:
                if self.var_name not in warned:
                    self.info(f"Env Var: {self.var_name} Not Found!")
                    warned.add(self.var_name)
                return None

        if self.type_conv is None:
            return secval
        else:
            return self.type_conv(secval)

    @property
    def in_env(self):
        return self.var_name in os.environ

    def remove(self):
        """removes this secret from the record"""
        if self in self.__class__._secrets:
            self.__class__._secrets.remove(self)

    @classmethod
    def load_env_vars(self):
        for s in EnvVariable._secrets.values():
            str(s)

    @classmethod
    def print_env_vars(cls):
        """prints env vars in memory"""
        # preload
        cls.load_env_vars()
        for var, s in sorted(
            EnvVariable._secrets.items(), key=lambda kv: kv[1].var_name
        ):
            print(f"{s.var_name:<40}|{s}")


# DEFAULT ENV VARIABLES
try:
    # This should always work unless we don't have privideges (rare assumed)
    host = socket.gethostname().upper()
except:
    host = "MASTER"

global HOSTNAME, SLACK_WEBHOOK

HOSTNAME = EnvVariable(
    "OTTR_HOSTNAME", default=host, obscure=False, dontovrride=True
)
SLACK_WEBHOOK = EnvVariable(
    "OTTR_SLACK_LOG_WEBHOOK", default=None, obscure=False, dontovrride=True
)
