from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.orm import backref
from sqlalchemy.orm import scoped_session
from sqlalchemy_utils import *

import functools

import os,sys
import logging
import numpy
import time
import signal
import functools
import time
from urllib.request import urlopen
from psycopg2.extensions import register_adapter, AsIs
import psycopg2

import cachetools

from ottermatics.env_var import EnvVariable
from ottermatics.patterns import SingletonMeta,InputSingletonMeta
from ottermatics.logging import (
    LoggingMixin,
    change_all_log_levels,
)
from ottermatics.locations import client_path

from ottermatics.common import is_ec2_instance
from ottermatics.tabulation import *  # This should be considered a module of data

from sqlalchemy_batch_inserts import enable_batch_inserting

from contextlib import contextmanager

import diskcache

log = logging.getLogger("otterlib-data")

#Env Vars
DB_NAME = EnvVariable("OTTR_DB_NAME")
DB_HOST = EnvVariable("OTTR_DB_HOST",default='localhost')
DB_PORT = EnvVariable("OTTR_DB_PORT",int,default=5432)
DB_USER = EnvVariable('OTTR_DB_USER',default='postgres')
DB_PASS = EnvVariable('OTTR_DB_PASS',default='postgres')


def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)


def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)


def addapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))


register_adapter(numpy.float64, addapt_numpy_float64)
register_adapter(numpy.int64, addapt_numpy_int64)
register_adapter(numpy.float32, addapt_numpy_float32)
register_adapter(numpy.int32, addapt_numpy_int32)
register_adapter(numpy.ndarray, addapt_numpy_array)


# This handles nans (which present as floats)!
def nan_to_null(f):
    if not numpy.isnan(f) and not numpy.isinf(f):
        return psycopg2.extensions.Float(f)
    return AsIs("NULL")


register_adapter(float, nan_to_null)

DataBase = declarative_base()

# TODO: Get this ray remote decorator working, issues with structure reliability
# @ray.remote
# class RemoteFunctionCache():

#     def __init__(self, func, maxsize=1024):
#         self.func = cachetools.cached(cachetools.LRUCache(maxsize=maxsize))(func)

#     def call(self, key, *args, **kwargs):
#         return self.func(key,*args, **kwargs)

# class CachingMaybeRemoteFunc:
#     _cache = None

#     def __init__(self, func,maxsize=1024):
#         self.func = cachetools.cached(cachetools.LRUCache(maxsize=maxsize))(func)

#     @property
#     def cache(self):
#         if self._cache is not None:
#             return self._cache

#         elif ray.is_initialized():
#             self._cache = RemoteFunctionCache.remote(self.func,maxsize=maxsize)

#         return self._cache #will be None when not using ray

#     def select_call(self,key,*args, **kwargs):
#         if self.cache is not None:
#             log.info(f'getting with remote {key}')
#             obj = self.cache.call.remote(key, *args, **kwargs)
#             return obj.get()

#         else:
#             log.info(f'getting local {key}')
#             return self.func(key, *args, **kwargs)

#     def __call__(self, key, *args, **kwargs):
#         log.info(f'__call__ {key}')
#         out = self.select_call( key, *args, **kwargs )
#         log.info(f'got value for {key}: {out}')
#         return out


# def ray_cache(func):
#     return CachingMaybeRemoteFunc(func)

import numpy as np
import matplotlib.pyplot as pl
from scipy.fftpack import fft, ifft


def autocorrelation_fft(x):
    xp = ifftshift((x - np.average(x)) / np.std(x))
    (n,) = xp.shape
    xp = np.r_[xp[: n // 2], np.zeros_like(xp), xp[n // 2 :]]
    f = fft(xp)
    p = np.absolute(f) ** 2
    pi = ifft(p)
    return np.real(pi)[: n // 2] / (np.arange(n // 2)[::-1] + n // 2)


def autocorrelation_direct(x):
    maxdelay = int(len(x) / 5)
    N = len(x)
    mean = np.average(x)
    var = np.var(x)
    xp = (x - mean) / np.sqrt(var)
    autocorrelation = np.zeros(maxdelay)
    for r in range(maxdelay):
        for k in range(N - r):
            autocorrelation[r] += xp[k] * xp[k + r]
        autocorrelation[r] /= float(N - r)
    return autocorrelation


def autocorrelation_numpy(x):
    xp = (x - np.mean(x)) / np.std(x)
    result = np.correlate(xp, xp, mode="full")
    return result[int(result.size / 2) :] / len(xp)


# def main():
#     t = np.linspace(0,20,1024)
#     x = np.exp(-t**2)
#     pl.plot(t[:200], autocorrelation_fft(x)[:200],label='scipy fft')
#     pl.plot(t[:200], autocorrelation_direct(x)[:200],label='direct autocorrelation')
#     pl.plot(t[:200], autocorrelation_numpy(x)[:200],label='numpy correlate')
#     pl.legend()
#     pl.show()


# @singleton_meta_object
class DiskCacheStore(LoggingMixin, metaclass=SingletonMeta):
    """A singleton object with safe methods for file access,
    Aims to prevent large number of file pointers open

    These should be subclassed for each cache location you want"""

    _cache = None
    size_limit = 10e9  # 10GB
    alt_path = None
    cache_class = diskcache.Cache
    timeout = 1.0
    cache_init_kwargs = None

    last_expire = None
    _current_keys = None
    expire_threshold = 60.0

    retries = 3
    sleep_time = 0.1

    def __init__(self, **kwargs):
        if kwargs:
            self.cache_init_kwargs = kwargs
        else:
            self.cache_init_kwargs = {}
        self.info(f"Created DiskCacheStore In {self.cache_root}")
        self.cache

    @property
    def cache_root(self):
        # TODO: CHECK CACHE IS NOT SYNCED TO DROPBOX
        if self.alt_path is not None:
            return os.path.join(
                client_path(skip_wsl=False), "cache", self.alt_path
            )
        return os.path.join(
            client_path(skip_wsl=False),
            "cache",
            "{}".format(type(self).__name__).lower(),
        )

    @property
    def cache(self):
        if self._cache is None:
            self.debug("setting cache")
            self._cache = self.cache_class(
                self.cache_root,
                timeout=self.timeout,
                size_limit=self.size_limit,
                **self.cache_init_kwargs,
            )
        return self._cache

    def set(self, key=None, data=None, retry=True, ttl=None, **kwargs):
        """Passes default arguments to set the key:data relationship
        :param expire: time in seconds to expire the data
        """
        if ttl is None:
            ttl = self.retries  # onstart
        self.last_expire = None

        try:
            with self.cache as ch:
                ch.set(key, data, retry=retry, **kwargs)

        except Exception as e:
            ttl -= 1
            if ttl > 0:
                time.sleep(self.sleep_time * (self.retries - ttl))
                return self.set(key=key, data=data, retry=True, ttl=ttl)
            else:
                self.error(e, "Issue Getting Item From Cache")

    # @ray_cache
    def get(self, key=None, on_missing=None, retry=True, ttl=None):
        """Helper method to get an item, return None it doesn't exist and warn.
        :param on_missing: a callback to use if the data is missing, which will set the data at the key, and return it
        """
        if ttl is None:
            ttl = self.retries  # onstart

        try:
            with self.cache as ch:
                if key in ch:
                    return ch.get(key, retry=retry)
                else:
                    if on_missing is not None:
                        data = on_missing()
                        self.set(key=key, data=data)
                        return data
                    self.warning("key {} not in cache".format(key))
            return None

        except Exception as e:
            ttl -= 1
            if ttl > 0:
                time.sleep(self.sleep_time * (self.retries - ttl))
                return self.get(
                    key=key, on_missing=on_missing, retry=True, ttl=ttl
                )
            else:
                self.error(e, "Issue Getting Item From Cache")

    def expire(self):
        """wrapper for diskcache expire method that only permits expiration on a certain interval
        :return: bool, True if expired called"""
        now = time.time()
        if (
            self.last_expire is None
            or now - self.last_expire > self.expire_threshold
        ):
            self.cache.expire()
            self.last_expire = now
            return True
        return False

    @property
    def current_keys(self):
        has_new_keys = self.expire()  # will be updated locally max every 60s
        if has_new_keys or self._current_keys is None:
            self._current_keys = set(list(self.cache))

        return self._current_keys

    def __iter__(self):
        return self.cache.__iter__()

    @property
    def identity(self):
        return "{}".format(self.__class__.__name__.lower())

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_cache"] = None  # don't pickle file objects!
        return d

    def __setstate__(self, d):
        for key, val in d.items():
            self.__dict__[key] = val
        self.cache  # create cache


class DBConnection(LoggingMixin, metaclass=InputSingletonMeta):
    """A database singleton that is thread safe and pickleable (serializable)
    to get the active instance use DBConnection.instance(**non_default_connection_args)
    """

    # TODO: Make Threadsafe W/ ThreadPoolExecutor!
    # we love postgres!
    _connection_template = "postgresql://{user}:{passd}@{host}:{port}/{database}"  

    pool_size = 20
    max_overflow = 0
    echo = False

    dbname = None
    host = None
    user = None
    passd = None
    port = 5432

    # Reset
    connection_string = None
    engine = None
    scopefunc = None
    session_factory = None
    Session = None

    _batchmode = False

    connect_args = {"connect_timeout": 5}

    def __init__(
        self, database_name=None, host=None, user=None, passd=None, **kwargs
    ):
        """On the Singleton DBconnection.instance(): __init__(*args,**kwargs) will get called, technically you
        could do it this way but won't be thread safe, or a single instance
        :param database_name: the name for the database inside the db server
        :param host: hostname
        :param user: username
        :param passd: password
        :param port: hostname
        :param echo: if the engine echos or not"""
        self.info("initalizing db connection")
        # Get ENV Defaults

        if database_name is not None:
            self.dbname = database_name
        else:
            self.dbname = DB_NAME.secret

        if host is not None:
            self.host = host
        else:
            self.host = HOST = DB_HOST.secret

        if user is not None:
            self.user = user
        else:
            self.user = USER = DB_USER.secret

        if passd is not None:
            self.info("Getting DB pass arg")
            self.passd = passd
        else:
            self.passd = PASS = DB_PASS.secret

        # Args with defaults
        if "port" in kwargs:
            self.port = kwargs["port"]
        else:
            self.port = DB_PORT.secret

        if "echo" in kwargs:
            self.echo = kwargs["echo"]
        else:
            self.echo = False

        if "batchmode" in kwargs:
            self._batchmode = True  # kwargs['batchmode']

        self.resetLog()
        self.configure()

    def configure(self):
        """A boilerplate configure method"""
        self.info("Configuring...")
        self.connection_string = self._connection_template.format(
            host=self.host,
            user=self.user,
            passd=self.passd,
            port=self.port,
            database=self.dbname,
        )
        extra_args = {}
        if self._batchmode:
            extra_args["executemany_mode"] = "values"

        self.engine = create_engine(
            self.connection_string,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            connect_args={"connect_timeout": 5},
            **extra_args,
        )
        self.engine.echo = self.echo

       # self.scopefunc = functools.partial(context.get, "uuid")

        self.session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=True
        )
        self.Session = scoped_session(self.session_factory)

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        if not hasattr(self, "Session"):
            self.configure()
        session = self.Session()
        try:
            if self._batchmode:
                enable_batch_inserting(session)

            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
        del session


    def rebuild_database(self, confirm=True):
        """Rebuild database on confirmation, create the database if nessicary"""
        if not is_ec2_instance():
            answer = input(
                "We Are Going To Overwrite The Databse {}\nType 'CONFIRM' to continue:\n".format(
                    HOST
                )
            )
        else:
            answer = "CONFIRM"

        if answer == "CONFIRM" or confirm == False:
            # Create Database If It Doesn't Exist
            if not database_exists(self.connection_string):
                self.info("Creating Database")
                create_database(self.connection_string)
            else:
                # Otherwise Just Drop The Tables
                self.debug("Dropping DB Metadata")
                DataBase.metadata.drop_all(self.engine)
            # (Re)Create Tables
            self.debug("Creating DB Metadata")
            DataBase.metadata.create_all(self.engine)
        else:
            try:
                raise Exception("Ah ah ah you didn't say the magic word")
            except Exception as e:
                self.error(e)

    def ensure_database_exists(self, create_meta=True):
        """Check if database exists, if not create it and tables"""
        self.info(f"checking database existinence... {self.engine}")
        if not database_exists(self.connection_string):
            self.info("doesn't exist, creating database!")
            create_database(self.connection_string)
            if create_meta:
                DataBase.metadata.create_all(self.engine)

    def cleanup_sessions(self):
        self.info("Closing All Active Sessions")
        self.Session.close_all()

    @property
    def identity(self):
        return "DB Con: {s.user}@{s.dbname}".format(s=self)

    def __getstate__(self):
        """Remove active connection objects, they are not picklable"""
        # TODO: Should we remove credentials? How do we distibute object
        d = self.__dict__.copy()
        d["connection_string"] = None
        d["engine"] = None
        d["scopefunc"] = None
        d["session_factory"] = None
        d["Session"] = None
        return d

    def __setstate__(self, d):
        """We reconfigure on opening a pickle"""
        self.__dict__ = d
        self.configure()
