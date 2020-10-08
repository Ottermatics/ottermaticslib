from contextlib import contextmanager
import attr

from .logging import LoggingMixin

import numpy
import functools


class inst_vectorize(numpy.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

@attr.s
class Configuration(LoggingMixin):

    _temp_vars = None

    name = attr.ib(default='default')

    log_fmt = "[%(identity)-24s]%(message)s"
    log_silo = True

    @contextmanager
    def difference(self,**kwargs):
        '''Change Variables Temporarilly'''
        _temp_vars = {}
            
        _temp_vars.update({arg: getattr(self,arg) for arg in kwargs.keys() if hasattr(self,arg)})

        bad_vars = set.difference(set(kwargs.keys()),set(_temp_vars.keys()))
        if bad_vars:
            print('Could Not Change {}'.format( ','.join(list(bad_vars ) )))

        try: #Change Variables To Input
            for arg,var in kwargs.items():
                setattr(self,arg,var)
            yield self
        finally:
            for arg in kwargs.keys():
                var = _temp_vars[arg]
                setattr(self,arg,var)

    def add_fields(self, record):
        '''Overwrite this to modify logging fields'''
        record.identity = self.identity

    @property
    def filename(self):
        return self.name.replace(' ','_')

    @property
    def identity(self):
        return '{}-{}'.format(type(self).__name__,self.name).lower()