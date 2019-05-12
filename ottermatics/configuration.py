from contextlib import contextmanager

class Configuration(object):

    _temp_vars = None

    def __init__(self,**kwargs):
        for arg,val in kwargs.items():
            if hasattr(self,arg):
                setattr(self,arg,val)
            else:
                print '{} has no variable: {}'.format(self,arg)
                
    @contextmanager
    def difference(self,**kwargs):
        '''Change Variables Temporarilly'''
        self._temp_vars = {arg: getattr(self,arg) for arg in kwargs.keys() if hasattr(self,arg)}
        
        bad_vars = set.difference(set(kwargs.keys()),set(self._temp_vars.keys()))
        if bad_vars:
            print 'Could Not Change {}'.format( ','.join(list(bad_vars ) ))
        
        try: #Change Variables To Input
            for arg,var in kwargs.items():
                setattr(self,arg,var)
            yield self
        finally:
            for arg,var in self._temp_vars.items():
                setattr(self,arg,var)
    


