import attr
from ottermatics.configuration import otterize
from ottermatics.components import Component
import datetime

@otterize  
class Analysis(Component):
    '''A type of configuration that will reach down among all attribues of a configuration,
    and interact with their solvers and stored data
    
    data will be stored a colmn row type arrangement in csv, database table (via sqlalchemy), 
    and gsheets table, a saving strategy will be chosen using _____________
    
    configuration analysis's own table will be structure mostly around primary variables for this 
    problem, they could be created by hand for your table, reaching into other configuraitons, or 
    they could be the default.

    We will keep track of how deep we reach into the configuration to store tables, `store_level` 
    starting at zero will only save the table of this configuration
    '''

    _solved = False

    def solve(self,*args,**kwargs):
        '''dont override this unless you call evaluate, followed by save_data. 
        This will maintain table saving functionality'''
        self.info("Solving {}...".format(self.displayname))
        if not self._solved:
            output = self.evaluate(*args,**kwargs)
            self.save_data()
            self._solved = True
            return output
        else:
            raise Exception('Analysis Already Solved')

    def reset_analysis(self):
        self.reset_data()
        self._solved = False



class Report(Analysis):
    '''Class capable of comparing several analyses'''

    def post_process(self):
        pass

    def save_report(self):
        pass
    
    def remove_report(file):
        pass
     