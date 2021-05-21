import attr
from ottermatics.configuration import otterize, Configuration
from ottermatics.components import Component, ComponentIterator
from ottermatics.patterns import SingletonMeta
from ottermatics.data import DBConnection
from ottermatics.reporting import ResultsRegistry


import datetime
import os 
from uuid import uuid4

import random

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

    Analysis has an iterator which is looped over for the table creation, also a default mode with just a single evaluate()

    Several name modes exist when having an iterator, names can be based on both the analysis and iterator
    '''

    iterator = None
    _solved = False #this is used
    _uploaded = False #this is used by reporting system
    
    modes = ['default','iterator']
    mode = 'default'

    namepath_modes = ['analysis','iterator','both']
    namepath_mode = 'both'
    namepath_root = 'reports/'

    run_id = attr.ib(default=None) #this gets logged!

    _report_db = None

    @property
    def report_db(self):
        if self._report_db is not None:
            return self._report_db
        else:
            self.warning('no report db connection set!')

        #else: #this is alot, lets be more careful
        #    self.warning('autogenerating database...')
        #    report_db = DBConnection(database_name='reports')
        #    report_db.ensure_database_exists()

    @report_db.setter
    def report_db(self,inputval:DBConnection):
        if isinstance(inputval, DBConnection):
            self._report_db = inputval
        else:
            self.warning(f'Got bad value {inputval} of type {type(inputval)}')

    @property
    def solved(self):
        return self._solved

    @property
    def uploaded(self):
        if not self.solved:
            return False
        return self._uploaded

    @property
    def _report_path(self):
        '''Add some name options that work into ClientInfoMixin'''

        if self.namepath_mode == 'both' and self.mode == 'iterator' and isinstance(self.component_iterator,Component):
            return os.path.join( self.namepath_root, f'{self.name}', f'{self.component_iterator.name}' )
        elif self.namepath_mode == 'iterator' and self.mode == 'iterator' and isinstance(self.component_iterator,Component):
            return os.path.join( self.namepath_root, f'{self.component_iterator.name}' )
        else:
            if self.name != 'default':
                return os.path.join( self.namepath_root, f'{self.classname.lower()}_{self.name}' )
            return os.path.join( self.namepath_root, f'{self.classname.lower()}' )

    @property
    def component_iterator(self)-> ComponentIterator: 
        '''Override me!'''
        return self.iterator

    def solve(self,*args,**kwargs):
        '''override at your own peril'''
        prev_item = None
        self.info(f'running analysis: {self} with input {self.component_iterator}')

        if not self._solved:
            #with alive_bar(len(locations)) as bar:
            self.run_id = str(uuid4())
            if self.mode=='iterator' and self.component_iterator is not None:
                output = []
                for item in self.component_iterator:
                    #curloc = self.component_iterator.current_location
                    
                    # if prev_item is not None: #We hotpatch the table since we loop over configs
                    #     #loc._table = prev_loc._table
                    #     prev_item.reset_table()
                    
                    output.append(self.evaluate(item,*args,**kwargs))

                    self.save_data()
                    
                    prev_item = item #We hotpatch the table since we loop over configs

                self._solved = True

            else: #mode == 'default':                 
                output = self.evaluate(*args,**kwargs)
                self.save_data()
                self._solved = True
                return output

        else:
            raise Exception('Analysis Already Solved')            
    
    def post_process(self):
        '''override me!'''
        pass

    def report_data(self):
        if self.report_db and self.solved:
            try:
                rr = ResultsRegistry(self.report_db)
                rr.ensure_analysis(self)
                rr.upload_analysis(self)

            except Exception as e:
                self.error(e, 'Issue Reporting Data')
        elif not self.solved:
            self.warning('Analysis Not Solved, Cannot Upload')

        elif not self.report_db:
            self.warning('No Report Database Initated')


    def reset_analysis(self):
        self.reset_data()
        self._solved = False
        self.run_id = None

    def gsync_results(self,filename='Analysis', meta_tags = None):
        '''Syncs All Variable Tables To The Cloud'''
        with self.drive.context(filepath_root=self.local_sync_path, sync_root=self.cloud_sync_path) as gdrive:
            with self.drive.rate_limit_manager(self.gsync_results,6,filename=filename, meta_tags = meta_tags):
                
                old_sleep = gdrive._sleep_time
                gdrive.reset_sleep_time( max(old_sleep,2.5) )                
                
                gpath = gdrive.sync_path(self.local_sync_path)
                
                self.debug(f'saving as gsheets {gpath}')
                parent_id = gdrive.get_gpath_id(gpath)
                #TODO: delete old file if exists
                
                gdrive.sleep(12*random.random()) 
                
                gdrive.cache_directory(parent_id)
                gdrive.sleep()

                #Remove items with same name in parent dir
                parent = gdrive.item_nodes[parent_id]
                parent.remove_contents_with_title(filename)
                
                df = self.joined_dataframe

                #Make the new sheet
                sht = gdrive.gsheets.create(filename,folder=parent_id)
                gdrive.sleep(2*(1+gdrive.time_fuzz*random.random())) 

                wk = sht.add_worksheet(filename)
                gdrive.sleep(2*(1+gdrive.time_fuzz*random.random()))
                
                wk.rows = df.shape[0]
                gdrive.sleep(2*(1+gdrive.time_fuzz*random.random()))

                wk.set_dataframe(df,start='A1',fit=True)
                gdrive.sleep(2*(1+gdrive.time_fuzz*random.random()))                 

                for df_result in self.variable_tables:
                    df = df_result['df']
                    conf = df_result['conf']

                    if meta_tags is not None and type(meta_tags) is dict:
                        for tag,value in meta_tags.items():
                            df[tag] = value

                    gdrive.sleep(2*(1+gdrive.time_fuzz*random.random())) 
                    wk = sht.add_worksheet(conf.displayname)
                    gdrive.sleep(2*(1+gdrive.time_fuzz*random.random())) 

                    wk.rows = df.shape[0]
                    gdrive.sleep(2*(1+gdrive.time_fuzz*random.random()))
                    
                    wk.set_dataframe(df,start='A1',fit=True)
                    gdrive.sleep(2*(1+gdrive.time_fuzz*random.random())) 

                sht.del_worksheet(sht.sheet1)
                gdrive.sleep(2*(1+gdrive.time_fuzz*random.random()))

                #TODO: add in dataframe dict with schema sheename: {dataframe,**other_args}
                self.info('gsheet saved -> {}'.format(os.path.join(gpath,filename)))            

                gdrive.reset_sleep_time( old_sleep )            




#WIP
# class Report(Analysis):
#     '''Class capable of comparing several analyses'''

#     def post_process(self):
#         pass

#     def save_report(self):
#         pass
    
#     def remove_report(file):
#         pass
     