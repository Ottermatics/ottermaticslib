import attr
from ottermatics.configuration import *
from ottermatics.gdocs import *
import datetime

@otter_class  
class AnalysisConfiguration(Configuration):
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

    #tabulation parameters
    max_col_width_static:int = 10
    _store_level:int = -1

    _solved = False

    def solve(self,*args,**kwargs):
        '''dont override this unless you call evaluate, followed by save_data. 
        This will maintain table saving functionality'''
        self.info("Solving Analysis...")
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

    #High Level Component Tables
    @property
    def toplevel_static(self):
        out_df = self.cleanup_dataframe( self.static_dataframe )
        df_list = self.split_dataframe_by_colmum(out_df,self.max_col_width_static)

        return {'conf':self,'dfs':  df_list }

    @property
    def variable_tables(self):
        '''Grabs all valid variable dataframes and puts them in a list'''

        rds = self.recursive_data_structure(self.store_level)

        output = []
        for index, components in rds.items():
            for comp in components:
                if comp['variable'] is not None:
                    output.append({'conf':comp['conf'],'df':comp['variable']})

        return output

    @property
    def joined_table(self):
        if self.variable_tables:
            return pandas.concat([ vt['df'] for vt in self.variable_tables],axis=1)
        return None

    def get_field_from_table(self,field):
        '''Converts Pandas To Numpy Array By Key, also handles poorly formated fields'''
        if field in self.joined_table:                                              
            table = self.joined_table[field]
        elif field.title():
            table = self.joined_table[field.title()]
        else:
            raise Exception('No Field Named {}'.format(field))
        
        #Remove Infinity
        table = table.replace([numpy.inf, -numpy.inf], numpy.nan)
        return table.to_numpy(dtype=float,copy=True)
        

    @property
    def other_static_tables(self):
        rds = self.recursive_data_structure(self.store_level)
        output = []
        for index, components in rds.items():
            if index > 0:
                for comp in components:
                    if comp['static'] is not None:
                        df_list = self.split_dataframe_by_colmum(comp['static'],self.max_col_width_static)                        
                        output.append({'conf':comp['conf'],'dfs': df_list})
        return output

    def save_to_worksheet(self,worksheet:pygsheets.Worksheet):
        '''Saves to a gsheets via pygsheets'''

        title = self.identity.replace('_',' ').title()

        self.info('saving worksheet as {}'.format(title))
        wksh = worksheet

        wksh.clear()

        #Static data
        start = pygsheets.Address((2,2))

        tld = self.toplevel_static
        sdf = tld['dfs']

        cur_index = start + (1,0)
        for i,df in enumerate(sdf):
            self.debug('saving dataframe {}'.format(df))
            wksh.update_value(start.label,self.identity)
            wksh.cell(start.label).set_text_format('bold',True)
            wksh.set_dataframe(df,cur_index.label , extend=True)
            cur_index += (2,0)

        cur_index += (3,0)

        var_index = pygsheets.Address(cur_index.label)

        max_row = 0

        vrt = self.variable_tables
        self.info('saving {} other static tables'.format(len(vrt)))

        for dfpack in vrt:
            conf = dfpack['conf']
            df = dfpack['df']
            self.debug('saving dataframe {}'.format(df))

            (num_rows,num_cols) = df.shape
            max_row = max(max_row,num_rows)
            
            wksh.update_value((var_index-(1,0)).label,conf.classname)
            wksh.cell((var_index-(1,0)).label).set_text_format('bold',True)    

            wksh.set_dataframe(df,start=var_index.label,extend=True)
            
            var_index += (0,num_cols)
            
        cur_index += (3+max_row,0) 
            
        ost = self.other_static_tables

        self.info('saving {} other static tables'.format(len(ost)))

        for dfpack in ost:
            conf = dfpack['conf']
            sdf = dfpack['dfs']

            wksh.update_value((cur_index-(1,0)).label,conf.identity)
            wksh.cell((cur_index-(1,0)).label).set_text_format('bold',True)            

            for i,df in enumerate(sdf):
                
                self.debug('saving {} dataframe {}'.format(conf.identity, df))
                wksh.set_dataframe(df,start=cur_index.label ,extend=True)
                cur_index += (2,0)
            
            cur_index += (3,0)
        


    #Report Functionality
    def post_process(self):
        pass

    def save_report(self):
        pass
    
    def remove_report(file):
        pass
     