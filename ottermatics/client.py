
import os, shutil, sys
import re
import pathlib
from pathlib import Path
from platform import system,uname

from ottermatics.common import *
from ottermatics.locations import *
from ottermatics.logging import LoggingMixin
from ottermatics.gdocs import OtterDrive
import logging

class GCloudNotEnabled(Exception): pass

class ClientInfoMixin(LoggingMixin):
    '''In which we provide methods to infer a client, and provide path targets for reports ect'''
    stored_path = None
    stored_client_folder = None
    stored_client_name = None

    #plotting & post processing calls
    _stored_plots = []
    _report_path = 'reports/'
    _default_image_format = '.png'

    _drive = None
    _drive_init_args = None

    @property
    def client_name(self):
        
        if self.stored_client_name is None:
            if 'CLIENT_NAME' in os.environ:
                self.stored_client_folder = os.environ['CLIENT_NAME']

            elif in_client_dir(): #infer from path
                self.stored_client_name = client_dir_name()

            else:
                self.stored_client_folder  = 'Ottermatics'
                self.warning('no client info found - using{}'.format(self.stored_client_folder))

        return self.stored_client_folder
            

    @property
    def client_folder_root(self):
        if self.stored_client_folder is None:
            self.stored_client_folder = ottermatics_project(self.client_name)
        return self.stored_client_folder

    @property
    def report_path(self):
        try:
            if self.stored_path is None:
                self.stored_path =  os.path.realpath( os.path.join( self.client_folder_root, self._report_path ) )

            if not os.path.exists( self.stored_path ):
                self.warning('report does not exist {}'.format( self.stored_path ))

            return self.stored_path

        except Exception as e:
            self.error(e)
        return None

    @property
    def report_path_daily(self):
        start_date = self._created_datetime.date()
        return os.path.join(self.report_path,'{}'.format(start_date).replace('-','_'))                  

    @property
    def config_path(self):
        return os.path.join(self.report_path,self.filename)

    @property
    def config_path_daily(self):
        return os.path.join(self.report_path_daily,self.filename)


    #Sync Convenicne Functions
    @property
    def drive(self):
        '''wrapper for private OtterDrive Instance'''
        if self._drive is None:
            raise GCloudNotEnabled('use self.enable_cloud_sync(...) to configure an OtterDrive')
        return self._drive
    
    def enable_cloud_sync(self,**kwargs):
        '''Ensure configuration's daily path is created'''
        self.info('enabling cloud sync')
        self._drive = OtterDrive(**kwargs)
        self._drive_init_args = kwargs #self.drive.initial_args
        
    def ensure_sync_path_exist(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._drive.filepath_root = path #Hacks

        gpath = self.drive.sync_path(path)
        self.drive.ensure_g_path_get_id( gpath )

    def gsync_this_config(self,force = True):
        #Sync all local to google
        old_root = self.drive.filepath_root
        try:
            self.ensure_sync_path_exist(self.config_path_daily)
            self.drive.filepath_root = self.config_path_daily
            self.drive.sync(force=force)
        except Exception as e:
            self.error(e, 'issue syncing config')
        finally:
            self.drive.filepath_root = old_root


    def gsync_this_report(self,force = True):
        #Sync all local to google
        old_root = self.drive.filepath_root
        try:
            self.ensure_sync_path_exist(self.report_path_daily)
            self.drive.filepath_root = self.report_path_daily
            self.drive.sync(force=force)
        except Exception as e:
            self.error(e, 'issue syncing config')
        finally:
            self.drive.filepath_root = old_root        

    def gsync_all_reports(self,force = True):
        #Sync all local to google
        old_root = self.drive.filepath_root
        try:
            self.ensure_sync_path_exist(self.report_path)
            self.drive.filepath_root = self.report_path
            self.drive.sync(force=force)
        except Exception as e:
            self.error(e, 'issue syncing config')
        finally:
            self.drive.filepath_root = old_root 
        
    def gsync_client_folder(self,force = True):
        #Sync all local to google
        old_root = self.drive.filepath_root
        try:
            self.ensure_sync_path_exist(self.client_folder_root)
            self.drive.filepath_root = self.client_folder_root
            self.drive.sync(force=force)
        except Exception as e:
            self.error(e, 'issue syncing config')
        finally:
            self.drive.filepath_root = old_root 