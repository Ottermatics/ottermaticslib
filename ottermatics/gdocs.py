import os
from ottermatics.locations import google_api_token, creds_folder, ottermatics_clients, bool_from_env, client_dir_name, ottermatics_project
from ottermatics.logging import LoggingMixin
import pygsheets
import pydrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import time

import attr

GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = google_api_token()

#Filed will be None until assigned an folder id
STANDARD_FOLDERS = {'ClientFolders':None,'InternalClientDocuments':None}
#We'll hook this up with functionality at the module level
CLIENT_G_DRIVE,CLIENT_GDRIVE_SYNC,CLIENT_GMAIL = None,None,None
if 'CLIENT_GDRIVE_PATH' in os.environ:
    CLIENT_G_DRIVE = os.environ['CLIENT_GDRIVE_PATH']
if 'CLEINT_GDRIVE_SYNC' in os.environ:
    CLIENT_GDRIVE_SYNC = bool_from_env(os.environ['CLIENT_GDRIVE_SYNC'])
if 'CLIENT_GMAIL' in os.environ:
    CLIENT_GMAIL = os.environ['CLIENT_GMAIL']    


class OtterDrive(LoggingMixin):
    '''Authenticates To Your Google Drive To Edit Sheets and Your Drive Files
    :param otter_sheets: is the google sheets object
    :param otter_drive: is the google drive object'''    
    gauth = None
    gsheets = None
    gdrive = None

    #Store folder paths, so we only have to look up once, key is path, value is id
    _folder_ids = None
    _folder_contents = None
    sync_root = ''
    client_folder = None

    file_cache = {}

    def __init__(self,sync_root=CLIENT_G_DRIVE):
        ''':param sync_root: the client path to sync to on command'''
        
        #self._folder_contents = {'/': None} #Should be lists when activated
        self.sync_root = sync_root
        self._folder_ids = {'/':'root',None:'root','root':'root','':'root'} #Should be lists when activated
        
        self.file_cache = {}

        self.authoirze_google_integrations()

        self.client_name = client_dir_name()
        self.client_folder = ottermatics_project(self.client_name)

    #Initalization Methods
    @property
    def is_syncable_to_client_gdrive(self):
        if self.client_folder is not None and self.sync_root is not None:
            return True
        return False

    def authoirze_google_integrations(self):
        self.info('Authorizing...')
        self.gauth = GoogleAuth()
        self.gauth.LoadCredentialsFile(os.path.join(creds_folder(),"gdrive_ottercreds.txt"))
        if self.gauth.credentials is None:
            # Authenticate if they're not there
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            # Refresh them if expired
            self.gauth.Refresh()
        else:
            # Initialize the saved creds
            self.gauth.Authorize()
        # Save the current credentials to a file
        self.gauth.SaveCredentialsFile(os.path.join(creds_folder(),"gdrive_ottercreds.txt"))

        #Do Sheets Authentication
        self.gsheets = pygsheets.authorize( client_secret=google_api_token(),\
                                            credentials_directory =creds_folder())

        self.gdrive = GoogleDrive(self.gauth)

        self.info('Authorized')

    def initalize_google_drive(self,parent_id='root'):
        self.info('initalizing ottermatics on gdrive')
        for sfol,sref in STANDARD_FOLDERS.items():
            if sref is None:
                fol = self.get_or_create_folder(sfol,parent_id=parent_id)
                STANDARD_FOLDERS[sfol] = fol['id']
                for clientname in ottermatics_clients():
                    if clientname != 'Archive':
                        self.get_or_create_folder(clientname,parent_id=fol['id'])

    def sync_to_client_folder(self,force=False):
        skipped_paths = []
        if self.is_syncable_to_client_gdrive:
            self.info('syncing {} to client folder: {}'.format(self.client_folder,self.sync_root))
            
            if self.sync_root not in self.folder_cache:
                parent_id = self.ensure_g_path_get_id(self.sync_root)
            for dirpath, dirnames, dirfiles in os.walk(self.client_folder):
                
                try:
                    any_hidden = any([ pth.startswith('.') for pth in dirpath.split(os.sep)])

                    if '.skip_gsync' in dirfiles or '.skip_gsync' in dirnames:
                        self.info('skipping {}'.format(dirpath))
                        skipped_paths.append(dirpath)
                        continue

                    if any([os.path.commonpath([dirpath,spath]) == os.path.commonpath([spath]) \
                                                                            for spath in skipped_paths]):
                        self.info('skipping {}'.format(dirpath))
                        continue


                    if not any_hidden:
                        time.sleep(1)
                        self.debug('syncing {}'.format(dirpath))
                        dirpath = os.path.realpath(dirpath)
                        gdrive_path = self.sync_path(dirpath)

                        #parent_dir = os.path.split(gdrive_path)[0]
                        if self.sync_root != gdrive_path:
                            parent_id = self.ensure_g_path_get_id(gdrive_path)

                        #This folder
                        if gdrive_path not in self.folder_cache:
                            gdir_id = self.get_or_create_folder(os.path.split(gdrive_path)[-1],parent_id)['id']
                        else:
                            gdir_id = self.folder_cache[gdrive_path]
                        
                        self.sync_folder_contents_locally(gdir_id)
                        
                        #Folders
                        for folder in dirnames:
                            absfold = os.path.join(dirpath,folder)
                            gfold = self.sync_path(absfold)
                            if gfold not in self.folder_cache:
                                if self.valid_sync_file(folder):
                                    gfol = self.get_or_create_folder(folder,gdir_id)
                                else:
                                    self.debug('skipping {}'.format(absfold))                                
                            

                        #Finally files
                        for file in dirfiles:
                            absfile = os.path.join(dirpath,file)
                            gfile = self.sync_path(absfile)
                            if gfile not in self.file_cache or force:
                                if self.valid_sync_file(file):
                                    self.upload_or_update_file(absfile,gdir_id)
                                else:
                                    self.debug('skipping {}'.format(file))
                            else:
                                self.debug('found existing {}'.format(gfile))
                    else:
                        self.debug('skipping hidden dir {}'.format(dirpath))
                        
                except Exception as e:
                    self.warning('Failure in gdrive sync: {}'.format(e))
                             
        else:
            self.warning('not able to sync: client folder: {} sync root: {}'.format(self.client_folder,self.sync_root))

    def valid_sync_file(self,file):
        if file.startswith('.'):
            return False
        if file.startswith('__') and file.endswith('__'):
            return False
        return True

    def sync_path(self,path):
        rel_root = os.path.relpath(path,self.client_folder)
        gdrive_root = os.path.join(self.sync_root,rel_root)
        #remove current directory /.
        if gdrive_root.endswith('{}.'.format(os.sep)):
            gdrive_root = os.path.split(gdrive_root)[0]
        if gdrive_root.startswith(os.sep):
            gdrive_root = gdrive_root.replace(os.sep,'',1)
        return gdrive_root

    #User Methods
    def upload_or_update_file(self,file_path,parent_id):
        #Only sync from client folder
        assert os.path.commonpath([self.client_folder]) == os.path.commonpath([file_path, self.client_folder])

        #Check existing
        file_path = os.path.realpath(file_path)
        file_name = os.path.basename(file_path)
        gfile_path = self.sync_path(file_path)

        if gfile_path in self.file_cache:
            self.info('updating {}->{}'.format(parent_id,file_path))
            file = self.gdrive.CreateFile({"id": self.file_cache[gfile_path],
                                            'parents': [{'id': parent_id }]})

        else:
            self.info('creating {}->{}'.format(parent_id,file_path))
            file = self.gdrive.CreateFile({"title":file_name,
                                         'parents': [{'id': parent_id }]})

        file.SetContentFile(file_path)
        file.Upload() # Upload the file.
        self.debug('uploaded {}->{}'.format(parent_id,file_path))


        
        self.add_cached_file(file,gfile_path)
        time.sleep(1)


    def add_cached_folder(self,folder_meta,parent_id):
        if parent_id in self.reverse_folder_cache:
            parent_path = self.reverse_folder_cache[parent_id]
            folder_path = os.path.join('',parent_path,folder_meta['title'])
            self.debug('caching file {}'.format(folder_path))
            if folder_meta['mimeType'] == 'application/vnd.google-apps.folder':
                self.folder_cache[folder_path] = folder_meta['id']
            elif folder_meta['mimeType'] == 'application/vnd.google-apps.shortcut':
                shortcut = self.gdrive.CreateFile({'id':folder_meta['id']})
                shortcut.FetchMetadata(fields='shortcutDetails')
                if 'shortcutDetails' in shortcut:
                    details = shortcut['shortcutDetails']
                    if 'targetId' in details and details['targetMimeType'] == 'application/vnd.google-apps.folder':
                        self.folder_cache[folder_path] = details['targetId']

    def add_cached_file(self,file_meta,gfile_parent_path):

        if gfile_parent_path is not None:
            
            gfile_path = os.path.join(gfile_parent_path,file_meta['title'])
            self.debug('caching file {}'.format(gfile_path))

            self.file_cache[gfile_path] = file_meta['id']



    def get_or_create_folder(self,folder_name,parent_id='root',**kwargs):
        '''Creates a folder in the parent folder if it doesn't already exist, otherwise return folder
        :param folder_name: the name of the folder to create
        :param parent_id: the id of the parent folder, if None will create in the parent directory
        '''
        if parent_id is None: parent_id = 'root'
        folders_in_path =  self.dict_by_title(self.folders_in_folder(parent_id))
        self.debug('found folder in parent {}: {}'.format(parent_id,folders_in_path.keys()))

        if folder_name not in folders_in_path.keys():
            self.info('creating {}->{}'.format(parent_id,folder_name))
            if parent_id is None:
                fol = self.gdrive.CreateFile({'title': folder_name,
                                            "mimeType": "application/vnd.google-apps.folder"})
            else:
                fol = self.gdrive.CreateFile({'title': folder_name, 
                                        'parents': [ {"id": parent_id }],  
                                        "mimeType": "application/vnd.google-apps.folder"})            
            self.debug('uploading {}->{}'.format(parent_id,folder_name))                                        
            fol.Upload()
            self.add_cached_folder(fol,parent_id)

        else:
            self.debug('found folder {} in parent {}'.format(folder_name,parent_id))
            fol = folders_in_path[folder_name]
            self.add_cached_folder(fol,parent_id)
        
        return fol

    def sync_folder_contents_locally(self,parent_id,parent_folder_path=None):
        #Update Subdirectories - more efficient all at once
        if parent_id in self.reverse_folder_cache and parent_folder_path is None:
            parent_folder_path = self.reverse_folder_cache[parent_id]

        self.debug('updating subdirecories of {}'.format(parent_id))
        for sfol in self.folders_in_folder(parent_id):
            self.add_cached_folder(sfol,parent_id)

        for sfil in self.files_in_folder(parent_id):
            self.add_cached_file(sfil,parent_folder_path)                

    def ensure_g_path_get_id(self,gpath):
        '''walks these internal google paths ensuring everythign is created from root'''
        
        self.info('ensuring path {}'.format(gpath))
        parent_id = 'root'

        #avoid expensive calls
        if gpath in self.folder_cache:
            return self.folder_cache[gpath]
        elif gpath.startswith('/'):
            if gpath.replace('/','',1) in self.folder_cache:
                 return self.folder_cache[gpath.replace('/','',1)]


        for sub in gpath.split(os.sep):
            if sub != '' and sub != 'root':
                fol = self.get_or_create_folder(sub,parent_id)
                parent_id = fol['id']

        return fol['id'] #should be last!

    #Utility Methods
    def files_in_folder(self,folder_id='root'):
        if folder_id is not None:
            self.debug("searching {} for files".format(folder_id))
            for file in self.gdrive.ListFile({'q':"'{}' in parents and trashed=false and mimeType != 'application/vnd.google-apps.folder' and mimeType != 'application/vnd.google-apps.shortcut'".format(folder_id)}).GetList():
                yield file

    def folders_in_folder(self,folder_id='root'):
        if folder_id is not None:
            self.debug("searching {} for files".format(folder_id))
            #Folders
            for file in self.gdrive.ListFile({'q':"'{}' in parents and trashed=false and mimeType = 'application/vnd.google-apps.folder'".format(folder_id)}).GetList():
                yield file
            #Shortcuts
            for file in self.gdrive.ListFile({'q':"'{}' in parents and trashed=false and mimeType = 'application/vnd.google-apps.shortcut'".format(folder_id)}).GetList():
                yield file                

    def all_in_folder(self,folder_id='root'):
        if folder_id is not None:
            self.debug("searching {} for files".format(folder_id))
            for file in self.gdrive.ListFile({'q':"'{}' in parents and trashed=false".format(folder_id)}).GetList():
                yield file                

    def dict_by_title(self,items_list):
        return {it['title']:it for it in items_list}

    @property
    def folder_cache(self):
        return self._folder_ids

    @property
    def reverse_folder_cache(self):
        reverse_dict = {val:key for key,val in self.folder_cache.items()}
        return reverse_dict        