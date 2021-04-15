import os
import ottermatics
from ottermatics.common import *
from ottermatics.locations import google_api_token, creds_folder, ottermatics_clients, client_dir_name, ottermatics_project
from ottermatics.logging import LoggingMixin, logging
from ottermatics.patterns import Singleton

import pygsheets
import pydrive
import traceback

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import time
import json
import attr

log = logging.getLogger('otterlib-gdocs')

GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = google_api_token()

#Filed will be None until assigned an folder id
STANDARD_FOLDERS = {'ClientFolders':None,'InternalClientDocuments':None}
#We'll hook this up with functionality at the module level
global CLIENT_G_DRIVE,CLIENT_GDRIVE_SYNC,CLIENT_GMAIL,CLIENT_NAME




@Singleton
class OtterDrive(LoggingMixin):
    '''Authenticates To Your Google Drive To Edit Sheets and Your Drive Files

    OtterDrive is a singleton so insantiated: OtterDrive.instance() and you can do this
    anywhere in the same code and preserve cached information, however its not threadsafe (yet)

    You should connect share drives by linking a shortcut in to a folder in the top level of the share drive you want to upload to. Then just modify `sync_root` to match the path to that shortcut on google drive!
    '''    
    gauth = None
    gsheets = None
    gdrive = None

    #Store folder paths, so we only have to look up once, key is path, value is id
    _folder_ids = None
    _folder_contents = None
    _sync_root = ''
    _gdrive_root = None
    client_folder = None

    file_cache = None
    folder_parents = None
    file_parents = None
    meta_cache = None

    _sleep_time = 0.15
    
    #TODO - Integrate optimized timing growth / decay based on rate limit issues (SP vs MP)
    #TODO - add a queue system, posssibly async or in thread to manage the robust syncing of google drive files. 

    def __init__(self,sync_root=CLIENT_G_DRIVE):
        ''':param sync_root: the client path to sync to on command'''
        self.sync_root = sync_root
        self.file_cache = {} #Key - file path: Val item id
        self.folder_parents = {} #Key - item id: Val - parent id
        self.file_parents = {} #Key - item id: Val - parent id
        self.meta_cache = {} #Key - item id: Val item-meta
        self._folder_ids = {'/':'root',None:'root','root':'root','':'root'} #Should be lists when activated
        
        

        self.authoirze_google_integrations()

        if CLIENT_NAME is not None:
            self.client_name = CLIENT_NAME
        else:
            self.client_name = client_dir_name()

        self.client_folder = ottermatics_project(self.client_name)
        self.set_root_folder_id()

    def set_root_folder_id(self):
        roots = list(self.search_items("'root' in parents",'root'))
        if roots:
            self.root_id = roots[0]['parents'][0]['id']
            self._folder_ids['root'] = self.root_id        

    @property
    def sync_root(self):
        return self._sync_root

    @sync_root.setter
    def sync_root(self,sync_root):
        old_root = self._sync_root

        if sync_root is None and 'CLIENT_GDRIVE_PATH' in os.environ: #for env configuration
            self.debug('getting CLIENT_GDRIVE_PATH from environment')
            sync_root = os.environ['CLIENT_GDRIVE_PATH']

        self._sync_root = sync_root

    @property
    def shared_drives(self):
        meta,content = self.gdrive.http.request('https://www.googleapis.com/drive/v3/drives')
        drives = json.loads(content)['drives']
        output = {}
        for drive in drives:
            output[drive['name']] = drive['id']
            self._folder_ids[f"shared:{drive['name']}"] = drive['id']
            self.sync_folder_contents_locally(drive['id'],f"shared:{drive['name']}",recursive=True,ttl=1)
        return output

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
        self.gsheets.enableTeamDriveSupport = True

        self.gdrive = GoogleDrive(self.gauth)
        self.gdrive.http = self.gauth.Get_Http_Object() #Do in advance for share drives
        self.shared_drives

        self.info('Ready!')

    def initalize_google_drive(self,parent_id='root'):
        self.info('initalizing ottermatics on gdrive')
        for sfol,sref in STANDARD_FOLDERS.items():
            if sref is None:
                fol = self.get_or_create_folder(sfol,parent_id=parent_id)
                STANDARD_FOLDERS[sfol] = fol['id']
                for clientname in ottermatics_clients():
                    if clientname != 'Archive':
                        self.get_or_create_folder(clientname,parent_id=fol['id'])

    def sync_to_client_folder(self,force=False,sub_path=None):
        skipped_paths = []
        if self.is_syncable_to_client_gdrive:
            self.info('syncing {} to client folder: {}'.format(self.client_folder,self.sync_root))
            
            if self.sync_root not in self.folder_cache:
                parent_id = self.ensure_g_path_get_id(self.sync_root)
            for dirpath, dirnames, dirfiles in os.walk(self.client_folder):            
                try:
                    if sub_path is None or os.path.commonpath([dirpath,sub_path]) == os.path.commonpath([sub_path]):
                        any_hidden = any([ pth.startswith('.') or pth.startswith('_') for pth in dirpath.split(os.sep)])

                        if '.skip_gsync' in dirfiles or '.skip_gsync' in dirnames:
                            self.debug('skipping {}'.format(dirpath))
                            skipped_paths.append(dirpath)
                            continue

                        if any([os.path.commonpath([dirpath,spath]) == os.path.commonpath([spath]) \
                                                                                for spath in skipped_paths]):
                            self.debug('skipping {}'.format(dirpath))
                            continue


                        if not any_hidden:
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
                            
                            self.cache_directory(gdir_id)
                            
                            #Folders
                            for folder in dirnames:
                                absfold = os.path.join(dirpath,folder)
                                gfold = self.sync_path(absfold)
                                if gfold not in self.folder_cache:
                                    if self.valid_sync_file(folder):
                                        gfol = self.get_or_create_folder(folder,gdir_id)
                                    else:
                                        self.debug('skipping {}'.format(absfold))                                
                                else:
                                    self.debug('found existing {}'.format(absfold))
                                

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
                                    if gfile in self.file_cache: self.debug('found existing {}'.format(gfile))
                        else:
                            self.debug('skipping hidden dir {}'.format(dirpath))
                        
                except Exception as e:
                    #self.warning('Failure in gdrive sync: {}'.format(e))
                    self.error(e)
        else:
            self.warning('not able to sync: client folder: {} sync root: {}'.format(self.client_folder,self.sync_root))

    def valid_sync_file(self,file):
        if file.startswith('.'):
            return False
        if file.startswith('__') and file.endswith('__'):
            return False
        return True

    def sync_path(self,path):
        '''Sync path likes absolute paths'''
        assert os.path.commonpath([self.client_folder]) == os.path.commonpath([path, self.client_folder])

        self.debug(f'finding relative path from {path} and {self.client_folder}')
        rel_root = os.path.relpath(path,self.client_folder)
        self.debug(f'getting gdrive path {self.sync_root} and {rel_root}')
        gdrive_root = os.path.join(self.sync_root,rel_root)
        #remove current directory /.
        if gdrive_root.endswith('{}.'.format(os.sep)):
            gdrive_root = os.path.split(gdrive_root)[0]
        if gdrive_root.startswith(os.sep):
            gdrive_root = gdrive_root.replace(os.sep,'',1)
        return gdrive_root

    #User Methods
    def create_file(self,input_args,file_path):
        '''A wrapper for creating a folder with CreateFile, we check for rate limits ect'''
        
        file_path = os.path.realpath(file_path)
        file_name = os.path.basename(file_path)
        gfile_path = self.sync_path(file_path)        
        
        self.debug(f'creating file with args {input_args} -> {gfile_path}')

        file = self.gdrive.CreateFile(input_args)

        try:
            
            file.SetContentFile(file_path)
            file.Upload(param={'supportsTeamDrives': True}) # Upload the file.
            self.debug(f'uploaded {file_path}')
            self.add_cached_file(file,gfile_path)
            
  
        except Exception as e:
            self.error(e,'Error Creating File')

        finally:
            for c in file.http.connections.values():
                c.close()

        self.sleep()

    def create_folder(self,input_args):
        '''A wrapper for creating a folder with CreateFile, we check for rate limits ect'''
        self.debug(f'creating Folder with args {input_args}')
        
        file = self.gdrive.CreateFile(input_args)

        try:
            file.Upload(param={'supportsTeamDrives': True}) # Upload the file.
            self.debug(f'uploaded {input_args}')
            
        except Exception as e:
            self.error(e,'Error Creating Folder')

        finally:
            for c in file.http.connections.values():
                c.close()

        self.sleep()
        return file

    def upload_or_update_file(self,file_path,parent_id):
        '''As Described...
        :param file_path: the absolute file path directory, the g-path will be deteremined
        '''
        #Only sync from client folder
        assert os.path.commonpath([self.client_folder]) == os.path.commonpath([file_path, self.client_folder])
        #Check existing

        file_path = os.path.realpath(file_path)
        file_name = os.path.basename(file_path)
        gfile_path = self.sync_path(file_path)    

        if gfile_path in self.file_cache:
            self.info('updating {}->{}'.format(parent_id,file_path))
            self.create_file({"id": self.file_cache[gfile_path], 'parents': [{'id': parent_id }]},file_path)

        else:
            self.info('creating {}->{}'.format(parent_id,file_path))
            self.create_file({"title":file_name,'parents': [{'id': parent_id }]},file_path)

    def get_or_create_folder(self,folder_name,parent_id='root',**kwargs):
        '''Creates a folder in the parent folder if it doesn't already exist, otherwise return folder
        :param folder_name: the name of the folder to create
        :param parent_id: the id of the parent folder, if None will create in the parent directory
        '''

        if parent_id is None: parent_id = 'root'

        if parent_id in self.reverse_folder_parents and all([idd in self.meta_cache  \
                                    for idd in self.reverse_folder_parents[parent_id] ]):
            self.debug(f'found folder meta in cache {parent_id}')
            folders_in_path = {os.path.basename(self.reverse_folder_cache[idd]): self.meta_cache[idd]
                                                for idd in self.reverse_folder_parents[parent_id]}
        else:
            folders_in_path =  self.dict_by_title(self.folders_in_folder(parent_id))

        self.debug('found folder in parent {}: {}'.format(parent_id,folders_in_path.keys()))

        if folder_name not in folders_in_path.keys():
            self.info('creating {}->{}'.format(parent_id,folder_name))
            if parent_id is None:
                fol = self.create_folder({'title': folder_name,
                                            "mimeType": "application/vnd.google-apps.folder"})
            else:
                fol = self.create_folder({'title': folder_name, 
                                        'parents': [ {"id": parent_id }],  
                                        "mimeType": "application/vnd.google-apps.folder"})            
            self.add_cached_folder(fol,parent_id,True)

        else:
            self.debug('found folder {} in parent {}'.format(folder_name,parent_id))
            fol = folders_in_path[folder_name]
            self.add_cached_folder(fol,parent_id)
        
        return fol        

    #Meta Based Caching
    def cache_item(self,item_meta,parent_id=None,recursive=False):
        if parent_id is None: #Check in case we don't have a parent
            if 'parents' in item_meta and 'id' in item_meta['parents'][0]:
                parent_id = item_meta['parents'][0]['id']

        if parent_id in self.reverse_folder_cache:
            parent_path = self.reverse_folder_cache[parent_id]
            folder_path = os.path.join('',parent_path,item_meta['title'])  #path of new item!          
            if any([item_meta['mimeType'] == 'application/vnd.google-apps.folder',
                    item_meta['mimeType'] == 'application/vnd.google-apps.shortcut']):
                self.add_cached_folder(item_meta['id'],folder_path)
            else:
                self.add_cached_file(item_meta,folder_path)
        else:
            self.warning(f'cannot cache {item_meta["id"]}, no parent_id {parent_id} cached')

    def add_cached_folder(self,folder_meta,parent_id,sync=False):

        if parent_id in self.reverse_folder_cache:
            
            self.meta_cache[folder_meta['id']] = folder_meta

            parent_path = self.reverse_folder_cache[parent_id]
            folder_path = os.path.join('',parent_path,folder_meta['title'])
            
            self.debug('caching folder {}'.format(folder_path))

            if folder_meta['mimeType'] == 'application/vnd.google-apps.folder':
                self.folder_cache[folder_path] = folder_meta['id']
                self.folder_parents[folder_meta['id']] = parent_id
                if sync: self.sync_folder_contents_locally(folder_meta['id'],folder_path)

            elif folder_meta['mimeType'] == 'application/vnd.google-apps.shortcut':
                shortcut = self.gdrive.CreateFile({'id':folder_meta['id']})
                shortcut.FetchMetadata(fields='shortcutDetails')
                if 'shortcutDetails' in shortcut:
                    details = shortcut['shortcutDetails']
                    if 'targetId' in details and details['targetMimeType'] == 'application/vnd.google-apps.folder':
                        self.folder_cache[folder_path] = details['targetId']
                        self.folder_parents[details['targetId']] = parent_id
                        if sync: self.sync_folder_contents_locally(details['targetId'],folder_path)

    def add_cached_file(self,file_meta,gfile_parent_path):
        if gfile_parent_path is not None:
            self.meta_cache[file_meta['id']] = file_meta
            gfile_path = os.path.join(gfile_parent_path)
            self.debug('caching file {}'.format(gfile_path))

            self.file_cache[gfile_path] = file_meta['id']
            for parent in file_meta['parents']:
                self.file_parents[file_meta['id']] = parent['id']

    #Search Sync & Iteration Methods
    def sync_folder_contents_locally(self,parent_id,parent_folder_path=None,recursive=False,ttl=1):
        #Update Subdirectories - more efficient all at once
        if parent_id in self.reverse_folder_cache and parent_folder_path is None:
            parent_folder_path = self.reverse_folder_cache[parent_id]

        self.debug('updating subdirecories of {}'.format(parent_id))
        for sfol in self.all_in_folder(parent_id):
            if self.is_folder(sfol):
                self.add_cached_folder(sfol,parent_id)
                if recursive: 
                    ttl -= 1
                    if ttl == 0:
                        self.sync_folder_contents_locally(sfol['id'],recursive=False,ttl=ttl)
                    elif ttl > 0:
                        self.sync_folder_contents_locally(sfol['id'],recursive=True,ttl=ttl)

            if self.is_folder(sfol):
                self.add_cached_file(sfol,parent_folder_path)                

    def ensure_g_path_get_id(self,gpath):
        '''walks these internal google paths ensuring everythign is created from root
        This one is good for a cold entry of a path'''
        
        self.info('ensuring path {}'.format(gpath))
        parent_id = 'root'

        #avoid expensive calls
        if gpath in self.folder_cache:
            return self.folder_cache[gpath]
        elif gpath.startswith('/'):
            if gpath.replace('/','',1) in self.folder_cache:
                 return self.folder_cache[gpath.replace('/','',1)]

        current_pos = ''
        for sub in gpath.split(os.sep):
            if sub != '' and sub != 'root':
                if current_pos in self.folder_cache:
                    self.debug('ensure-path: grabing existing path {}'.format(current_pos) )
                    parent_id = self.folder_cache[current_pos]
                fol = self.get_or_create_folder(sub,parent_id)
                current_pos = os.path.join(current_pos,sub)
                parent_id = fol['id']

        return fol['id'] #should be last!

    
    def cache_directory(self,folder_id):
        self.info(f"{folder_id} - caching directory ")
        list(self.search_items(f"'{folder_id}' in parents and trashed=false",folder_id))
        return

    def search_items(self,q='',parent_id=None,**kwargs):
        '''A wrapper for `ListFile` that manages exceptions'''
        
        if 'in_trash' in kwargs:
            q += ' and trashed=true' 
        else:
            if 'trashed' not in q:
                q += ' and trashed=false'

        print(f'search: {q}')
        success = False
        
        input_args = {  'q':q,
                        'supportsAllDrives': True, 
                        'includeItemsFromAllDrives': True}
        for key in input_args.keys():
            if key in kwargs:
                kwargs.pop(key)
        input_args.update(kwargs)

        try: #Make Request
            for file in self.gdrive.ListFile(input_args).GetList():
                self.cache_item(file,parent_id)
                yield file
                
            self.sleep()

        except Exception as e:
            self.error(e)

    def _is_drivefile(self,metaFile):
        if 'kind' in metaFile and metaFile['kind'] == 'drive#file':
            return True
        return False

    def is_folder(self,metaFile):
        if self._is_drivefile(metaFile):
            if any([metaFile['mimeType'] == 'application/vnd.google-apps.folder',
                    metaFile['mimeType'] == 'application/vnd.google-apps.shortcut']):
                return True
        return False

    def is_file(self,metaFile):
        if self._is_drivefile(metaFile):
            if not any([metaFile['mimeType'] == 'application/vnd.google-apps.folder',
                    metaFile['mimeType'] == 'application/vnd.google-apps.shortcut']):
                return True
        return False    


    def files_in_folder(self,folder_id='root'):
        if folder_id is not None:
            self.debug(f"searching {folder_id} for files")
            for file in self.search_items(f"'{folder_id}' in parents and trashed=false"):
                if self.is_file(file):
                    yield file

    def folders_in_folder(self,folder_id='root'):
        if folder_id is not None:
            self.debug(f"searching {folder_id} for files")
            #Folders
            for file in self.search_items(f"'{folder_id}' in parents and trashed=false"):
                if self.is_folder(file):
                    yield file


    def all_in_folder(self,folder_id='root'):
        if folder_id is not None:
            self.debug(f"searching {folder_id} for files")
            for file in self.search_items(f"'{folder_id}' in parents and trashed=false"):
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

    @property
    def reverse_folder_parents(self):
        reverse_dict = {}
        for key,val in self.folder_parents.items():
            if val in reverse_dict:
                reverse_dict[val].append(key)
            else:
                reverse_dict[val] = [key]
        return reverse_dict    

    @property
    def reverse_file_parents(self):
        reverse_dict = {}
        for key,val in self.file_parents.items():
            if val in reverse_dict:
                reverse_dict[val].append(key)
            else:
                reverse_dict[val] = [key]
        return reverse_dict                  

    def sleep(self,val=None):
        if isinstance(val,(float,int)):
            time.sleep(val)
        else:
            time.sleep(self._sleep_time)