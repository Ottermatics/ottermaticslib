import os, sys
import ottermatics
from ottermatics.common import *
from ottermatics.locations import google_api_token, creds_folder, ottermatics_clients, client_dir_name, ottermatics_project
from ottermatics.logging import LoggingMixin, logging
from ottermatics.patterns import Singleton, SingletonMeta, singleton_meta_object



import pygsheets
import pydrive2
import traceback
import pathlib
import googleapiclient

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import time
import json
import attr
import datetime
import random

log = logging.getLogger('otterlib-gdocs')

GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = google_api_token()

#Filed will be None until assigned an folder id
STANDARD_FOLDERS = {'ClientFolders':None,'InternalClientDocuments':None}
#We'll hook this up with functionality at the module level
global CLIENT_G_DRIVE,CLIENT_GDRIVE_SYNC,CLIENT_GMAIL,CLIENT_NAME


#TODO - RATE LIMIT ERROR Handiling
# from funcy import cached_property, retry, wrap_prop, wrap_with
#From other code on github
# def _gdrive_retry(func):
#     def should_retry(exc):
#         from pydrive2.files import ApiRequestError

#         if not isinstance(exc, ApiRequestError):
#             return False

#         error_code = exc.error.get("code", 0)
#         result = False
#         if 500 <= error_code < 600:
#             result = True

#         if error_code == 403:
#             result = exc.GetField("reason") in [
#                 "userRateLimitExceeded",
#                 "rateLimitExceeded",
#             ]
#         if result:
#             logger.debug(f"Retrying GDrive API call, error: {exc}.")

#         return result

#     # 16 tries, start at 0.5s, multiply by golden ratio, cap at 20s
#     return retry(
#         16,
#         timeout=lambda a: min(0.5 * 1.618 ** a, 20),
#         filter_errors=should_retry,
#     )(func

#@Singleton
class OtterDrive(LoggingMixin, metaclass=SingletonMeta):
    '''Authenticates To Your Google Drive To Edit Sheets and Your Drive Files

    OtterDrive is a singleton so insantiated: OtterDrive() and you can do this
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
    root_id = None

    max_sleep_time = 10.0
    min_sleep_time = 0.1
    _sleep_time = 0.25
    time_fuzz = 4.0 #base * ( 1+ rand(0,time_fuzz))

    protected_parent_ids = None
    protected_ids = None
    protected_filenames = None
    dry_run = False

    _client_folder_id = None
    _target_folder_id = None
    
    #TODO - Integrate optimized timing growth / decay based on rate limit issues (SP vs MP)
    #TODO - add a queue system, posssibly async or in thread to manage the robust syncing of google drive files. 

    def __init__(self,sync_root=None,dry_run=False):
        ''':param sync_root: the client path to sync to on command'''
        if sync_root is not None:
            self.sync_root = sync_root
        else:
            self.sync_root = CLIENT_G_DRIVE

        self.file_cache = {} #Key - file path: Val item id
        self.folder_parents = {} #Key - item id: Val - parent id
        self.file_parents = {} #Key - item id: Val - parent id
        self.meta_cache = {}
        self._folder_ids = {'/':'root',None:'root','root':'root','':'root'} #Should be lists when activated
        
        self.protected_parent_ids = set()
        self.protected_ids = set()
        self.protected_filenames = list( STANDARD_FOLDERS.keys() )
        self.dry_run = dry_run

        self.sleep(self._sleep_time + 10 * random.random())
        self.authoirze_google_integrations()

        if CLIENT_NAME is not None:
            self.client_name = CLIENT_NAME
        else:
            self.client_name = client_dir_name()

        self.initalize()

    #Initalization Methods
    def initalize(self):
        '''Initalize maps the google root and shared folders, adds protections, and find the sync target'''
        self.info('Initalizing Drive Root Directories')
        self.client_folder = ottermatics_project(self.client_name)

        self.set_root_folder_id()
        self.sync_folder_contents_locally('root','',recursive=True,ttl=1, protect=True)
        self.shared_drives
        

        #Will Fail Here If Not Property Setup (fingers crossed)
        self._client_folder_id = self.folder_cache['ClientFolders']
        self.sync_folder_contents_locally(self.client_folder_id, protect=True)
        
        self._target_folder_id = self.folder_cache[self.sync_path(self.client_folder)]
        self.sync_folder_contents_locally(self.target_folder_id,recursive=True,ttl=3)

        if self.is_shared_target:
            self.gsheets.teamDriveId = self.target_root_id

        self.info(f'Otterdrive Ready For Use:\nClient Folder: {self.client_folder_id}\nTarget Folder: {self.target_folder_id}\nSync Root: {self.sync_root}')




    def sync_folder_contents_locally(self,parent_id,parent_folder_path=None,recursive=False,ttl=1,protect=False):
        '''This function takes a parent id for a folder then caches everything to folder / file caches
        Recrusvie functionality with recursive=True and ttl > 0'''
        #Update Subdirectories - more efficient all at once
        if parent_id in self.reverse_folder_cache and parent_folder_path is None:
            parent_folder_path = self.reverse_folder_cache[parent_id]

        if protect: self.protected_parent_ids.add(parent_id)

        self.debug( f'updating subdirecories of {parent_id}' )
        for sfol in self.all_in_folder(parent_id):
            if self.is_folder(sfol):
                self.add_cached_folder(sfol,parent_id, protect=protect)
                if recursive: 
                    ttl -= 1
                    if ttl == 0:
                        self.sync_folder_contents_locally(sfol['id'],recursive=False,ttl=ttl,protect=protect)
                    elif ttl > 0:
                        self.sync_folder_contents_locally(sfol['id'],recursive=True,ttl=ttl,protect=protect)

            if self.is_folder(sfol):
                self.add_cached_file(sfol,parent_folder_path, protect=protect)                

    
    def cache_directory(self,folder_id):
        self.info(f"{folder_id} - caching directory ")
        return list(self.search_items(f"'{folder_id}' in parents and trashed=false",folder_id))
        

    def search_items(self,q='',parent_id=None,**kwargs):
        '''A wrapper for `ListFile` that manages exceptions'''
        if 'in_trash' in kwargs:
            q += ' and trashed=true' 
        else:
            if 'trashed' not in q:
                q += ' and trashed=false'

        self.debug(f'search: pid:{parent_id} q:{q}')
        success = False
        
        input_args = {  'q':q,
                        'supportsAllDrives': True, 
                        'includeItemsFromAllDrives': True}
        for key in input_args.keys():
            if key in kwargs:
                kwargs.pop(key)
        input_args.update(kwargs)

        filenameset = set()
        duplicates = {} #store folder_name: [duplicate1, duplicate2, ... duplicateN]
        output_files = {}
        try: #Make Request
            for file in self.gdrive.ListFile(input_args).GetList():
                ftitle = file['title']
                #Remove Duplicates keep the first!? makes sense!
                if ftitle in filenameset and not self.item_in_root(file):
                    if ftitle in duplicates:
                        duplicates[ftitle].append(file)
                    else:
                        duplicates[ftitle] = [file]
                    continue

                elif ftitle in filenameset:
                    if ftitle not in duplicates:
                        self.warning(f'Duplicate Item In Root {ftitle}')
                    if ftitle in duplicates:
                        duplicates[ftitle].append(file)
                    else:
                        duplicates[ftitle] = [file]                    
                    continue #We don't want to override first references

                else:
                    filenameset.add(ftitle)
                    self.cache_item(file,parent_id)
                    output_files[ftitle] = file
            
            if duplicates:
                for key, othermatches in duplicates.items():
                    output_version = [output_files[key]]
                    all_name_matches = output_version + othermatches
                    self.warning(f'Duplicates Found for parent={parent_id} title={key}')
                    
                    new_version = self.handle_duplicates(all_name_matches, parent_id)
                    output_files[key] = new_version

            self.sleep()
            for file in list(output_files.values()):
                yield file
        
        except googleapiclient.errors.HttpError as err:
            # If the error is a rate limit or connection error,
            # wait and try again.
            if err.resp.status in [403]:
                self.hit_rate_limit()
                for item in self.search_items(q,parent_id,**kwargs):
                    yield item
            else:
                self.error(e,'Google API Error')

        except pydrive2.files.ApiRequestError as err:
            # If the error is a rate limit or connection error,
            # wait and try again.
            if 'code' in err.error and int(err.error['code']) in [403]:
                self.hit_rate_limit()
                for item in self.search_items(q,parent_id,**kwargs):
                    yield item
            else:
                self.error(e,'PyDrive2 Error')

        except Exception as e:
            self.error(e)


    @property
    def is_syncable_to_client_gdrive(self):
        if self.client_folder is not None and self.sync_root is not None:
            return True
        return False

    def authoirze_google_integrations(self,retry=True,ttl=3):
        try:

            self.info('Authorizing...')
            
            self.gauth = GoogleAuth()
            self.sleep()
            self.gauth.LoadCredentialsFile(os.path.join(creds_folder(),"gdrive_ottercreds.txt"))
            self.sleep()

            if self.gauth.credentials is None:
                # Authenticate if they're not there
                self.gauth.LocalWebserverAuth()
            elif self.gauth.access_token_expired:
                # Refresh them if expired
                self.gauth.Refresh()
                self.sleep()
            else:
                # Initialize the saved creds
                self.gauth.Authorize()
                self.sleep()
            # Save the current credentials to a file
            self.gauth.SaveCredentialsFile(os.path.join(creds_folder(),"gdrive_ottercreds.txt"))
            self.sleep()

            #Do Sheets Authentication
            self.gsheets = pygsheets.authorize( client_secret=google_api_token(),\
                                                credentials_directory =creds_folder())
            self.gsheets.enableTeamDriveSupport = True
            self.sleep()

            self.gdrive = GoogleDrive(self.gauth)
            self.sleep()
            self.gdrive.http = self.gauth.Get_Http_Object() #Do in advance for share drives
            self.sleep()

            self.info('Ready!')
        
        except Exception as e:
            ttl -= 1
            if retry and ttl > 0:
                self.warning(f'authorize failed {str(e)}')
                self.authoirze_google_integrations( retry=True, ttl=ttl )
            elif retry and ttl == 0:
                self.warning(f'authorize failed last try {str(e)}')
                self.authoirze_google_integrations( retry=False, ttl=ttl-1 )
            else:
                self.error(e,'authorize failed!!!')

                

    def initalize_google_drive_root(self,parent_id='root'):
        #TODO: THis creates duplicates!!
        #BUG: This Creates duplicates!!
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
            
            parent_id = self.target_folder_id
            self.cache_directory(parent_id)

            if sub_path is not None and not pathlib.Path(sub_path).is_absolute():
                sub_path = os.path.join(self.client_folder, sub_path)

            for i, (dirpath, dirnames, dirfiles) in enumerate(os.walk(self.client_folder)): #Its important to start at top to cache references
                try:
                    if sub_path is None or os.path.commonpath([dirpath,sub_path]) == os.path.commonpath([sub_path]):
                        
                        #Handle File Sync Ignores
                        any_hidden = any([ pth.startswith('.') or pth.startswith('_') for pth in dirpath.split(os.sep)])
                        if any_hidden:
                            continue

                        if '.skip_gsync' in dirfiles or '.skip_gsync' in dirnames:
                            self.debug('skipping {}'.format(dirpath))
                            skipped_paths.append(dirpath)
                            continue

                        if any([os.path.commonpath([dirpath,spath]) == os.path.commonpath([spath]) \
                                                                                for spath in skipped_paths]):
                            self.debug('skipping {}'.format(dirpath))
                            continue
                        
                        #Sync Work
                        self.debug('syncing {}'.format(dirpath))
                        self.cache_directory(parent_id)

                        dirpath = os.path.realpath(dirpath)
                        gdrive_path = self.sync_path(dirpath)

                        if self.sync_root != gdrive_path: #we shoudl have already cached it (god willing)
                            if gdrive_path in self.folder_cache:
                                parent_id = self.folder_cache[gdrive_path]
                            else:
                                parent_id = self.ensure_g_path_get_id(gdrive_path)
                        
                            self.cache_directory(parent_id)

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
                                    self.upload_or_update_file(gdir_id, file_path = absfile)
                                else:
                                    self.debug('skipping {}'.format(file))
                            else:
                                if gfile in self.file_cache: self.debug('found existing {}'.format(gfile))
                    else:
                        if i%100 == 0:
                            sys.stdout.write('.')

                        
                except Exception as e:
                    #self.warning('Failure in gdrive sync: {}'.format(e))
                    self.error(e,'Error Syncing Path ')
        else:
            self.warning('not able to sync: client folder: {} sync root: {}'.format(self.client_folder,self.sync_root))

    def valid_sync_file(self,file):
        if file.startswith('.'):
            return False

        if file.startswith('__') and file.endswith('__'):
            return False

        return True

    def sync_path(self,path):
        '''Sync path likes absolute paths to google drive relative'''
        assert self.in_client_folder(path)

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
    def create_file(self,input_args,file_path=None,content=None):
        '''A wrapper for creating a folder with CreateFile, we check for rate limits ect'''

        if self.is_shared_target:
            
            if 'kind' not in input_args:
                input_args['kind'] = 'drive#fileLink'

            if 'teamDriveId' not in input_args:
                input_args['teamDriveId'] = self.target_root_id
        
        self.debug(f'creating file w/ args: {input_args}')

        file = self.gdrive.CreateFile(input_args)
        self.sleep()
        try:
            if not self.dry_run and any((file_path is not None, content is not None )):
                
                if content is not None:
                    gfile_path = file_path #Direct conversion, we have to input correctly
                    if isinstance( content, (str,unicode)):
                        self.info(f'creating file with content string {input_args} -> {gfile_path}')
                        file.SetContentString(content)
                        self.sleep()
                    else:
                        self.info(f'creating file with content bytes {input_args} -> {gfile_path}')
                        file.content = content
                    

                elif file_path is not None:
                    self.debug(f'creating file with args {input_args} -> {gfile_path}')
                    file.SetContentFile(file_path)
                    self.sleep()
                    file_path = os.path.realpath(file_path)
                    file_name = os.path.basename(file_path)
                    gfile_path = self.sync_path(file_path)    

                file.Upload(param={'supportsTeamDrives': True}) # Upload the file.
                self.sleep()
                file.FetchMetadata(fields='permissions,labels,mimeType')
                self.sleep()

                self.debug(f'uploaded {file_path}')
                self.add_cached_file(file,gfile_path)
                

        except googleapiclient.errors.HttpError as err:
            # If the error is a rate limit or connection error,
            # wait and try again.
            if err.resp.status in [403]:
                self.hit_rate_limit()
                return self.create_file(input_args,file_path,content)
            else:
                self.error(e,'Google API Error')

        except pydrive2.files.ApiRequestError as err:
            # If the error is a rate limit or connection error,
            # wait and try again.
            if 'code' in err.error and int(err.error['code']) in [403]:
                self.hit_rate_limit()
                return self.create_file(input_args,file_path,content)
            else:
                self.error(e,'Google API Error')

        except Exception as e:
            self.error(e,'Error Creating File')
        
        finally:
            if file.content is not None:
                file.content.close()
            if file.http is not None:
                for c in file.http.connections.values():
                    c.close()

        self.sleep()
        return file

    def upload_or_update_file(self,parent_id, file_path=None, content=None, file_id=None,**kwargs):
        '''
        You need file_path or content and (g)file_path
        :param file_path: the absolute file path directory, the g-path will be deteremined, use as file title with content vs g filepath when provided alone
        :param parent_id: this is the id of the parent directory, we check that
        :param override: bypass protecte_parent_id, and in client folder protections
        :param content: a string to create the file along with file name in file_path to create a file
        :param file_id: a direct reference to use when updating a file
        :return: a file meta object or None
        '''
        if 'override' in kwargs:
            override = kwargs['override']
        else:
            override = False
        
        if not parent_id in self.protected_parent_ids:
            
            if content is None: #file path route
                
                file_path = os.path.realpath(file_path)
                file_name = os.path.basename(file_path)
                gfile_path = self.sync_path(file_path)

                if file_id is not None: #Override takes precident
                    self.info( 'updating id: {} {}->{}'.format(file_id, parent_id,gfile_path) )

                    fil = self.create_file( {"id": file_id, 'parents': [{'id': parent_id }]} ,
                                            file_path=file_path )
                    return fil        

                elif gfile_path in self.file_cache:
                    self.info( 'updating {}->{}'.format(parent_id,gfile_path) )

                    fil = self.create_file( {"id": self.file_cache[gfile_path], 'parents': [{'id': parent_id }]} ,
                                            file_path=file_path )
                    return fil            

                else:
                    self.info( 'creating {}->{}'.format(file_path,file_path) )
                    fil = self.create_file( {"title":file_name,'parents': [{'id': parent_id }]} ,
                                            file_path=file_path )
                    return fil
                               
            else: #we use content
                assert file_path is not None or file_id is not None #gotta have a title or existing reference

                if file_id is not None:
                    
                    if file_id in self.reverse_file_cache:
                        file_path = self.reverse_file_cache[ file_id ]

                    self.info( 'updating w/ content id:{} par:{}'.format(file_id, parent_id) )
                    fil = self.create_file( {"id": file_id, 'parents': [{'id': parent_id }]} ,
                                            file_path = file_path, content=content ) #Add file_path for the caching portion
                    return fil        

                elif file_path is not None:
                    file_name = os.path.basename(file_path)
                    
                    self.info( 'creating w/ content {}->{}'.format(parent_id,file_path) )
                    fil = self.create_file( {"title": file_name, 'parents': [{'id': parent_id }]} ,
                                            file_path = file_path, content=content ) #Add file_path for the caching portion          
                    return fil
            
            self.warning(f'could not create {file_path} in parent {parent_id}')
        else:
            self.warning(f'could not create {file_path} in protected parent {parent_id}')
        
        return None
        
    def create_folder(self,input_args,upload=True):
        '''A wrapper for creating a folder with CreateFile, we check for rate limits ect'''
        self.debug(f'creating Folder with args {input_args}')

        if 'mimeType' not in input_args:
            input_args['mimeType': "application/vnd.google-apps.folder"]

        file = self.gdrive.CreateFile(input_args)
        self.sleep()

        try:
            if upload and not self.dry_run:
                file.Upload(param={'supportsTeamDrives': True}) # Upload the file.
                self.sleep()
                self.debug(f'uploaded {input_args}')
                file.FetchMetadata(fields='permissions,labels,mimeType')
                self.sleep()
        
        except googleapiclient.errors.HttpError as err:
            # If the error is a rate limit or connection error,
            # wait and try again.
            if err.resp.status in [403]:
                self.hit_rate_limit()
                return self.create_folder(input_args,upload)
            else:
                self.error(e,'Google API Error')
        except pydrive2.files.ApiRequestError as err:
            # If the error is a rate limit or connection error,
            # wait and try again.
            if 'code' in err.error and int(err.error['code']) in [403]:
                self.hit_rate_limit()
                return self.create_folder(input_args,upload)
            else:
                self.error(e,'Google API Error')

        except Exception as e:
            self.error(e,'Error Creating Folder')

        finally:
            if file.http is not None:
                for c in file.http.connections.values():
                    c.close()

        self.sleep()
        return file  

    def get_or_create_folder(self,folder_name,parent_id='root',**kwargs):
        '''Creates a folder in the parent folder if it doesn't already exist, otherwise return folder
        :param folder_name: the name of the folder to create
        :param parent_id: the id of the parent folder, if None will create in the parent directory
        :param override: bypass root, parent_id and folder name protections
        '''

        if parent_id is None: parent_id = 'root'

        if 'override' in kwargs:
            override = kwargs['override']
        else:
            override = False

        #TODO: Implement Reverse Lookup OF Paths In Parent - danger is might not be complete, perhaps use a temp cache??
        # if parent_id in self.reverse_folder_parents and all([idd in self.meta_cache  \
        #                             for idd in self.reverse_folder_parents[parent_id] ]):
        #     self.debug(f'found folder meta in cache {parent_id}')
        #     folders_in_path = {os.path.basename(self.reverse_folder_cache[idd]): self.meta_cache[idd]
        #                                         for idd in self.reverse_folder_parents[parent_id]}

        folders_in_path =  self.dict_by_title(self.folders_in_folder(parent_id))

        protect_name = not folder_name in self.protected_filenames
        in_root = parent_id == 'root'
        protect_parent_id = parent_id in self.protected_parent_ids

        self.debug('found folder in parent {}: {}'.format(parent_id,folders_in_path.keys()))

        if folder_name not in folders_in_path.keys(): #Create It
            self.info( f'creating {parent_id}->{folder_name}' )
            if not any([all([protect_name,in_root]),in_root,protect_parent_id]) or override:
                fol = self.create_folder({'title': folder_name, 
                                        'parents': [ {"id": parent_id }],  
                                        "mimeType": "application/vnd.google-apps.folder"})            
                self.add_cached_folder(fol,parent_id,sync=True)
                return fol
            else:
                self.warning('could not create folder in protected ')

        else: #Grab It
            self.debug('found folder {} in parent {}'.format(folder_name,parent_id))
            fol = folders_in_path[folder_name]
            self.add_cached_folder(fol,parent_id,sync=True)

        #Finally look it up if it exists already in our cache - important for protected stuff
        self.debug(f'looking up {folder_name} in {parent_id}')
        if parent_id in self.reverse_folder_parents:
            content_ids = self.reverse_folder_parents[parent_id]
            valid_folder_ids = [cid for cid in  content_ids if self.reverse_folder_cache[cid].endswith(folder_name)]
            if len(valid_folder_ids) > 1: self.warning('get/ create folder: more than one matching parent found, grab first')
            return self.meta_cache[valid_folder_ids[0]]

    #Duplicate Handiling
    def delete(self,fileMeta):
        if not self.is_protected(fileMeta):
            if not self.dry_run:
                try:
                    self.info(f'deleting duplicate: {fileMeta["title"]}')                
                    fileMeta.Trash()
                    self.sleep()
                
                except googleapiclient.errors.HttpError as err:
                    # If the error is a rate limit or connection error,
                    # wait and try again.
                    if err.resp.status in [403]:
                        self.hit_rate_limit()
                        return self.delete(fileMeta)
                    else:
                        self.error(e,'Google API Error')

                except pydrive2.files.ApiRequestError as err:
                    # If the error is a rate limit or connection error,
                    # wait and try again.
                    if 'code' in err.error and int(err.error['code']) in [403]:
                        self.hit_rate_limit()
                        return self.delete(fileMeta)
                    else:
                        self.error(e,'Google API Error')

                except Exception as e:
                    self.error(e)

                finally:
                    #TODO: Remove item from internal stores
                    pass
        else:
            self.warning(f'could not delete {fileMeta["title"]} ')

    def copy_file(self, origin_id, target_id = None, create_filename=None, parent_id = None):
        self.debug(f'copy file: {origin_id}, {target_id}, {create_filename}, {parent_id} ')

        try:
            assert not any((all( (target_id is None, create_filename is None)) , parent_id is None))
                
            origin_file = self.create_file({'id' : origin_id })
            
            if origin_file is not None and 'mimeType' in origin_file and origin_file['mimeType'] == 'text/plain':
                #We can version plain text!
                content = origin_file.GetContentString()
            else:
                return None
                #TODO: Somehow Version Non Plain Text Files
                #origin_file.FetchContent()
                #content = origin_file.content

            self.sleep()

            if target_id is not None: #We know origin and target, good to go!
                #We're just getting the reference!
                target_file = self.upload_or_update_file( parent_id, file_id = target_id, content= content)
                return target_file

            elif create_filename is not None:
                target_file = self.upload_or_update_file( parent_id , file_path  =create_filename , content= content )
                return target_file

        except Exception as e:
            self.error(e,f'Could not copy file  {origin_id}, {target_id}, {create_filename}, {parent_id}')

    def merge_folder(self, target_id, origin_id):
        '''Use to map origin to target'''
        
        self.debug(f'merge_folder: {origin_id}, {target_id}')

        target_files = self.cache_directory(target_id) #This will take care of duplicates in target folder via search
        other_files = self.cache_directory(other_id) #This will take care of duplicates in other folder via search
        target_titles = { tar['title']:tar['id'] for tar in target_files }

        for ofil in other_files:
            otitle = fil['title']
            oid = fil['id']
            if self.is_folder(ofil):
                fol_id = self.get_or_create_folder( otitle ,parent_id = target_id)
                self.merge_folder( fol_id, oid)

            if self.is_file(ofil):
                if otitle in target_titles: #update it
                    fil = self.copy_file( oid, target_id = target_titles[otitle], parent_id = target_id)

            else: #create it
                fil = self.copy_file( oid, create_filename = otitle,  parent_id = target_id)

    def handle_duplicates(self, duplicate_canidates, parent_id):
        #Check if actually duplicates
        self.debug(f'handle duplicates: {len(duplicate_canidates)}, {parent_id}')
        titles  = [ rt['title']  for rt in duplicate_canidates ]
        parent_ids  = [ set([par['id'] for par in rt['parents']])  for rt in duplicate_canidates ]
        common_parent_ids = list(set.intersection(*parent_ids))
        common_titles = list(set(titles))
        if len(common_titles) > 1 or len(common_parent_ids) > 1:
            self.warning(f'These dont seem to be duplicates {common_titles} {common_parent_ids}')
            return
        
        #Ok they're actually duplicates, lets handle the files and folders sort by date
        sortByDate = lambda fol: self.get_datetime(fol['createdDate'])
        folders = sorted(filter(self.is_folder, duplicate_canidates) , key = sortByDate)
        files = sorted(filter(self.is_file, duplicate_canidates) , key = sortByDate)
        
        #if len(files) > 1 and len(fodlers) > 1:
            #self.warning('we have duplicates for both files and folders!!! will only take care of folders')
        #if folders:
        #oldest_folder = folders[0]
        #duplicate_folders = folders[1:]        
        #for other_folder in duplicate_folders:
        #     self.merge_folder(oldest_folder['id'], other_folder['id'] )
        #     self.info(f'DELTED ME: {other_folder}')
        #return oldest_folder

        if files:
            if all([fil['mimeType'] == 'text/plain' for fil in files]): #We use oldest to merge
                oldest_file = files[0]
                duplicate_files = files[1:]
            else: #we will keep the newest file since we can't version it
                oldest_file = files[-1]
                duplicate_files = files[:-1]
            for other_file in duplicate_files:
                self.info(f'moving file:{other_file["createdDate"]} -> Target:{oldest_file["createdDate"]}')
                self.copy_file(other_file['id'], oldest_file['id'], parent_id=parent_id)
                self.delete(other_file)

            return oldest_file
    
    def ensure_g_path_get_id(self,gpath):
        '''walks these internal google paths ensuring everythign is created from root
        This one is good for a cold entry of a path'''
        #TODO: Don't create files if near root
        #TODO: Dont create files matching sync_root
        #TODO: Check sync_root for shared drives

        self.debug('ensuring path {}'.format(gpath))
        parent_id = self.target_folder_id

        #avoid expensive calls
        if gpath in self.folder_cache:
            return self.folder_cache[gpath]
        elif gpath.startswith('/'):
            if gpath.replace('/','',1) in self.folder_cache:
                 return self.folder_cache[gpath.replace('/','',1)]

        current_pos = ''
        for sub in gpath.split(os.sep):
            if sub != '' and sub != 'root':
                
                current_pos = os.path.join(current_pos,sub)

                if current_pos in self.folder_cache:
                    self.debug(f'ensure-path: grabing existing path {current_pos}' )
                    parent_id = self.folder_cache[current_pos]
                else:
                    self.debug(f'path doesnt exist, create it {current_pos}' )
                    fol = self.get_or_create_folder(sub,parent_id)
                    parent_id = fol['id']

                
                

        return parent_id

    #Meta Based Caching
    def cache_item(self,item_meta,parent_id=None,recursive=False,protect=False):
        
        if not self.is_drivefile(item_meta):
            return

        if parent_id is None: #Check in case we don't have a parent
            if 'parents' in item_meta and 'id' in item_meta['parents'][0]:
                parent_id = item_meta['parents'][0]['id']

        if protect:
            self.protected_ids.add(item_meta['id'])

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

    def add_cached_folder(self,folder_meta,parent_id,sync=False,protect=False):
        
        if not self.is_folder(folder_meta):
            return

        if protect:
            self.protected_ids.add(folder_meta['id'])

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

    def add_cached_file(self,file_meta,gfile_parent_path,protect=False):
        if not self.is_file(file_meta):
            return

        if protect:
            self.protected_ids.add(file_meta['id'])

        if gfile_parent_path is not None:
            
            self.meta_cache[file_meta['id']] = file_meta

            gfile_path = os.path.join(gfile_parent_path)
            self.msg('caching file {}'.format(gfile_path))

            self.file_cache[gfile_path] = file_meta['id']
            for parent in file_meta['parents']:
                self.file_parents[file_meta['id']] = parent['id']

    #Search Sync & Iteration Methods
    def set_root_folder_id(self):
        roots = list(self.search_items("'root' in parents",'root'))
        if roots:
            parent_id_sets  = [ set([par['id'] for par in rt['parents']])  for rt in roots ]
            root_id = list(set.intersection(*parent_id_sets))
            if len(root_id) > 1:
                self.warning('More than one common parent in root items, choosing first')
            self.root_id = root_id[0]
            self._folder_ids['root'] = self.root_id

    @property
    def sync_root(self):
        '''This is the relative drive path to google drive root of client ex. /ClientFolder/Client1 '''
        return self._sync_root

    @sync_root.setter
    def sync_root(self,sync_root):
        old_root = self._sync_root

        if sync_root is None and 'CLIENT_GDRIVE_PATH' in os.environ: #for env configuration
            self.debug('getting CLIENT_GDRIVE_PATH from environment')
            sync_root = os.environ['CLIENT_GDRIVE_PATH']

        if self._sync_root: 
            #If there was an existing value do reinitalization
            do_reinitalize = True
        else:
            do_reinitalize = False

        self._sync_root = sync_root

        if do_reinitalize:
            self.initalize()

    @property
    def client_folder_id(self):
        return self._client_folder_id

    @property
    def target_folder_id(self):
        return self._target_folder_id

    @property
    def shared_drives(self):
        meta,content = self.gdrive.http.request('https://www.googleapis.com/drive/v3/drives')
        drives = json.loads(content)['drives']
        output = {}
        for drive in drives:
            output[drive['name']] = drive['id']
            self._folder_ids[f"shared:{drive['name']}"] = drive['id']
            self.sync_folder_contents_locally(drive['id'],f"shared:{drive['name']}")
        return output

    @property
    def cached_target_folders(self):
        return list([ titl for titl,key in self.folder_cache.items() if key == self.target_folder_id])

    @property
    def is_shared_target(self):
        is_shared = any([titl.startswith('shared:') for titl in self.cached_target_folders])
        return is_shared

    @property
    def target_root_id(self):
        '''This identifies the target root id, useful for workign with shared folders and setting teamFolderId'''
        is_shared = self.is_shared_target
        if not is_shared:
            return None
        shared_folder = [titl for titl in self.cached_target_folders if titl.startswith('shared:')][0]
        shared_root = shared_folder.split('/')[0]
        return self.folder_cache[shared_root]

    def item_in_root(self,item_meta):
        parents = item_meta['parents']
        return any([par['isRoot'] for par in parents])

    def is_drivefile(self,metaFile):
        if metaFile is None:
            return False

        if 'id' not in metaFile:
            return False

        if 'kind' in metaFile and metaFile['kind'] == 'drive#file':
            return True
        return False

    def is_folder(self,metaFile):
        if self.is_drivefile(metaFile):
            if any([metaFile['mimeType'] == 'application/vnd.google-apps.folder',
                    metaFile['mimeType'] == 'application/vnd.google-apps.shortcut']):
                return True
        return False

    def is_file(self,metaFile):
        if self.is_drivefile(metaFile):
            if not any([metaFile['mimeType'] == 'application/vnd.google-apps.folder',
                    metaFile['mimeType'] == 'application/vnd.google-apps.shortcut']):
                return True
        return False    

    def is_protected(self,metaFile):
        protected_keys = ['starred','trashed','hidden']

        if self.is_drivefile(metaFile):
            labels = metaFile['labels']
            
            #Is Starred, Trashed, Restricted or Hidden
            if any([ labels[pk] for pk in protected_keys]):
                self.info(f'file has protected keys {labels}')
                return True
            #Protected Folder Name
            if metaFile['title'] in self.protected_filenames and self.is_folder(metaFile):
                self.info(f'folder has a protected filename {metaFile["title"]}')
                return True
            #File Id is protected
            elif metaFile['id'] in self.protected_ids:
                self.info(f'fil has a protected id {metaFile["id"]}')
                return True
            #Parent Id is protected
            elif any([pid['id'] in self.protected_parent_ids for pid in metaFile['parents']]):
                self.info(f'fil has a protected parent id')
                return True
            #Parent Is Root
            elif any([pid['isRoot'] for pid in metaFile['parents']]):
                self.info(f'fil is in root')
                return True
            
        #If these dont match its fair game!
        return False


    def in_client_folder(self,file_path):
        return os.path.commonpath([self.client_folder]) == os.path.commonpath([file_path, self.client_folder])      

    def files_in_folder(self,folder_id='root'):
        if folder_id is not None:
            self.debug(f"searching {folder_id} for files")
            for file in self.search_items(f"'{folder_id}' in parents and trashed=false", folder_id):
                if self.is_file(file):
                    yield file

    def folders_in_folder(self,folder_id='root'):
        if folder_id is not None:
            self.debug(f"searching {folder_id} for folders")
            #Folders
            for file in self.search_items(f"'{folder_id}' in parents and trashed=false", folder_id):
                if self.is_folder(file):
                    yield file


    def all_in_folder(self,folder_id='root'):
        if folder_id is not None:
            self.debug(f"searching {folder_id} for anything")
            for file in self.search_items(f"'{folder_id}' in parents and trashed=false", folder_id):
                yield file                

    def dict_by_title(self,items_list):
        return {it['title']:it for it in items_list}

    def get_datetime(self,dts:str):
        return datetime.datetime.strptime(dts, '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def folder_cache(self):
        return self._folder_ids

    @property
    def reverse_folder_cache(self):
        reverse_dict = {val:key for key,val in self.folder_cache.items()}
        return reverse_dict     

    @property
    def reverse_file_cache(self):
        reverse_dict = {val:key for key,val in self.file_cache.items()}
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


    @property
    def reverse_shared_folders(self):
        reverse_dict = {}
        for key,val in self.shared_drives.items():
            if val in reverse_dict:
                reverse_dict[val].append(key)
            else:
                reverse_dict[val] = [key]
        return reverse_dict     

    def hit_rate_limit(self,sleep_time=5):
        self.warning(f'Warning Hit Rate Limit, sleeping {sleep_time}s, then continuing')
        self.sleep(sleep_time)
        self._sleep_time = min(self._sleep_time * 2.0,self.max_sleep_time)
        self.sleep()
        

    def sleep(self,val=None):
        if isinstance(val,(float,int)):
            time.sleep(val)
        else:
            if self._sleep_time > self.min_sleep_time:
                self._sleep_time = self._sleep_time * 0.95
            else:
                self._sleep_time = max(self._sleep_time,self.min_sleep_time) * (1.0 + random.random()*self.time_fuzz)
            time.sleep(self._sleep_time)

    def __getstate__(self):
        '''Remove active connection objects, they are not picklable'''
        self.debug('removing unpiclable info')
        d = self.__dict__.copy()
        d['gsheets'] = None
        #d['engine'] = None
        d['_log'] = None
        d['gdrive'] = None
        d['gauth'] = None
        d['meta_cache'] = {}
        return d
    
    def __setstate__(self,d):
        '''We reconfigure on opening a pickle'''
        self.debug('seralizing')
        self.__dict__ = d

        self.authoirze_google_integrations()
        self.initalize()
