import os, sys

from ottermatics.common import *
import ottermatics

from ottermatics.locations import google_api_token, creds_folder, ottermatics_clients, client_dir_name, ottermatics_project
from ottermatics.logging import LoggingMixin, logging
from ottermatics.patterns import Singleton, SingletonMeta, InputSingletonMeta

from concurrent.futures import ThreadPoolExecutor

from contextlib import contextmanager
import pygsheets
import pydrive2
import traceback
import pathlib
import googleapiclient

from pydrive2.auth import GoogleAuth, ServiceAccountCredentials
from pydrive2.drive import GoogleDrive
import time
import json
import attr
import datetime
import random


log = logging.getLogger('otterlib-gdocs')

GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = google_api_token()

#Filed will be None until assigned an folder id
STANDARD_FOLDERS = {'ClientFolders':None,'InternalClientDocuments':None, 'Ottermatics': None, 'Research': None}
#We'll hook this up with functionality at the module level
global CLIENT_G_DRIVE,CLIENT_GDRIVE_SYNC,CLIENT_GMAIL,CLIENT_NAME


class OtterDriveException(Exception): pass
class CredsNotFound(OtterDriveException): pass
class SharedDriveNotFound(OtterDriveException): pass
class SourceFolderNotFound(OtterDriveException): pass
class SyncPathNotFound(OtterDriveException): pass


#@Singleton
class OtterDrive(LoggingMixin, metaclass=InputSingletonMeta):
    '''Authenticates To Your Google Drive To Edit Sheets and Your Drive Files

    OtterDrive is a input based singleton so insantiated: OtterDrive(**args1) and you can do this
    anywhere in the same code and preserve cached information for other access to OtterDirve(**args1), however its not threadsafe (yet)

    OtterDrive is tested to work with shared drives and service accounts, portions of the code may work with user drives. These drives start with shared:DRIVENAME, and the default is shared:OTTERBOX.
    '''
    gauth = None
    gsheets = None
    gdrive = None

    #Store folder paths, so we only have to look up once, key is path, value is id
    #Storage Items
    _folder_ids = None #exposed as folder_cache
    _folder_contents = None
    protected_parent_ids = None
    protected_ids = None
    protected_filenames = None
    dry_run = False
    file_cache = None
    folder_parents = None
    file_parents = None
    meta_cache = None
    
    #Determined Items
    _creds_file  = None  
    _shared_drives = None    

    #Input items
    _sync_root = None
    _filepath_root = None
    _shared_drive = None

    #Derived From Input
    _sync_root_id = None #For Shared Drive Root
    _target_folder_id = None #For Sync Path

    max_sleep_time = 10.0
    min_sleep_time = 0.1
    _sleep_time = 0.2
    time_fuzz = 2.0 #base * ( 1+ rand(0,time_fuzz))

    #Default is most permissive
    explict_input_only = False
    guess_sync_path = True

    #Thread Pool
    use_threadpool = True
    _max_num_threads = 16

    #defaults
    default_shared_drive = 'shared:OTTERBOX'
    default_sync_path = 'ClientFolders'
    default_service_id = 'ottermatics@ottermaticsgdocs.iam.gserviceaccount.com'
    creds_file_name = 'ottermaticsgdocs_serviceid.json'

    filepath_inferred = False #sets on filepath_root.setter

    #TODO - add a thread queue system, posssibly async or in thread to manage the robust syncing of google drive files. 

    #TODO: Add shared_drive, sync_root, and filepath_root, and initalization options for different possibilities (default, service_acct, env_vars, input) with tests to ensure everythign is gucci
    #TODO: Remove concept of user 'root', we shouldn't be using our 'User' Drives

    def __init__(self,shared_drive = None, sync_root=None, filepath_root=None, creds_path = None, dry_run=False):
        '''
        :param shared_drive: share drive to use as a root directory
        :param sync_root: the relative directory from shared_drive root to the place where sync occurs
        :param filepath_root: the local machine file path to sync to google drive
        :param creds_path: can be none if filesys if configured, otherwise you can input a path to a json, or a folder to look for `ottermaticsgdocs_serviceid.json`'''
        
        self.info(f'creating OtterDrive with input share={shared_drive}, filepath={filepath_root}, sync_root={sync_root}')
        #Never replace these values if already configured (that would be redoing work)
        self.file_cache = {} #Key - file path: Val item id
        self.folder_parents = {} #Key - item id: Val - parent id
        self.file_parents = {} #Key - item id: Val - parent id
        self.meta_cache = {}
        self._folder_ids = {} #{'/':'root',None:'root','root':'root','':'root'} #Should be lists when activated
        
        self.protected_parent_ids = set()
        self.protected_ids = set()
        self.protected_filenames = list( STANDARD_FOLDERS.keys() )
        
        self.dry_run = dry_run

        #Handle Different Creds Input
        if creds_path:
            if creds_path.endswith('.json'):
                self.creds_file = creds_path
            else:
                self.creds_file = os.path.join( creds_path, self.creds_file_name)
        else:
            self.creds_file = os.path.join( creds_folder(), self.creds_file_name)
        
        #Authoirize yoself foo
        self.authoirze_google_integrations()

        #These use setters via properties so order is important
        #0) Get Share Drive Info
        self.update_shared_drives()
        #1) set and check share drive and check other souces for input
        self.shared_drive = shared_drive
        #2) set and check filepath infer from current directory if not input
        self.filepath_root = filepath_root
        #3) set and check sync path and infer from #1 / 2 if not input
        self.sync_root = sync_root

        #self.initalize()

    @property
    def initalized_values(self):
        return [self._sync_root_id,self._target_folder_id,self.sync_root,self.gauth,self.gsheets,self.gdrive]

    @property
    def is_initalized(self):
        tests = list([val is not None for val in  self.initalized_values])
        return all( tests )

    #Initalization Methods
    def initalize(self):
        '''Initalize maps the google root and shared folders, adds protections, and find the sync target'''
        self.info('Initalizing')
        
        do_protect = False # True if self.shared_drive == self.default_shared_drive else False
        self.sync_folder_contents_locally(self.sync_root_id, protect = do_protect)

        gpath = self.sync_path(self.filepath_root)
        if gpath in self.folder_cache:
            self._target_folder_id = self.folder_cache[gpath]
        elif gpath:
            self._target_folder_id = self.ensure_g_path_get_id(gpath)
        elif gpath == '':
            self._target_folder_id = self.ensure_g_path_get_id(self.shared_drive)

        self.sync_folder_contents_locally(self.target_folder_id,recursive=True,ttl=3)

        if self.is_shared_target:
            self.gsheets.drive.enable_team_drive(self.target_root_id)

        self.status_message('Otterdrive Ready For Use')

    def status_message(self,header=''):
        self.info(f'{header}:\nSharedDrive: {self.shared_drive}:{self.sync_root_id}\nTarget Folder:  {self.sync_root}:{self.target_folder_id}\nFilePath Conversion: {self.filepath_root}->{self.full_sync_root}')

    def authoirze_google_integrations(self,retry=True,ttl=3):
        try:
            self.sleep(self._sleep_time + 10 * random.random())
            self.info('Authorizing...')
            #Drive Authentication Using Service Account
            scope = ['https://www.googleapis.com/auth/drive']

            self.gauth = GoogleAuth()
            self.sleep()

            self.gauth.auth_method = 'service'            
            self.gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(self.creds_file, scope)

            self.gauth.Authorize()
            self.sleep()            

            #Do Sheets Authentication
            self.gsheets = pygsheets.authorize( service_account_file = self.creds_file )
            self.sleep()

            self.gdrive = GoogleDrive(self.gauth)
            self.sleep()

            self.gdrive.http = self.gauth.Get_Http_Object() #Do in advance for share drives
            self.sleep()

            self.info('Ready!')
        
        except Exception as e:
            ttl -= 1
            if retry and ttl > 0:
                self.hit_rate_limit()
                self.warning(f'authorize failed {str(e)}')
                self.authoirze_google_integrations( retry=True, ttl=ttl )
            elif retry and ttl == 0:
                self.hit_rate_limit()
                self.warning(f'authorize failed last try {str(e)}')
                self.authoirze_google_integrations( retry=False, ttl=ttl-1 )
            else:
                self.error(e,'authorize failed!!!')

    def reset(self):
        #Never replace these values if already configured (that would be redoing work)
        self.file_cache = {} #Key - file path: Val item id
        self.folder_parents = {} #Key - item id: Val - parent id
        self.file_parents = {} #Key - item id: Val - parent id
        self.meta_cache = {}
        self._folder_ids = {} #{'/':'root',None:'root','root':'root','':'root'} #Should be lists when activated
        
        self.protected_parent_ids = set()
        self.protected_ids = set()
        self.protected_filenames = list( STANDARD_FOLDERS.keys() )
        
        self.authoirze_google_integrations()
        self.update_shared_drives()
        self.initalize()

    @property
    def filepath_root(self):
        '''This is the location on the filepath where the system will begin its sync'''
        return self._filepath_root

    @filepath_root.setter
    def filepath_root(self,filepath_root):
        '''Check the filepath exists and set it'''
        if filepath_root is not None:
            if os.path.exists(filepath_root):
                self._filepath_root = os.path.realpath(filepath_root)
                self.filepath_inferred = False
                self.info(f'using existing filepath {self._filepath_root}')
            else:
                raise SourceFolderNotFound(f'could not find filepath {filepath_root}')
        elif not self.explict_input_only and self.guess_sync_path: #Infer It
            if in_client_dir():
                self._filepath_root = current_diectory()
                self.filepath_inferred = True
                self.info(f'using infered client path: {self._filepath_root}')
            elif self._shared_drive == self.default_shared_drive and in_dropbox_dir(): #OTTERSYNC
                self._filepath_root = current_diectory()
                self.filepath_inferred = True
                self.info(f'using infered ottermatics path: {self._filepath_root}')

    @property
    def shared_drive(self):
        '''This is the shared drive to be used, ensures starting with ex: shared:DRIVENAME'''
        return self._shared_drive

    @shared_drive.setter
    def shared_drive(self,shared_drive):
        '''Check the share drive exists and set it, a shared: will prepend the drive name'''
        if shared_drive is not None:
            just_drive = shared_drive.replace('shared:','')
            has_drives = self.shared_drives
            if just_drive in has_drives:
                self._shared_drive = f'shared:{just_drive}'
                self.info(f'using input shared drive: {self._shared_drive}')
            else:
                raise SharedDriveNotFound(f'Could not find {shared_drive}')
        elif not self.explict_input_only: #Inferance
            creds_info = self.credentials_info
            
            #1) Look for credentials in service account creds
            if 'shared_drive' in creds_info: 
                self._shared_drive = creds_info['shared_drive']
                self.info(f'using service account shared drive: {self._shared_drive}')
            
            #2) Look for enviornmental variable
            elif 'CLIENT_GDRIVE_PATH' in os.environ \
                and os.environ['CLIENT_GDRIVE_PATH'].startswith('shared:'):

                gpath = os.environ['CLIENT_GDRIVE_PATH']
                drive_canidate = None
                if '/' in gpath:
                    sep = '/'
                    drive_canidate = gpath.split(sep)[0]
                elif '\\' in gpath:
                    sep = '\\'
                    drive_canidate = gpath.split(sep)[0]
                else:
                    drive_canidate = gpath
                
                
                if drive_canidate is not None and drive_canidate: #should be a string not empty
                    drive_canidate = drive_canidate.replace('shared:','')
                    if drive_canidate in self.shared_drives:
                        self._shared_drive = f'shared:{drive_canidate}'
                        self.info(f'using env-var shared drive: {self._shared_drive}')
                    else:
                        raise SharedDriveNotFound(f'env var shared drive not found {gpath}->{drive_canidate}')
                else:
                    raise SharedDriveNotFound(f'env var for shared drive invalid {gpath}->{drive_canidate}')

            #3) Use Default
            else:
                self._shared_drive = self.default_shared_drive
                self.info(f'using default shared drive: {self._shared_drive}')

        #Set the root_id after everything has settled
        if self._shared_drive is not None:
            direct_name = self._shared_drive.replace('shared:','') 
            if direct_name in self.shared_drives:
                self._sync_root_id =  self.shared_drives[direct_name]
                return #otherwise we fail here

        raise SharedDriveNotFound(f'shared drive identification failed, our current assignment {self._shared_drive}')
    
    @property
    def sync_root(self):
        '''This is the relative drive path to google drive root of client ex. ClientFolder/Client1
        These values should not start with a sep `/` or a `shared:DRIVE` however they will be removed on setting'''
        return self._sync_root

    @property
    def full_sync_root(self):
        '''A conveience method to get the shared drive full path to use with folder_cache'''
        return os.path.join(self.shared_drive,self.sync_root)
        
    @sync_root.setter
    def sync_root(self,sync_root):
        
        if self._sync_root is not None: 
            do_reinitalize = True #We already initalize with previous input, do it again!
        else:
            do_reinitalize = False

        if sync_root is not None: #Use Input First
            #Do some formatting
            if sync_root.startswith('/'): sync_root = sync_root.replace('/','',1)
            if sync_root.startswith('\\'): sync_root =  sync_root.replace('\\','',1)
            if sync_root.startswith('shared:'): 
                if '/' in sync_root:
                    sync_root = os.path.join('/',sync_root.split('/')[1:])
                if '\\' in sync_root:
                    sync_root = os.path.join('\\',sync_root.split('\\')[1:])
            self._sync_root = sync_root
            self.info(f'using input sync root {self._sync_root}')         

        elif not self.explict_input_only:
            creds_info = self.credentials_info

            #1) Look for credentials in service account creds
            if 'shared_sync_path' in creds_info and creds_info['shared_sync_path']: 
                self._sync_root = creds_info['shared_sync_path']
                self.info(f'using service account sync path: {self._sync_root}')
            
            #2) Look for enviornmental variable
            elif 'CLIENT_GDRIVE_PATH' in os.environ and \
                os.environ['CLIENT_GDRIVE_PATH'] and \
                os.environ['CLIENT_GDRIVE_PATH'].startswith(self.shared_drive) and \
                os.environ['CLIENT_GDRIVE_PATH'] != self.shared_drive:

                gpath = os.environ['CLIENT_GDRIVE_PATH']
                drive_canidate = None
                if 'shared:' in gpath: #Git rid of first drive as thats the shared folder
                    if '/' in gpath:
                        sep = '/'
                        drive_canidate = gpath.split(sep)[1:]
                    elif '\\' in gpath:
                        sep = '\\'
                        drive_canidate = gpath.split(sep)[1:]
                    else:
                        drive_canidate = '' #root
                else:
                    drive_canidate = gpath #one could only assume this is correct
                
                self._sync_root = drive_canidate
                self.info(f'using env-var sync path: {self._sync_root}')

            #3) We do our best
            elif self.guess_sync_path:
                                    
                if not self.filepath_inferred: #We gave a filepath
                    guessfilepath = self.filepath_root
                else:
                    guessfilepath = current_diectory()

                if in_client_dir(guessfilepath): #in a client directory, so map to ClientFolders or Relative Dir
                    if self.shared_drive == self.default_shared_drive: 
                        #Map to ClientFolders/Client/matching/path...
                        relpath = os.path.relpath(guessfilepath,ottermatics_projects())
                        self._sync_root = os.path.join( self.default_sync_path, relpath)
                    else:
                        #Map to /matching/path... since client drive assumed root
                        self._sync_root =  os.path.relpath(guessfilepath,client_path())
                    self.info(f'client infered sync path from given filepath : {self._sync_root}')

                elif in_dropbox_dir(guessfilepath): #we're not in a client folder
                    if self.shared_drive == self.default_shared_drive: 
                        #Map to Dropbox/whatever/path
                        self._sync_root = os.path.relpath(guessfilepath,ottermatics_dropbox())
                    else:
                        #Its Root!
                        self._sync_root =  ''
                    self.info(f'client infered sync path from given filepath : {self._sync_root}')
                else:
                    self._sync_root = ''
                    self.info(f'client infered sync path from given filepath : {self._sync_root}')

        if self._sync_root is None: raise SyncPathNotFound(f'No sync_path found for {self._sync_root}')

        if do_reinitalize:
            self.initalize()

    @property
    def sync_root_id(self):
        return self._sync_root_id

    @property
    def target_folder_id(self):
        return self._target_folder_id

    def sync_path(self,path):
        '''Sync path likes absolute paths to google drive relative'''
        assert os.path.commonpath([self.filepath_root,path]) == os.path.commonpath([self.filepath_root])


        self.debug(f'finding relative path from {path} and {self.filepath_root}')
        rel_root = os.path.relpath(path,self.filepath_root)
        
        self.debug(f'getting gdrive path {self.full_sync_root} and {rel_root}')
        gdrive_root = os.path.join(self.full_sync_root,rel_root)

        #remove current directory /.
        if gdrive_root.endswith('{}.'.format(os.sep)):
            gdrive_root = os.path.split(gdrive_root)[0]

        if gdrive_root.startswith(os.sep):
            gdrive_root = gdrive_root.replace(os.sep,'',1)

        return gdrive_root

    @property
    def shared_drives(self):
        if self._shared_drives is None:
            self.update_shared_drives()
        return self._shared_drives

    def update_shared_drives(self):
        '''returns the shared drives associated with the service account'''
        meta,content = self.gdrive.http.request('https://www.googleapis.com/drive/v3/drives')
        drives = json.loads(content)['drives']
        output = {}
        for drive in drives:
            output[drive['name']] = drive['id']
            self._folder_ids[f"shared:{drive['name']}"] = drive['id']
            self.sync_folder_contents_locally(drive['id'],f"shared:{drive['name']}")
        self._shared_drives = output
        return output

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

    @contextmanager
    def rate_limit_manager(self,retry_function,*args,**kwargs):
        '''A context manager that handles authentication and rate limit errors'''

        if 'gfileMeta' in kwargs:
            file = kwargs.pop('gfileMeta')
        else:
            file = None

        try:

            yield self #We just want error handiling, context should be useful
                
        except googleapiclient.errors.HttpError as err:
            # If the error is a rate limit or connection error,
            # wait and try again.

            #Close connection to existing file
            if file is not None and file.http is not None:
                for c in file.http.connections.values():
                    c.close()

            if err.resp.status in [403]:
                self.hit_rate_limit()
                return retry_function(*args,**kwargs)
            else:
                self.error(err,'Google API Error')

        except pydrive2.files.ApiRequestError as err:
            # If the error is a rate limit or connection error,
            # wait and try again.
            
            #Close connection to existing file
            if file is not None and file.http is not None:
                for c in file.http.connections.values():
                    c.close()

            if 'code' in err.error and int(err.error['code']) in [403]:
                self.hit_rate_limit()
                return retry_function(*args,**kwargs)
            else:
                self.error(err,'Google API Error')

        except Exception as e:
            self.error(e)
            
        #Close connection to existing file
        finally:
            if file is not None and file.http is not None:
                for c in file.http.connections.values():
                    c.close()        

        return None
    
    def cache_directory(self,folder_id):
        '''quick way to get contents of directory'''
        #self.info(f"{folder_id} - caching directory ")
        #return list(self.search_items(f"'{folder_id}' in parents and trashed=false",folder_id))
        out = list([item for item in self.all_in_folder(folder_id)])
        return out
        
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

        with self.rate_limit_manager(self.search_items,q,parent_id,**kwargs):
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
                        self.debug(f'Duplicate Item In Root: {ftitle}')
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
                    if parent_id == self.target_folder_id:
                        self.debug(f'Duplicates Found for root {parent_id} title={key}')
                    else:
                        self.warning(f'Duplicates Found for parent={parent_id} title={key}')

                    
                    new_version = self.handle_duplicates(all_name_matches, parent_id)
                    output_files[key] = new_version

            self.sleep()
            for file in list(output_files.values()):
                yield file

    @property
    def is_syncable(self):
        if self.filepath_root is not None and self.sync_root is not None and self.shared_drive is not None:
            return True
        return False
             
    def initalize_google_drive_root(self):
        #BUG: This can creates duplicates
        self.info('initalizing ottermatics on gdrive')
        
        #Only do this when we're in our main shared drive
        assert self.shared_drive == self.default_shared_drive

        parent_id = self.sync_root_id
        for sfol,sref in STANDARD_FOLDERS.items():
            fol = self.get_or_create_folder(sfol,parent_id=parent_id)
            if 'client' in sfol.lower():
                for clientname in ottermatics_clients():
                    if clientname.lower() != 'archive':
                        self.get_or_create_folder(clientname,parent_id=fol['id'])

    #Sync Methods
    def generate_sync_filepath_pairs(self):
        '''Generates pairs of local and gdrive paths based on the filepath root
        
        This doesn't include paths that contain directories with `.` or `_` in the path, or any directories with
        a `.skip_gsync` file included in it.

        :return: tuple - (filepath, gdrivepath)
        '''
        skipped_paths = []

        parent_id = self.target_folder_id
        self.cache_directory(parent_id)

        for i, (dirpath, dirnames, dirfiles) in enumerate(os.walk(self.filepath_root)):

                #Handle File Sync Ignores
                any_hidden = any([ pth.startswith('.') or pth.startswith('_') for pth in dirpath.split(os.sep)])
                if any_hidden:
                    continue

                if '.skip_gsync' in dirfiles or '.skip_gsync' in dirnames:
                    #self.debug('skipping {}'.format(dirpath))
                    skipped_paths.append(dirpath)
                    continue

                if any([os.path.commonpath([dirpath,spath]) == os.path.commonpath([spath]) \
                                                                        for spath in skipped_paths]):
                    #self.debug('skipping {}'.format(dirpath))
                    continue

                for fil in dirfiles:
                    filpath = os.path.realpath(os.path.join(dirpath,fil))
                    gdrive_path = self.sync_path(filpath)
                    yield filpath , gdrive_path              

    def sync(self,force=False):
        #if self.is_syncable_to_client_gdrive:
        if self.is_syncable:
            self.info('syncing {} to client folder: {}'.format(self.filepath_root,self.sync_root))
            
            if not self.use_threadpool:
                for lpath, gpath in self.generate_sync_filepath_pairs():
                    
                    if gpath not in self.file_cache or force:                        
                        dirname =  os.path.dirname(gpath)
                        if dirname not in self.folder_cache:
                            parent_id = self.ensure_g_path_get_id(dirname) #Prevent duplicates!!!
                        else:
                            parent_id = self.folder_cache[dirname]
                        
                        self.cache_directory(parent_id)

                        self.sync_path_pair_single(lpath,gpath)

                    else:
                        if gpath in self.file_cache: self.debug( 'found existing {}'.format(gpath) )
            
            else:
                with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                    
                    for lpath, gpath in self.generate_sync_filepath_pairs():
                        dirname =  os.path.dirname(gpath)
                        if dirname not in self.folder_cache:
                            parent_id = self.ensure_g_path_get_id(dirname) #Prevent duplicates!!!
                        else:
                            parent_id = self.folder_cache[dirname]
                        
                        self.cache_directory(parent_id)

                        if gpath not in self.file_cache or force:
                            pool.submit(self.sync_path_pair_thread,lpath,gpath,parent_id)
                        else:
                            if gpath in self.file_cache: self.debug( 'found existing {}'.format(gpath) )

        else:
            self.status_message('Otter Drive Not Ready!!!')

    def sync_path_pair_single(self,filepath,syncpath):
        try:
            self.debug(f'syncing paths {filepath} -> {syncpath}')
            gdirpath = os.path.dirname( syncpath )
            par_id = self.ensure_g_path_get_id( gdirpath)
            self.upload_or_update_file(par_id,file_path=filepath)    
        
        except Exception as e:
            self.error(e,'Error Syncing Path ')    

    def sync_path_pair_thread(self,filepath,syncpath,parent_id=None):
        try:
            self.debug(f'syncing paths {filepath} -> {syncpath}')
            self.upload_or_update_file(parent_id,file_path=filepath)    
        
        except Exception as e:
            self.error(e,'Error Syncing Path ')             

    def valid_sync_file(self,file):
        if file.startswith('.'):
            return False

        if file.startswith('__') and file.endswith('__'):
            return False

        return True

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

        with self.rate_limit_manager(self.create_file,input_args,file_path,content):
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
                    file_path = os.path.realpath(file_path)
                    file_name = os.path.basename(file_path)
                    gfile_path = self.sync_path(file_path)    

                    self.info(f'creating file with args {input_args} -> {gfile_path}')
                    file.SetContentFile(file_path)
                    self.sleep()


                file.Upload(param={'supportsTeamDrives': True}) # Upload the file.
                self.sleep()
                file.FetchMetadata(fields='permissions,labels,mimeType')
                self.sleep()

                self.debug(f'uploaded {file_path}')
                self.add_cached_file(file,gfile_path)

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
                    self.debug( 'updating id: {} {}->{}'.format(file_id, parent_id,gfile_path) )

                    fil = self.create_file( {"id": file_id, 'parents': [{'id': parent_id }]} ,
                                            file_path=file_path )
                    return fil        

                elif gfile_path in self.file_cache:
                    self.debug( 'updating file {}->{}'.format(parent_id,gfile_path) )

                    fil = self.create_file( {"id": self.file_cache[gfile_path], 'parents': [{'id': parent_id }]} ,
                                            file_path=file_path )
                    return fil            

                else:
                    self.debug( 'creating file {}->{}'.format(file_path,file_path) )
                    fil = self.create_file( {"title":file_name,'parents': [{'id': parent_id }]} ,
                                            file_path=file_path )
                    return fil
                               
            else: #we use content
                assert file_path is not None or file_id is not None #gotta have a title or existing reference

                if file_id is not None:
                    
                    if file_id in self.reverse_file_cache:
                        file_path = self.reverse_file_cache[ file_id ]

                    self.debug( 'updating file w/ content id:{} par:{}'.format(file_id, parent_id) )
                    fil = self.create_file( {"id": file_id, 'parents': [{'id': parent_id }]} ,
                                            file_path = file_path, content=content ) #Add file_path for the caching portion
                    return fil        

                elif file_path is not None:
                    file_name = os.path.basename(file_path)
                    
                    self.debug( 'creating file w/ content {}->{}'.format(parent_id,file_path) )
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

        with self.rate_limit_manager(self.create_folder,input_args,upload,gfileMeta=file):
            if upload and not self.dry_run:
                file.Upload(param={'supportsTeamDrives': True}) # Upload the file.
                self.sleep()
                self.debug(f'uploaded {input_args}')
                file.FetchMetadata(fields='permissions,labels,mimeType')
                self.sleep()
        
        if file.http is not None:
            for c in file.http.connections.values():
                c.close()

        self.sleep()
        return file

    def get_or_create_folder(self,folder_name,parent_id=None,**kwargs):
        '''Creates a folder in the parent folder if it doesn't already exist, otherwise return folder
        :param folder_name: the name of the folder to create
        :param parent_id: the id of the parent folder, if None will create in the parent directory
        :param override: bypass root, parent_id and folder name protections
        '''

        if parent_id is None: parent_id = self.target_folder_id
        in_root = parent_id == self.target_folder_id

        if 'override' in kwargs:
            override = kwargs['override']
        else:
            override = False

        #FIXME: Implement Reverse Lookup OF Paths In Parent- danger is might be outdated, use a fast expire cache??
        # if parent_id in self.reverse_folder_parents and all([idd in self.meta_cache  \
        #                             for idd in self.reverse_folder_parents[parent_id] ]):
        #     self.debug(f'found folder meta in cache {parent_id}')
        #     folders_in_path = {os.path.basename(self.reverse_folder_cache[idd]): self.meta_cache[idd]
        #                                         for idd in self.reverse_folder_parents[parent_id]}
        folders_in_path =  self.dict_by_title(self.folders_in_folder(parent_id))

        protect_name = not folder_name in self.protected_filenames
        
        protect_parent_id = parent_id in self.protected_parent_ids

        self.debug('found folder in parent {}: {}'.format(parent_id,folders_in_path.keys()))

        if folder_name not in folders_in_path.keys(): #Create It
            
            self.debug( f'creating {parent_id}->{folder_name}' )

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
                with self.rate_limit_manager(self.delete,fileMeta,gfileMeta=fileMeta):
                    self.debug(f'deleting duplicate: {fileMeta["title"]}')                
                    fileMeta.Trash()
                    self.sleep()

                    if 'id' in fileMeta:
                        fid = fileMeta['id']
                        if fid in self._folder_ids:
                            self.debug('removing folder id')
                            self._folder_ids.pop(fid)
                            if fid in self.folder_parents:
                                self.debug('removing folder parent id')
                                self.folder_parents.pop(fid)                          
                        elif fid in self.file_cache:
                            self.debug('removing file id')
                            self.pop(fid)
                            if fid in self.file_parents:
                                self.debug('removing file parent id')
                                self.file_parents.pop(fid)
                        if fid in self.meta_cache:
                            item = self.meta_cache.pop(fid)
                            del item
        else:
            self.debug(f'could not delete protected {fileMeta["title"]} ')

    def copy_file(self, origin_id, target_id = None, create_filename=None, parent_id = None):
        self.debug(f'copy file: {origin_id}, {target_id}, {create_filename}, {parent_id} ')

        try:
            assert not any((all( (target_id is None, create_filename is None)) , parent_id is None))
                
            origin_file = self.create_file({'id' : origin_id })
            
            if origin_file is not None and 'mimeType' in origin_file and origin_file['mimeType'] == 'text/plain':
                #We can version plain text!
                content = origin_file.GetContentString()

                if origin_file is not None and origin_file.http is not None:
                    for c in origin_file.http.connections.values():
                        c.close()                
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
        '''NOT READY: Use to map origin to target'''
        
        self.debug(f'merge_folder: {origin_id}, {target_id}')

        target_files = self.cache_directory(target_id) #This will take care of duplicates in target folder via search
        other_files = self.cache_directory(origin_id) #This will take care of duplicates in other folder via search
        target_titles = { tar['title']:tar['id'] for tar in target_files }

        for ofil in other_files:
            otitle = ofil['title']
            oid = ofil['id']
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
        
        if len(files) > 1 and len(folders) > 1:
            self.warning('we have duplicates for both files and folders!!! will only take care of folders')

        if folders:
            oldest_folder = folders[0]
            duplicate_folders = folders[1:]        
            for other_folder in duplicate_folders:
                #self.merge_folder(oldest_folder['id'], other_folder['id'] )
                self.delete(other_folder)

                #Update References
                if oldest_folder['id'] not in self.reverse_folder_cache \
                    and other_folder['id'] in self.reverse_folder_cache:

                    folder_path = self.reverse_folder_cache[other_folder['id']]
                    self.folder_cache[folder_path] = oldest_folder['id']



        if files:
            if all([fil['mimeType'] == 'text/plain' for fil in files]): #We use oldest to merge
                oldest_file = files[0]
                duplicate_files = files[1:]
            else: #we will keep the newest file since we can't version it
                oldest_file = files[-1]
                duplicate_files = files[:-1]
            for other_file in duplicate_files:
                if not self.is_protected(other_file):
                    self.info( f'moving file:{other_file["createdDate"]} -> Target:{oldest_file["createdDate"]}' )
                    self.copy_file( other_file['id'], oldest_file['id'], parent_id=parent_id)
                    self.delete( other_file )

            #Update References
            if oldest_file['id'] in self.reverse_file_cache:
                file_path = self.reverse_file_cache[oldest_file['id']]
                self.file_cache[file_path] = oldest_file['id']

        #self.cache_directory(parent_id)

        return oldest_file
    
    def ensure_g_path_get_id(self,gpath):
        '''walks these internal google paths ensuring everythign is created from root
        This one is good for a cold entry of a path
        
        This method can create duplicates when used in parallel contexts'''

        self.info('ensuring path {}'.format(gpath))
        parent_id = self.target_folder_id

        #avoid expensive calls
        if gpath in self.folder_cache:
            parent_id = self.folder_cache[gpath]
            self.cache_directory(parent_id)
            return parent_id 

        elif gpath.startswith('/'):
            if gpath.replace('/','',1) in self.folder_cache:
                parent_id = self.folder_cache[gpath.replace('/','',1)]
                self.cache_directory(parent_id)
                return parent_id                  

        current_pos = ''
        for sub in gpath.split(os.sep):
            if sub != '' and sub != 'root':
                
                current_pos = os.path.join(current_pos,sub)
                self.info(f'ensure-path: {current_pos}')
                if current_pos in self.folder_cache:
                    parent_id = self.folder_cache[current_pos]
                    self.info(f'ensure-path: grabing existing path {current_pos}:{parent_id}' )
                    self.cache_directory(parent_id)    

                else:
                    self.info(f'path doesnt exist, create it {current_pos}' )
                    fol = self.get_or_create_folder(sub,parent_id)
                    parent_id = fol['id']
                    self.cache_directory(parent_id)

        return parent_id

    #Meta Based Caching
    def cache_item(self,item_meta,parent_id=None,recursive=False,protect=False):
        
        if not self.is_drivefile(item_meta):
            self.debug(f'not caching {item_meta["title"]} - not a drive file')
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
            self.warning(f'cannot cache meta {item_meta["id"]}, no parent_id {parent_id} cached')

    def add_cached_folder(self,folder_meta,parent_id,sync=False,protect=False):
        
        if not self.is_folder(folder_meta):
            return

        if protect:
            self.protected_ids.add(folder_meta['id'])

        if parent_id in self.reverse_folder_cache:
            
            self.meta_cache[folder_meta['id']] = folder_meta

            parent_path = self.reverse_folder_cache[parent_id]
            folder_path = os.path.join('',parent_path,folder_meta['title'])
            
            self.debug('caching folder meta {}'.format(folder_path))

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
            self.msg('caching file meta {}'.format(gfile_path))

            self.file_cache[gfile_path] = file_meta['id']
            for parent in file_meta['parents']:
                self.file_parents[file_meta['id']] = parent['id']

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

        if 'kind' in metaFile:
            
            if metaFile['kind'] == 'drive#file':
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
                self.debug(f'file has protected keys {labels}')
                return True
            #Protected Folder Name
            if metaFile['title'] in self.protected_filenames and self.is_folder(metaFile):
                self.debug(f'folder has a protected filename {metaFile["title"]}')
                return True
            #File Id is protected
            elif metaFile['id'] in self.protected_ids:
                self.debug(f'fil has a protected id {metaFile["id"]}')
                return True
            #Parent Id is protected
            elif any([pid['id'] in self.protected_parent_ids for pid in metaFile['parents']]):
                self.debug(f'fil has a protected parent id')
                return True
            #Parent Is Root
            elif any([pid['isRoot'] for pid in metaFile['parents']]):
                self.debug(f'fil is in root')
                return True
            
        #If these dont match its fair game!
        return False

    def in_client_folder(self,file_path):
        return os.path.commonpath([self.filepath_root]) == os.path.commonpath([file_path, self.filepath_root])      

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
        old_sleeptime = self._sleep_time
        self._sleep_time = min(self._sleep_time * 2.0**(2.0/self.num_threads),self.max_sleep_time)

        self.warning(f'Warning Hit Rate Limit, sleeping {sleep_time}s, change sleep {old_sleeptime} -> {self._sleep_time} then continuing')
        
        self.sleep(sleep_time)        
        self.sleep()
        
    def sleep(self,val=None):
        if isinstance(val,(float,int)):
            time.sleep(val)
        else:
            if self._sleep_time > self.min_sleep_time:
                self._sleep_time = self._sleep_time * (0.95**(1.0/self.num_threads)) #This normalizes decay
            else:
                self._sleep_time = max(self._sleep_time,self.min_sleep_time) * (1.0 + random.random()*self.time_fuzz)
            time.sleep(self._sleep_time)

    @property
    def num_threads(self):
        if self.use_threadpool:
            return self._max_num_threads
        return 1

    #Creds Methods
    @property
    def creds_file(self):
        return self._creds_file

    @creds_file.setter
    def creds_file(self,file_path):
        if os.path.exists(file_path):
            self._creds_file = file_path
        else:
            raise CredsNotFound(f'no credentials found in {file_path}')

    @property
    def credentials_info(self):
        with open(self._creds_file,'r') as fp:
            out = json.loads(fp.read())
            if 'private_key' in out:
                out.pop( 'private_key' ) #Security yay
            return out

    #Magic Methods (Pickling here)
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
