import os, shutil, sys
import re
import pathlib
from platform import system,uname
from ottermatics.logging import LoggingMixin
import logging
import tempfile

'''
This locations system is designed to work with dropbox, there are workaroudn for linux
by creating these in your home directory which will then point to the correct location
.dropbox_home: link
.creds_home: link

Do not use global variables set in __init__.py in this file, locations is key to discovering those
'''

log = logging.getLogger('otterlib-locations')

CORE_FOLDERS = ['invoices','research','media','documents','reports']
SOFTWARE_PROJECT = ['.creds','software']
ELECTRONICS_PROJECT = ['electronics','firmware']
DESIGN_PROJECT = ['design','render']
ANALYSIS_PROJECT = ['analysis']

PROJECT_FOLDER_OPTIONS = {'core': CORE_FOLDERS,
                          'software': CORE_FOLDERS+SOFTWARE_PROJECT,
                          'electronics': CORE_FOLDERS+ELECTRONICS_PROJECT+ANALYSIS_PROJECT,
                          'design': CORE_FOLDERS+DESIGN_PROJECT,
                          'engineering': CORE_FOLDERS+DESIGN_PROJECT+ANALYSIS_PROJECT,
                          'iot': CORE_FOLDERS+SOFTWARE_PROJECT+ELECTRONICS_PROJECT,
                          'hardware': CORE_FOLDERS+ELECTRONICS_PROJECT+DESIGN_PROJECT+ANALYSIS_PROJECT,
                          'analysis': CORE_FOLDERS+ANALYSIS_PROJECT,
                          'all': CORE_FOLDERS+SOFTWARE_PROJECT+ELECTRONICS_PROJECT+DESIGN_PROJECT+ANALYSIS_PROJECT}

SERIVCE_CREDS_FILE = 'ottermaticsgdocs_serviceid.json'

#TODO: Map Client Drives (or others) as graphs in networkX, then we can compare different drive versions ect.
#TODO: Create A Context Class To Expose drive info (dropbox, userenv, client, remote, ect)

def in_wsl() -> bool:
    """
    WSL is thought to be the only common Linux kernel with Microsoft in the name, per Microsoft:

    https://github.com/microsoft/WSL/issues/4071#issuecomment-496715404
    """

    return  system().lower() in ('linux', 'darwin') and 'microsoft' in uname().release.lower()

def wsl_home():
    if in_wsl():
        stream = os.popen('wslpath "$(wslvar USERPROFILE)"')
        out = stream.read()
        return str(out.strip())
    return None

def _get_appdata_path():
    import ctypes
    from ctypes import wintypes, windll, create_unicode_buffer
    CSIDL_APPDATA = 26
    _SHGetFolderPath = windll.shell32.SHGetFolderPathW
    _SHGetFolderPath.argtypes = [wintypes.HWND,
                                 ctypes.c_int,
                                 wintypes.HANDLE,
                                 wintypes.DWORD,
                                 wintypes.LPCWSTR]
    path_buf = create_unicode_buffer(wintypes.MAX_PATH)
    result = _SHGetFolderPath(0, CSIDL_APPDATA, 0, 0, path_buf)
    return path_buf.value

def current_diectory():
    return os.path.realpath(os.curdir)

def dropbox_home():
    
    import base64
    import os.path
    _system = system()
    if '.dropbox_home' in os.listdir( os.path.expanduser('~')):
        return os.path.realpath(os.path.expanduser(os.path.join('~','.dropbox_home')))

    elif _system in ('Windows', 'cli'):
        host_db_path = os.path.join(_get_appdata_path(),
                                    'Ottermatics Dropbox')
        return host_db_path
    elif _system in ('Linux', 'Darwin'):
        host_db_path = os.path.expanduser('~'
                                          '/.dropbox'
                                          '/host.db')
    else:
        raise RuntimeError('Unknown system={}'
                           .format( _system ))
    if not os.path.exists(host_db_path):
        raise RuntimeError("Config path={} doesn't exists"
                           .format(host_db_path))
    with open(host_db_path, 'r') as f:
        data = f.read().split()
    return os.path.split(str(base64.b64decode(data[1]),encoding='utf-8'))[0]

def ottermatics_dropbox(skip_wsl = False):
    if not skip_wsl and in_wsl():
        return os.path.join(tempfile.gettempdir(),'otterlib')
    company_dropbox_home = dropbox_home()
    return company_dropbox_home

def user_home():
    return dropbox_home()

def ottermatics_folder(skip_wsl =False):
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_dropbox(skip_wsl =skip_wsl),'Ottermatics')))

def ottermatics_projects(skip_wsl =False):
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_dropbox(skip_wsl =skip_wsl),'Projects')))

def ottermatics_project(project_name,skip_wsl = False):
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_projects(skip_wsl =skip_wsl),project_name)))

def in_dropbox_dir(input_path=None):
    '''Checks if the current path is in the projects folder'''
    if input_path is None:
        input_path = current_diectory()     

    parent_path = os.path.abspath( ottermatics_dropbox(skip_wsl =True) )
    child_path = os.path.abspath( input_path )
    return os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path])    

def in_otter_dir(input_path=None):
    '''Checks if the current path is in the projects folder'''
    if input_path is None:
        input_path = current_diectory() 

    parent_path = os.path.abspath( ottermatics_folder(skip_wsl =True) )
    child_path = os.path.abspath( input_path )
    return os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path])    

def creds_folder():
    if '.creds_home' in os.listdir( os.path.expanduser('~')):
        return os.path.realpath(os.path.expanduser(os.path.join('~','.creds_home')))    
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_folder(skip_wsl =True),'.creds')))

def ottermatics_clients():
    return list([ path for path in os.listdir( ottermatics_projects(skip_wsl =True) ) \
                if os.path.isdir(os.path.join(ottermatics_projects(skip_wsl =True),path)) ] )

def google_api_token(): #Depricated now we use service accounts
    credfile = SERIVCE_CREDS_FILE
    return str(os.path.abspath(os.path.join(os.sep,creds_folder(),credfile)))

def in_client_dir(input_path=None):
    '''Checks if the current path is in the projects folder'''
    if input_path is None:
        input_path = current_diectory()

    parent_path = os.path.abspath( ottermatics_projects(skip_wsl =True) )
    child_path = os.path.abspath( input_path )
    return os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path])

def client_dir_name():
    '''Finds subdirectory name in projects folder if its in a client folder
    :returns: the client folder name in projects, otherwise none'''
    if not in_client_dir():
        return None
    current_rel_path  = os.path.relpath(current_diectory(), ottermatics_projects(skip_wsl =True))
    return pathlib.Path(current_rel_path).parts[0]

def client_path(skip_wsl=True):
    if not in_client_dir():
        return None
    return ottermatics_project( client_dir_name(),skip_wsl=skip_wsl)

def main_cli():
    '''
    Creates a Folder System With A Project Name In The Ottermatics Drobbox folder
    Additive in adding folders, defaults to lower case, but will check any case for existence
    '''
    import argparse

    parser = argparse.ArgumentParser( 'Initialize A Project')
    parser.add_argument('project_name')
    parser.add_argument('type',
                        type=str,
                        choices=list(PROJECT_FOLDER_OPTIONS.keys()),
                        help='comma or space delimited list of components to add')
    #parser.add_argument('--capitalize',action='store_true')

    args = parser.parse_args()

    #TODO: raise error or handle in WSL (we dont want to use the /mnt/c/ filesystem due to poor performance)

    log.info(f'Checking For Folder `{args.project_name}` In {ottermatics_projects(skip_wsl =True)}')

    ott_projects = os.listdir(ottermatics_projects(skip_wsl =True))
    lower_projects = [folder.lower() for folder in ott_projects]

    if args.project_name.lower() in lower_projects:
        log.info('Found Project Folder: {}'.format(args.project_name))

        index = lower_projects.index(args.project_name.lower())
        actual_folder = ott_projects[index]

        project_folder = os.path.join(ottermatics_projects(skip_wsl =True),actual_folder)

        folder_contents = os.listdir(project_folder)
        folder_contents_lower = map(lambda s: s.lower(), folder_contents)

        for folder_stub in PROJECT_FOLDER_OPTIONS[ args.type ]:
            
            if folder_stub in folder_contents_lower:
                log.info('Found Content Folder: {}'.format(folder_stub))
                continue    
            else:
                log.info('Making Content Folder: {}'.format(folder_stub))
                folder_path = os.path.join(project_folder,folder_stub)
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

    else: #Is A Fresh Go
        log.info('Making Project Folder: {}'.format(args.project_name))
        project_folder = os.path.join(ottermatics_projects(skip_wsl =True),args.project_name.lower())
        os.mkdir( project_folder )

        for folder_stub in PROJECT_FOLDER_OPTIONS[ args.type ]:
            log.info('Making Content Folder: {}'.format(folder_stub))
            os.mkdir( os.path.join(project_folder, folder_stub))


if __name__ == '__main__':
    main_cli()
