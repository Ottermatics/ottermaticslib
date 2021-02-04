import os, shutil, sys
import re
import pathlib
from platform import system,uname
import logging


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


def in_wsl() -> bool:
    """
    WSL is thought to be the only common Linux kernel with Microsoft in the name, per Microsoft:

    https://github.com/microsoft/WSL/issues/4071#issuecomment-496715404
    """

    return  system() in ('Linux', 'Darwin') and 'Microsoft' in uname().release

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

def ottermatics_dropbox():
    company_dropbox_home = dropbox_home()
    return company_dropbox_home

def user_home():
    return dropbox_home()

def ottermatics_folder():
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_dropbox(),'Ottermatics')))

def ottermatics_projects():
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_dropbox(),'Projects')))

def ottermatics_project(project_name):
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_projects(),project_name)))

def creds_folder():
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_folder(),'.creds')))

def ottermatics_clients():
    return list([ path for path in os.listdir( ottermatics_projects() ) \
                if os.path.isdir(os.path.join(ottermatics_projects(),path)) ] )

def google_api_token():
    credfile = 'client_secrets.json'
    return str(os.path.abspath(os.path.join(os.sep,creds_folder(),credfile)))

def in_client_dir():
    '''Checks if the current path is in the projects folder'''
    parent_path = os.path.abspath( ottermatics_projects() )
    child_path = os.path.abspath( current_diectory() )
    return os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path])

def client_dir_name():
    '''Finds subdirectory name in projects folder if its in a client folder
    :returns: the client folder name in projects, otherwise none'''
    if not in_client_dir():
        return None
    current_rel_path  = os.path.relpath(current_diectory(), ottermatics_projects())
    return pathlib.Path(current_rel_path).parts[0]

def bool_from_env(bool_env_canidate):
    if bool_env_canidate.lower() in ('yes','true','y','1'):
        return True
    if bool_env_canidate.lower() in ('no','false','n','0'):
        return False
    return None

def load_from_env(creds_path='./.creds/',env_file='env.sh'):
    '''extracts export statements from bash file and aplies them to the python env'''
    creds_path = os.path.join(creds_path,env_file)
    log.info("checking {} for creds".format(creds_path))
    if not 'OTTER_CREDS_SET' in os.environ or not bool_from_env(os.environ['OTTER_CREDS_SET']):        
        if os.path.exists(creds_path):
            log.info('creds found')
            with open(creds_path,'r') as fp:
                txt = fp.read()

            lines = txt.split('\n')
            for line in lines:
                if line.startswith('export'):
                    key,val = line.replace('export','').split('=')
                    log.info('setting {}'.format(key))
                    os.environ[key.strip()]=val
        
        os.environ['OTTER_CREDS_SET'] = 'yes'
    else:
        log.info('credientials already set') 

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
                        choices=PROJECT_FOLDER_OPTIONS.keys(),
                        help='comma or space delimited list of components to add')
    #parser.add_argument('--capitalize',action='store_true')

    args = parser.parse_args()

    log.info('Checking For Folder `{}` In {}'.format(args.project_name,ottermatics_projects()))

    ott_projects = os.listdir(ottermatics_projects())
    lower_projects = [folder.lower() for folder in ott_projects]

    if args.project_name.lower() in lower_projects:
        log.info('Found Project Folder: {}'.format(args.project_name))

        index = lower_projects.index(args.project_name.lower())
        actual_folder = ott_projects[index]

        project_folder = os.path.join(ottermatics_projects(),actual_folder)

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
        project_folder = os.path.join(ottermatics_projects(),args.project_name.lower())
        os.mkdir( project_folder )

        for folder_stub in PROJECT_FOLDER_OPTIONS[ args.type ]:
            log.info('Making Content Folder: {}'.format(folder_stub))
            os.mkdir( os.path.join(project_folder, folder_stub))

if __name__ == '__main__':
    main_cli()
