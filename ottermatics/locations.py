import os, shutil, sys
import re

CORE_FOLDERS = ['invoices','research','media','documents']
SOFTWARE_PROJECT = ['.creds','software']
ELECTRONICS_PROJECT = ['electronics','firmware']
DESIGN_PROJECT = ['design','render']

PROJECT_FOLDER_OPTIONS = {'core': CORE_FOLDERS,
                          'software': CORE_FOLDERS+SOFTWARE_PROJECT,
                          'electronics': CORE_FOLDERS+ELECTRONICS_PROJECT,
                          'design': CORE_FOLDERS+DESIGN_PROJECT,
                          'iot': CORE_FOLDERS+SOFTWARE_PROJECT+ELECTRONICS_PROJECT,
                          'hardware': CORE_FOLDERS+ELECTRONICS_PROJECT+DESIGN_PROJECT,
                          'all': CORE_FOLDERS+SOFTWARE_PROJECT+ELECTRONICS_PROJECT+DESIGN_PROJECT}


def _get_appdata_path():
    import ctypes
    from ctypes import wintypes, windll
    CSIDL_APPDATA = 26
    _SHGetFolderPath = windll.shell32.SHGetFolderPathW
    _SHGetFolderPath.argtypes = [wintypes.HWND,
                                 ctypes.c_int,
                                 wintypes.HANDLE,
                                 wintypes.DWORD,
                                 wintypes.LPCWSTR]
    path_buf = wintypes.create_unicode_buffer(wintypes.MAX_PATH)
    result = _SHGetFolderPath(0, CSIDL_APPDATA, 0, 0, path_buf)
    return path_buf.value

def dropbox_home():
    from platform import system
    import base64
    import os.path
    _system = system()
    if _system in ('Windows', 'cli'):
        host_db_path = os.path.join(_get_appdata_path(),
                                    'Dropbox',
                                    'host.db')
    elif _system in ('Linux', 'Darwin'):
        host_db_path = os.path.expanduser('~'
                                          '/.dropbox'
                                          '/host.db')
    else:
        raise RuntimeError('Unknown system={}'
                           .format(_system))
    if not os.path.exists(host_db_path):
        raise RuntimeError("Config path={} doesn't exists"
                           .format(host_db_path))
    with open(host_db_path, 'r') as f:
        data = f.read().split()
    return str(base64.b64decode(data[1]),encoding='utf-8')

def ottermatics_dropbox():
    company_dropbox_home = os.path.split(dropbox_home())[0]
    return company_dropbox_home

def user_home():
    return dropbox_home()


def ottermatics_folder():
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_dropbox(),'Ottermatics')))

def ottermatics_projects():
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_dropbox(),'Projects')))

def ottermatics_project(project_name):
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_projects(),project_name.lower())))

def creds_folder():
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_folder(),'creds')))

def google_api_token():
    credfile = 'client_secret_456980504049-d52oq3rod7cfntg97eje5ui451j8dhs1.apps.googleusercontent.com.json'
    return str(os.path.abspath(os.path.join(os.sep,creds_folder(),credfile)))


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

    print('Checking For Folder `{}` In {}'.format(args.project_name,ottermatics_projects()))

    ott_projects = os.listdir(ottermatics_projects())
    lower_projects = [folder.lower() for folder in ott_projects]

    if args.project_name.lower() in lower_projects:
        print('Found Project Folder: {}'.format(args.project_name))

        index = lower_projects.index(args.project_name.lower())
        actual_folder = ott_projects[index]

        project_folder = os.path.join(ottermatics_projects(),actual_folder)

        folder_contents = os.listdir(project_folder)
        folder_contents_lower = map(lambda s: s.lower(), folder_contents)

        for folder_stub in PROJECT_FOLDER_OPTIONS[ args.type ]:
            
            if folder_stub in folder_contents_lower:
                print('Found Content Folder: {}'.format(folder_stub))
                continue    
            else:
                print('Making Content Folder: {}'.format(folder_stub))
                os.mkdir(os.path.join(project_folder,folder_stub))

    else: #Is A Fresh Go
        print('Making Project Folder: {}'.format(args.project_name))
        project_folder = os.path.join(ottermatics_projects(),args.project_name.lower())
        os.mkdir( project_folder )

        for folder_stub in PROJECT_FOLDER_OPTIONS[ args.type ]:
            print('Making Content Folder: {}'.format(folder_stub))
            os.mkdir( os.path.join(project_folder, folder_stub))

if __name__ == '__main__':
    main_cli()
