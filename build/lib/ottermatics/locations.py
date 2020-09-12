import os, shutil, sys

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

    return base64.b64decode(data[1])

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
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_projects(),project_name)))

def creds_folder():
    return str(os.path.abspath(os.path.join(os.sep,ottermatics_folder(),'creds')))

def google_api_token():
    credfile = 'client_secret_456980504049-d52oq3rod7cfntg97eje5ui451j8dhs1.apps.googleusercontent.com.json'
    return str(os.path.abspath(os.path.join(os.sep,creds_folder(),credfile)))
