from engforge.env_var import EnvVariable

FORGE_PATH_VAR = EnvVariable(
    "FORGE_REPORT_PATH", default=None, dontovrride=True
)


def client_path(alternate_path=None, **kw):
    path = FORGE_PATH_VAR.secret
    if path is None:
        if alternate_path is None:
            raise KeyError(
                f"no `FORGE_REPORT_PATH` set and no alternate path in client_path call "
            )
        return alternate_path
    else:
        return path
