from posix import environ
from ottermatics.configuration import otterize, Configuration

import attr
import os
import pysecret  # For AWS Secrets
import subprocess
import itertools
import pathlib

import uuid
import json

# load_from_env('./.creds/','env.sh',set_env=False)

(
    CLIENT_G_DRIVE,
    CLIENT_GDRIVE_SYNC,
    CLIENT_GMAIL,
    CLIENT_NAME,
    SLACK_WEBHOOK_NOTIFICATION,
) = (None, None, None, None, None)

# TODO: Backwards Compatability, Here's Our Key Variables. What are our covered use cases? DB / SSH / AWS / SLACK / CLIENT / GOOGLE SERVICE ACCTS
default_secrets = {}

# Secret Handlers
_local = None
_env = None


PORT = 5432
HOST = "localhost"
USER = "postgres"
PASS = "dumbpass"
DB_NAME = "dumbdb"


# Special values
STAGE_ENVVAR = "DEV_ENV"
DEFAULT_GDRIVE = "shared:OTTERBOX"
SLACK_WEBHOOK_NOTIFICATION = None  # TOOD: Make a default slack webhook!

# Local Credentials


# TODO: Remove this and use aws copilot concepts
@otterize
class Secrets(Configuration, metaclass=InputSingletonMeta):
    """Secrets handles an explicit user manage input and output of secrets

    Security philosophy here embodies wrapping access of enviornmental variables, and keeping their source on the developers computer. Distribution can be achieved via a cloud provider.
    1) hear no evil: never interact with a external unauthorized sources, only limited input will return credentials.
    2) speak no evil: Never send or set credentials internally or externally other than an authorized source.
    3) see no evil: Only look in authorized sources (use enviornment for our local reference)

    """

    # FIXME: Implement parent class to client secrets

    credential_key = attr.ib()  # What to refer to this credential as
    credential_location = attr.ib()  # Where to find the credential file

    _aws_kms: pysecret.AWSSecret = (
        None  # a storage for the aws key managment object
    )

    public_up_level = True  # makes the public cred record one level up (creds are stored protected). Yes its hacky but we have deadlines

    def __on_init__(self):
        self.info("initalizing secrets...")
        self._aws_kms = pysecret.AWSSecret()
        self.stored_env_vars = set()

    @property
    def credential_dir(self):
        return os.path.dirname(self.credential_location)

    @property
    def public_record_dir(self):
        if self.public_up_level:
            return os.path.dirname(self.credential_dir)
        return self.credential_dir

    @property
    def public_credential_upload_record(self):
        """A file link to the public record of secrets we have for this aws kms cred key"""
        return os.path.join(
            self.public_record_dir,
            f"{self.credential_key}.cred_upload_check.json",
        )

    def sync(self):
        data = self.dicionary_from_bash(self.credential_location)

        if data:
            self.set_creds_in_env(data)
            update_cloud = lambda cdata: self._aws_kms.deploy_secret(
                name=self.credential_key, secret_data=cdata
            )
            if self.valid_public_record:
                self.info("checking public upload record")
                self.update_public_record(data, update_callback=update_cloud)

            else:
                update_cloud()
                self.update_public_record(data)

        elif self.valid_public_record:
            public_record = self.public_record
            self.info("getting data from AWS KMS")
            data = {
                stage: {
                    key: self._aws_kms.get_secret_value(
                        secret_id=self.credential_key, key=key
                    )
                    for key, hash in pubcred.items()
                }
                for stage, pubcred in public_record.keys()
            }
            self.set_creds_in_env(data)

        else:
            self.warning("No Valid Sync Method Utilized")

    @property
    def valid_public_record(self):
        if os.path.exists(self.public_credential_upload_record):
            if os.path.getsize(self.public_credential_upload_record) > 16:
                return True

        return False  # Guilty until proven innocent

    @property
    def public_record(self):
        self.info("reading public record")
        with open(self.public_credential_upload_record, "r") as fp:
            return json.loads(fp.read())

    def generate_public_record_from(self, data):
        return {
            f"{self.credential_key}": {
                key: str(uuid.uuid5(uuid.NAMESPACE_DNS, val))
                for key, val in data.items()
            }
        }

    def update_public_record(self, data, update_callback=None):
        """Checks the public record, and decides if the current local creds should update the cloud config"""

        public_record = self.generate_public_record_from(data)
        if os.path.exists(self.public_credential_upload_record):
            if not public_record == self.public_record:
                self.info("updating public upload record")
                with open(self.public_credential_upload_record, "w") as fp:
                    fp.write(json.dumps(public_record))

                if update_callback is not None:
                    update_callback(data)

            else:
                self.info("credentials are up to date")

        else:
            self.info("no existing record, setting public upload record")
            with open(self.public_credential_upload_record, "w") as fp:
                fp.write(json.dumps(public_record))

        return public_record

    def set_creds_in_env(self, creds_dict):
        """sets credentials in enviornment"""
        # TODO: Ensure all keys are uppercase

        for cred, value in creds_dict.items():
            self.info(f"setting cred: {cred}")
            self.environ[cred] = value
            self.stored_env_vars.add(cred)

    def get_envvar(self, key, missing_callback=None, default=None, **kwargs):
        """A method to check the enviornment for a variable, and uses a default or callback if it doesn't exist
        :param key: a key value to check in the enviornment
        :param missing_callback: use the callback if key not found in environment, which takes a key arg, and anyo ther kwargs passed to this function
        :param default: if no callback and no key, we'll use this default which by default is None
        """
        if key in self.environ:
            return self.environ[key]

        elif missing_callback is not None:
            return missing_callback(key, **kwargs)

        return default

    def dicionary_from_bash(self, creds_path):
        """extracts export statements from bash file and aplies them to the python env"""
        self.info("checking {} for creds".format(creds_path))

        output = {}
        if os.path.exists(creds_path):
            self.debug("creds found")
            with open(creds_path, "r") as fp:
                txt = fp.read()

            os.chmod(creds_path, 0o740)  # Ensure correct permission

            lines = txt.split("\n")
            for line in lines:
                if line.startswith("export"):
                    key, val = line.replace("export", "").split("=")
                    self.debug(f"getting {key}")
                    output[key.strip()] = val.replace("'", "").replace('"', "")

            return output

        self.warning(f"no credentials found in {creds_path}")
        return output

    def bool_from(self, bool_env_canidate):
        if bool_env_canidate.lower() in ("yes", "true", "y", "1"):
            return True
        if bool_env_canidate.lower() in ("no", "false", "n", "0"):
            return False
        return None

    def __getitem__(self, key):
        if key in self.envvars:
            return os.environ[key]
        else:
            self.warning(f"no stored cred for {key}")

    @property
    def envvars(self):
        return list(self.stored_env_vars)

    @property
    def environ(self):
        # TODO: Expose as limited interface to env vars
        return os.environ

    def __contains__(self):
        return self.envvars

    def __getstate__(self):
        # Do not persist secrets other than on disk, or provided key managment service
        # Do not serialize object without removing keys, however provied a way to reinitalize.

        # TODO: Giveout AWS Secret Key Id (but they need aws creds)
        return {}

    def __setstate__(self, new_state):
        # TODO: Recreate Secrets object with new_state
        self.warning("setstate not implemented")
        pass

    @property
    def identity(self):
        return f"secrets-{self.credential_key}"


@otterize
class ClientSecrets(Secrets, ClientInfoMixin):
    """Automatically handles client secrets for managing application credentials between bash, env-vars, and aws secret manager

    TODO: Secrets will ensure all client .creds folders have appropriate permissions (644), and be ignored from git repos, maybe later dropbox.

    Secrets will store an keys as a unique combo of client+location+stage, like `YourClient.analysis.default`. Stages must be specified throguh DEV_ENV

    Client automatically creates stages and scopes for variables from knowledge about their .creds folder structures.
    """

    skip_wsl = True  # This ensures that the system is always real and not a fast temporary file cache

    # super_secret_types =  ('.pem','pub') #file extensions that are too secret for us to touch ( les untouchables )
    secret_scripts = ("env.sh",)
    creds_folders = (".creds",)

    # LOCAL FILE STRUCTURE INFUERRENCE
    def sync_local_to_cloud(self):
        """We sync all local credentials to the cloud, and create a cred_record.token file with each stage and its key and all the internal variable keys, but no values. This constitues a public record of the credentials so we can know when / if they've been updated"""
        lcreds = self.local_stage_credentials_package()

        stages = self.local_cred_stages(lcreds)
        locations = self.local_cred_locations(lcreds)

        prefix = os.path.commonpath(locations)
        loc_keys = {}
        for loc in locations:
            val = os.path.relpath(loc, start=prefix).replace(os.sep, ".")
            loc_keys[loc] = val if val != "." else ""

    def activate_env_from_local(self, stage=None, location=None):
        if stage is None and location is None:
            creds = self.local_credentials(self.stage_name, self.location)
            self.set_creds_in_env(creds)

    def local_credentials(self, stage=None, location=None):
        """Gets credentials by stage and location"""
        lcreds = self.local_stage_credentials_package()

        if stage is None:
            stage = self.stage_name

        if location is None:
            location = self.location

        if stage in lcreds and location in lcreds[stage]:
            return lcreds[stage][location]

        self.warning(f"no credentials found for stage: {stage} loc: {location}")

        return {}

    def local_credential_files(self):
        """Get all the appropriate credentials for the local directory, and segment them per stage"""
        current_location = self.location
        root = self.client_folder_root
        rel_location = os.path.relpath(os.path.realpath(os.path.curdir), root)
        rel_paths = list(filter(None, os.path.split(rel_location)))

        paths = [
            os.path.join(root, acc)
            for acc in itertools.accumulate(
                rel_paths, func=os.path.join, initial=root
            )
        ]

        creds_paths = {}

        stages = {"default": None}
        # Get a list of content per
        for path in paths:
            fils = [os.path.join(path, fil) for fil in os.listdir(path)]
            mtch = [
                fil
                for fil in fils
                if os.path.isdir(fil)
                and any(
                    list(
                        map(
                            lambda sec: os.path.basename(fil) == sec,
                            self.creds_folders,
                        )
                    )
                )
            ]

            for mtc in mtch:
                stagepaths = [
                    os.path.join(mtc, fil)
                    for fil in os.listdir(mtc)
                    if os.path.isdir(os.path.join(mtc, fil))
                ]

                for stagepath in stagepaths:
                    stage = os.path.basename(stagepath)

                    stage_contents = [
                        os.path.join(fp, fil)
                        for fp, _, fils in os.walk(stagepath)
                        for fil in fils
                        if any(list(map(fil.endswith, self.secret_scripts)))
                    ]

                    if stage in stages:
                        stages[stage][mtc] = stage_contents
                    else:
                        stages[stage] = {mtc: stage_contents}

                contents = [
                    os.path.join(mtc, fil)
                    for fil in os.listdir(mtc)
                    if any(list(map(fil.endswith, self.secret_scripts)))
                ]

                creds_paths[mtc] = contents

        stages["default"] = creds_paths

        return stages

    def local_credential_packages(self):
        """Get the list of files in order of priority for each stage provided in the local credentials"""
        stage_file_list = self.local_credential_files()

        stage_packages = {"default": None}

        for stage, cred_file_dict in stage_file_list.items():

            def decredential_file(filepath):
                for cred in self.creds_folders:
                    if filepath.endswith(cred):
                        return str(
                            pathlib.Path(filepath).parent
                        )  # Can't be dtwo types

            raw_dict = {
                decredential_file(key): creds
                for key, creds in cred_file_dict.items()
                if creds
            }

            # TODO: Ensure all credentials are from the same client (or exclusively in our company)
            # assert all(list(map(  )))

            paths_contains_creds = lambda path: [
                cred
                for key, creds in raw_dict.items()
                for cred in creds
                if key in path
            ]

            output = {
                key: paths_contains_creds(key)
                for key, creds in raw_dict.items()
            }
            stage_packages[stage] = output

        return stage_packages

    def local_stage_credentials_package(self):
        """Each Path / Stage & Its Credentials. Do not store these results in memory"""
        pkg = self.local_credential_packages()
        stage_envs = {}
        # default_env =
        for stage, creds_package in pkg.items():
            env = stage_envs[stage] = {}

            for pathkey, credfiles in creds_package.items():
                cred = {}
                for credfile in credfiles:
                    cred.update(self.dicionary_from_bash(credfile))
                env[pathkey] = cred

        # At this point all the creds files should have been read, acculating through each stage, however below we ensure each stage has the default parameters
        default = stage_envs["default"]
        for stage, creds_package in stage_envs.items():
            if stage != "default":
                for pathkey, env in creds_package.items():
                    if pathkey in default:  # should be
                        out = {}
                        out.update(default[pathkey])
                        out.update(env)
                        env.update(out)

                    else:
                        self.warning(f"path {pathkey} not found")

                    env["DEV_NAME"] = stage

        return stage_envs  # REMEMBER NEVER STORE THIS

    def local_cred_locations(self, stage_creds_package):
        """a dynamic property which returns the unique credential paths in the system"""
        out = set()
        for stage, creds in stage_creds_package.items():
            for cpath, cred in creds.items():
                out.add(cpath)
        return list(out)

    def local_cred_stages(self, stage_creds_package):
        stages = list(stage_creds_package.keys())
        return stages

    def set_conda_with_env(self, env_name, enviornment_dictionary):
        """extracts export statements from bash file and aplies them to the python env"""
        # TODO: Create Conda Envs For Client
        self.info("checking creds {} ".format(enviornment_dictionary))
        for key, val in enviornment_dictionary.items():
            self.info("setting {}".format(key))
            subprocess.run(
                [
                    "conda",
                    "env",
                    "config",
                    "vars",
                    "set",
                    f"{key.strip()}={val}",
                ]
            )

    @property
    def environ(self):
        return os.environ

    def __getstate__(self):
        # Do not persist secrets other than on disk, or provided key managment service
        # Do not serialize object without removing keys, however provied a way to reinitalize.

        # TODO: Giveout AWS Secret Key Id (but they need aws creds)
        return {}

    def __setstate__(self, new_state):
        # TODO: Recreate Secrets object with new_state
        self.warning("setstate not implemented")
        pass

    @property
    def identity(self):
        if self.stored_client_name is not None:
            self._ident = f"secrets-{self.stored_client_name}-{self.stage_name}"
        else:
            self._ident = f"secrets-{self.stage_name}"

        if (
            self.stored_client_name
            and not self.stored_client_name in self._ident
        ):
            self._log = None  # lazy cycle log name

        return self._ident

    def sync(self):
        """Gets the appropriate credential information and"""
        if "CLIENT_GDRIVE_PATH" in os.environ:
            self.info("got CLIENT_GDRIVE_PATH")
            CLIENT_G_DRIVE = os.environ["CLIENT_GDRIVE_PATH"]
        else:
            CLIENT_G_DRIVE = self.DEFAULT_GDRIVE

        if "CLIENT_GDRIVE_SYNC" in os.environ:
            self.info("got CLIENT_GDRIVE_SYNC")
            CLIENT_GDRIVE_SYNC = self.bool_from(
                os.environ["CLIENT_GDRIVE_SYNC"]
            )

        if "CLIENT_GMAIL" in os.environ:
            self.info("got CLIENT_GMAIL")
            CLIENT_GMAIL = os.environ["CLIENT_GMAIL"]

        if "CLIENT_NAME" in os.environ:
            self.info("got CLIENT_NAME")
            CLIENT_NAME = os.environ["CLIENT_NAME"]

        if "DB_NAME" in os.environ:
            DB_NAME = os.environ["DB_NAME"]
            self.info("Getting ENV DB_NAME")

        if "DB_CONNECTION" in os.environ:
            HOST = os.environ["DB_CONNECTION"]
            self.info("Getting ENV DB_CONNECTION")

        if "DB_USER" in os.environ:
            USER = os.environ["DB_USER"]
            self.info("Getting ENV DB_USER")

        if "DB_PASS" in os.environ:
            PASS = os.environ["DB_PASS"]
            self.info("Getting ENV DB_PASS")

        if "DB_PORT" in os.environ:
            PORT = os.environ["DB_PORT"]
            self.info("Getting ENV DB_PORT")

        if (
            self.SLACK_WEBHOOK_NOTIFICATION is None
            and "SLACK_WEBHOOK_NOTIFICATION" in os.environ
        ):
            self.info("getting slack webhook")
            self.SLACK_WEBHOOK_NOTIFICATION = os.environ[
                "SLACK_WEBHOOK_NOTIFICATION"
            ]


# TODO: Add CLI Method


if __name__ == "__main__":
    secret = Secrets()
