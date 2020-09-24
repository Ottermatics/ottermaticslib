from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.orm import backref
from sqlalchemy.orm import scoped_session
from sqlalchemy_utils import *

import functools
from twisted.internet import defer, reactor, threads, task
from twisted.python import context

import os

import numpy

import signal

from urllib.request import urlopen
from psycopg2.extensions import register_adapter, AsIs
from os import sys


def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)

def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)

def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

def addapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))

register_adapter(numpy.float64, addapt_numpy_float64)
register_adapter(numpy.int64, addapt_numpy_int64)
register_adapter(numpy.float32, addapt_numpy_float32)
register_adapter(numpy.int32, addapt_numpy_int32)
register_adapter(numpy.ndarray, addapt_numpy_array)

#Pull IN User / Password from OS ENVIORNMENTAL
#HOST = 'testdb.ctiuz8xrxfqe.us-east-1.rds.amazonaws.com'
#HOST = 'smartx-dev-db.cluster-ctiuz8xrxfqe.us-east-1.rds.amazonaws.com'

if 'DB_CONNECTION' in os.environ:
    HOST = os.environ['DB_CONNECTION']
    print("Getting OS DB_CONNECTION")
else:
    HOST = 'localhost'
if 'DB_USER' in os.environ:
    USER = os.environ['DB_USER']
    print("Getting OS DB_USER")
else:
    USER = 'postgres'
if 'DB_PASS' in os.environ:
    PASS = os.environ['DB_PASS']
    print("Getting OS DB_PASS")
else:
    PASS = ''
if 'DB_PORT' in os.environ:
    PORT = os.environ['DB_PORT']
    print("Getting OS DB_PORT")
else:
    PORT = 5432

#HOST = 'localhost'

def is_ec2_instance():
    """Check if an instance is running on AWS."""
    result = False
    meta = 'http://169.254.169.254/latest/meta-data/public-ipv4'
    try:
        result = urlopen(meta,timeout=5.0).status == 200
        return True
    except:
        return False
    return False   

#connection_string = "host='{host}' user='{user}'  password='{passd}'".format(host=HOST,user=USER,passd=PASS)
def configure_db_connection(database_name):
    '''Returns Scoped Session, engine, and a connection_string'''
    connection_string = "postgresql://{user}:{passd}@{host}:{port}/{database}"\
                        .format(host=HOST,user=USER,passd=PASS,port=PORT,database=database_name)

    print('Connecting with {}'.format(connection_string))

    engine = create_engine(connection_string)
    engine.echo = False

    scopefunc = functools.partial(context.get, "uuid")

    session_factory  = sessionmaker(bind=engine,expire_on_commit=True)
    Session = scoped_session(session_factory, scopefunc = scopefunc)

    return Session,engine,connection_string

DataBase = declarative_base()

def rebuild_database(engine,connection_string):
    #Create Database If It Doesn't Exist
    if not is_ec2_instance():
        answer = input("We Are Going To Overwrite The Databse {}\nType 'CONFIRM' to continue:\n".format(HOST))
    else:
        answer = 'CONFIRM'

    if answer == 'CONFIRM':
        #Create Database If It Doesn't Exist
        if not database_exists(connection_string):
            print("Creating Database: {}".format(connection_string))
            create_database(connection_string)
        else:
            #Otherwise Just Drop The Tables
            DataBase.metadata.drop_all(engine)
        #(Re)Create Tables
        DataBase.metadata.create_all(engine)
    else:
        raise Exception("Ah ah ah you didn't say the magic word")

def cleanup_sessions(Session):
    print("Closing All Active Sessions")
    Session.close_all()

def cleanup_sessions_on_exit(Session):
    import atexit

    def customHandler(signum, stackframe):
        print("Got signal: {}".format(signum))
        reactor.callFromThread(reactor.stop) # to stop twisted code when in the reactor loop
    signal.signal(signal.SIGINT, customHandler)

    reactor.addSystemEventTrigger('during', 'shutdown', cleanup_sessions)

    atexit.register(cleanup_sessions,Session)