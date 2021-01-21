from multiprocessing import *
from patterns import *

from deco import *


from install_asyncio_reactor import *

from twisted.internet import endpoints, reactor, protocol
from twisted.spread import pb
from twisted.application import service, internet as appnet

from adpt_logging import *
from process_interaction import ProcessIO

import os,sys
import argparse

class AutoProcess(object):
    '''Uses Deco Library To Parallalize Functions using @concurrent decorator'''
        __metaclass__ = Singleton


#unix_socket_path = 'unix:path=/var/run/adaptproc.sock'
TRAN = 'tcp'
HOST = '127.0.0.1'
REMOTE_PORT = 42069

#Consider These Remote Objects Insecure
class ProcessManager(pb.Root,LoggingMixin):
    '''In which we create processes based on USB Events
    
    We store process communication structures per PID
    -remote protocols stored when the remote process calls 'remote_process_created'
    -process protocols are added when their connect callback is triggered

    '''
    
    _remote_processes = {}
    _process_protocols = {}
    factory = None

    def __init__(self,**kwargs):
        super(ProcessManager,self).__init__()

    #Process Managment Functions
    def shutdown_client_nice(self, client_id):
        '''We will attempt to shutdown a client using the remove protocol'''

        rp = self.remote_processes[client_id]

        #Run Client Shutdown Errback Chain
        shutdown = rp.callRemote("shutdown")
        shutdown.addCallback(self.check_process_shutdown, client_id)
        shutdown.addErrback(self.shutdown_error, client_id)
        shutdown.addCallback(self.cleanup_process, client_id)

        return shutdown

    def shutdown_client_force(self, client_id):
        '''We will attempt to shutdown a client using the remove protocol'''

        pp = self._process_protocols[client_id]
        
        shutdown = task.deferLater( 1, pp.killProcess)
        shutdown.addCallback(self.check_process_shutdown, client_id)
        shutdown.addErrback(self.shutdown_error, client_id)
        shutdown.addCallback(self.cleanup_process, client_id)        
   

    def cleanup_process(self,success,pid):
        '''Shutdown was a sucess we now cleaneup the process'''
        #Remove Remote Processes
        rmt = self._remote_processes.pop( pid )

        #Remove Process Protocols
        pp = self._process_protocols.pop( pid )
        
        self.info("Process Succesfully Shutdown: {}".format(pid))
        

    def shutdown_error(self,failure, pid):
        '''
        Shutdown was a failure, we will now force the process to shutdown
        
        Call Forceful Shutdown Later
        '''
        self.warning("Process Did Not Shutdown Correctly: {}".format(pid))
        reactor.callLater(0,self.shutdown_client_force,pid)

    async def check_process_shutdown(self, remote_return, pid, wait_time=10,wait=1):
        elapsed_time = 0
        while True:
            await asyncio.sleep(wait)
            if not psutil.pid_exists(pid):
                #Go To Success Callback
                return True
            elapsed_time += wait
            if elapsed_time > wait_time:
                #Go To Errback
                raise Exception("Process Not Shutdown In {}s".format(elapsed_time))



    #Process Storage and Identification 
    def remote_process_created(self, process, client_id):
        self._remote_processes[client_id] = process

    @property
    def remote_processes(self):
        return self._remote_processes

    @property
    def process_protocols(self):
        return self._process_protocols

    #Process Initiation
    def launch_process(self,device,*args,**connection_args):
        #reactor.spawnProcess(protocol.ProcessProtocol(),)
        self.info("Launching Process")
        #run_client()
        pp = ProcessIO(self)

        command = [ sys.executable.encode('utf8'), b'remote_process_managment.py'\
                                    ,b'-ho','{}'.format(HOST).encode('utf8')\
                                    ,b'-p','{}'.format(PORT).encode('utf8')\
                                    ,b'-r', b'remote' ]
        commandstr = b' '.join(command)                                 
        self.info('>>>{}'.format(commandstr.decode('utf8')))
        reactor.spawnProcess(pp,command[0],command,env={**os.environ})

        return pp.on_connection

    #Validation
    def check_process_health(self):
        '''We poll each process and store the results'''

        for client_id,remote_proc in self.processes.items():
            self.info("Checking Status Of Process: {}".format(client_id))
            try:
                d = remote_proc.callRemote("status")
                d.addErrback( self.status_error , client_id )
                d.addCallback( self.status_recieved, client_id )

            except Exception as e:
                self.error(e,"Trouble Checking Process {}".format(client_id))

    #Methods to Run This Class
    @classmethod
    def run_remote_server( cls, t=TRAN,h=HOST,p=PORT):
        server_connection_string = 'tcp:interface={h}:port={p}'.format(h=HOST,p=PORT)

        root = cls()
        factory = pb.PBServerFactory(root)

        root.info("Creating Server Endpoint: {}".format(server_connection_string))
        enpt = endpoints.serverFromString(reactor, server_connection_string )

        #Create Server & Run
        listeningport =  enpt.listen( factory )
        listeningport.addCallback(lambda lp: root.info("ListeningPort: {}".format(lp)))
        listeningport.addErrback( root.error )

        return root

    def start_remote_server( self, t=TRAN,h=HOST,p=PORT):
        server_connection_string = 'tcp:interface={h}:port={p}'.format(h=HOST,p=PORT)

        factory = pb.PBServerFactory(self)

        self.info("Creating Server Endpoint: {}".format(server_connection_string))
        enpt = endpoints.serverFromString(reactor, server_connection_string )

        #Create Server & Run
        listeningport =  enpt.listen( factory )
        listeningport.addCallback(lambda lp: self.info("ListeningPort: {}".format(lp)))
        listeningport.addErrback(self.error)

       

   

class RemoteProcess(pb.Referenceable,LoggingMixin):
    '''Remote protocol which runs in a separate process per USB device
    '''

    root = None

    def connect(self, root):
        self.root = root
        dcon = self.root.callRemote("process_created", self, self.remote_process_info())
        dcon.addCallback(self.connectionEstablished)
        
    def connectionEstablished(self,*ignore):
        self.log("connection Established")

    def remote_shutdown(self):
        self.log("shutting down")

        return self.killProcess()

    def remote_status(self):
        return 'OK'

    def remote_process_info(self):
        return os.getpid()

    def log(self,message,level=30):
        '''Logs To Process Manager'''
        if self.root is not None:
            self.root.callRemote("log",'RMT:{}|{}'.format(os.getpid(),message),level)
        else:
            reactor.callLater(5,self.log,'DELAYED:{}'.format(message),level)

    @defer.inlineCallbacks
    def cleanupProcess(self, ignore):
        '''Add in methods to cleanup this process'''
        #Stop STreams
        #Close COMS
        #Close Threads
        yield reactor.stop()
        return True


    def killProcess(self):
        '''We try to nicely kill the process'''
        d = self.cleanupProcess()
        d.addBoth(lambda ignore: os.kill( os.getpid(), 15))

        return d


    @classmethod
    def run_remote_client(cls,t=TRAN,h=HOST,p=PORT):

        start_system_logging()
                
        factory = pb.PBClientFactory()

        client_protocol = cls()
        client_connection_string = 'tcp:host={h}:port={p}'.format(h=h,p=p)    
        
        client_protocol.log("Creating Client Endpoint: {}".format(client_connection_string))
        enpt = endpoints.clientFromString(reactor,client_connection_string)
        
        def do_connection(broker): 
            d = factory.getRootObject()
            d.addCallback(client_protocol.connect)
            d.addErrback(lambda e: print(e))        

            client_protocol.log("Got Client: {}".format( client_protocol ))
        
        reconnectingService = appnet.ClientService( enpt, factory)
        onConnected = reconnectingService.whenConnected()
        onConnected.addCallback(do_connection)

        client_protocol.log("Starting Client")
        reconnectingService.startService()       





if __name__ == '__main__': #Python Approach

    parser = argparse.ArgumentParser(description="Data channels ping/pong")
    parser.add_argument("--port","-p", default = PORT)
    parser.add_argument("--host",'-ho', default = HOST)
    parser.add_argument("--tran","-t", default = TRAN)    
    parser.add_argument("--role",'-r' ,default='manager',choices = ["manager", "remote"])

    start_system_logging()

    args = parser.parse_args()

    if args.role == 'manager':
        
        pm = ProcessManager.run_remote_server(args.tran,args.host,args.port)
        #reactor.callWhenRunning( pm.launch_process )

    else:
        RemoteProcess.run_remote_client(args.tran,args.host,args.port)

    try:
        reactor.run()
    except Exception as e:
        print(e)
    reactor.stop()      
