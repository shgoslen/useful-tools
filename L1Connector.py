'''
Name: L1Connector.py
Author: Steve Goslen
Date: September 2020

This is the L1Connector that run's as a python process in the LaaS server
Its job is to connect to the L1 and login and stay logged in
It then listens on a named pipe for connection and disconnection requests that it then sends to the L1
waiting for a reply. The reply is then passed back to the named pipe which the responds back to the expect
process with the results

The goal is to avoid the time for login and logout that LaaS makes on every connection request.
'''


import logging, sys, os , re, signal, stat, time
import pexpect
import paramiko 
from multiprocessing import Process, Queue


logging.basicConfig(filename='/var/log/taas/L1Connector.log', level=logging.DEBUG)

class L1Connector():

    def __init__(self):
        self.loggedin = False
        pass

    def login(self):
        '''
        Initialization, login to the switch, and create pexpect obj for current session.
        :param loginInfo: Dict of strings containing L1 switch info (username, password, and chassisIP)
        '''
        if self.loggedin:
           return True

        logging.info('L1C=>Logging into L1')
        #See if we can make an SSH connection to the L1 switch
        try:
           self.L1 = paramiko.SSHClient()
           self.L1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
           self.L1.connect(hostname="10.207.196.19", username="iotauto", password="Autom8!t", port=443)
        except Exception as e:
            logging.error('SSHClient exception: {}'.format(str(e)))
            return False

        #Now create a channel to use for communication
        try:
            self.channel = self.L1.invoke_shell()
            self.channel.settimeout(60)
            self.channel.setblocking(0)
        except Exception as e:
            logging.error('Channel exception: {}'.format(str(e)))
            return False

        #This should have logged into the L1 switch so see if there is anything to receive
        for loopCount in range(60):
            time.sleep(1)
            if self.channel.recv_ready():
                logging.debug('recv_ready True, loopCount: {}'.format(loopCount)) 
                loopCount = 0
                break
        if loopCount != 0:
            logging.error('Logon to L1 switch timed out')
            return False
 
        #So get the data from the L1 and check to see that we are indeed loggin
        output = ''
        buff = ''
        for loopCount in range(10):
            time.sleep(1)
            try:
               buff = self.channel.recv(8000)
            except Exception as e:
               logging.error('recv exception: {} timeout?, loopCount: {}'.format(e,loopCount))
               return False

            output += buff
            if '=>' in buff:
                logging.debug('found L1 prompt, loopCount: {}'.format(loopCount))
                loopCount = 0
                break

        if loopCount !=0:
            logging.error('did not find find L1 prompt: {}'.format(output))
            return False

        logging.debug('output: {}'.format(output))
        goodLoginRegex = 'iotauto is now logged on\.'
        searchRslt = re.search(goodLoginRegex, output, re.MULTILINE)
        logging.debug('searchRslt: {}'.format(searchRslt))
        goodLoginRegex = 'iotauto is now logged on\.'
        if re.search(goodLoginRegex, output, re.MULTILINE) == None:
            #Failed to login
            logging.error('Failed to login to L1 switch')
            return False

        #Okay we are logged in so return True
        self.loggedin = True
        return True

    def connect_ports(self, connect_cmd):
        '''
        Creates connections between specified ports 
        :param src_port: the source/from port to be connected
        :param dest_port: the dest/to port to be connected 
        :return: True if connection established successfully, False otherwise '''

        logging.debug('L1C=>connect_ports')
            
        #See if its okay to send a command 
        for loopCount in range(20):
            time.sleep(1)
            if not self.channel.send_ready():
                logging.warning('channel not ready for sending')
            else:
                logging.debug('channel ready for sending')
                loopCount = 0
                break
 
        if loopCount !=0:
            logging.error('Not ready to send')
            return False

        #Now send the data
        try:
            sent = self.channel.send(connect_cmd + '\r\n')
        except Exception as e:
            logging.warning('channel send exception: {}'.format(str(e)))

        logging.debug('connect_ports sent {} bytes'.format(sent))
 
        #Now get the "echo" of the command back from the L1
        for loopCount in range(90):
            time.sleep(1)
            if self.channel.recv_ready(): 
                logging.debug('recv_ready True, loopCount: {}'.format(loopCount))
                loopCount = 0
                break
        if loopCount != 0:
            logging.error('Conn command to L1 timed out')
            return False
        #This should be the echo
        output = ''
        buff = ''
        for loopCount in range(90):
            time.sleep(1)
            try:
               buff = self.channel.recv(8000)
            except Exception as e:
               logging.error('recv exception: {} timeout?, loopCount: {}'.format(e,loopCount))
               rr = self.channel.recv_ready()
               logging.debug('recv_ready: {}'.format(rr))
               continue

            output += buff
            if 'CON' in buff:
                logging.debug('found echo, loopCount: {}'.format(loopCount))
                loopCount = 0
                break

        if loopCount !=0:
            logging.error('did not find find echo: {}'.format(output))
            return False

  
        #Now make sure that the connection was actually made
        for loopCount in range(90):
            time.sleep(1)
            if self.channel.recv_ready():
                logging.debug('recv_ready True, loopCount: {}'.format(loopCount))
                loopCount = 0
                break
        if loopCount != 0:
            logging.error('Conn command to L1 timed out')
            return False

        output = ''
        buff = ''
        for loopCount in range(90):
            time.sleep(1)
            try:
               buff = self.channel.recv(8000)
            except Exception as e:
               logging.error('recv exception: {} timeout?, loopCount: {}'.format(e,loopCount))
               rr = self.channel.recv_ready()
               logging.debug('recv_ready: {}'.format(rr))
               continue

            output += buff
            if '=>' in buff:
                logging.debug('found L1 prompt, loopCount: {}'.format(loopCount))
                loopCount = 0
                break

        if loopCount !=0:
            logging.error('did not find find L1 prompt: {}'.format(output))
            return False

        #So get the data from the L1 and check to see that we are indeed loggin
        goodConnRegex = 'Successful\.'
        logging.debug('output: {}'.format(output))
        if re.search(goodConnRegex, output, re.MULTILINE) == None:
            #Failed to connect 
            logging.error('Failed to make connection ')
            return False

        #No errors to deal with so the connection is made
        return True

    def disconnect_ports(self, disc_cmd):
        '''
        Disconnects specified existing L1 connections
        :param src_port: the source/from port to be disconnected
        :param dest_port: the dest/to port to be disconnected
        :return: True if disconnected successfully, False otherwise
        '''

        logging.debug('L1C=>disconnect_ports')

        #See if its okay to send a command
        for loopCount in range(20):
            time.sleep(1)
            if not self.channel.send_ready():
                logging.warning('channel not ready for sending')
            else:
                logging.debug('channel ready for sending')
                loopCount = 0
                break

        if loopCount !=0:
            logging.error('Not ready to send')
            return False

        #Now send the data
        try:
            sent = self.channel.send(disc_cmd + '\r\n')
        except Exception as e:
            logging.warning('channel send exception: {}'.format(str(e)))

        #Now get the "echo" of the command back from the L1
        for loopCount in range(90):
            time.sleep(1)
            if self.channel.recv_ready():
                logging.debug('recv_ready True, loopCount: {}'.format(loopCount))
                loopCount = 0
                break
        if loopCount != 0:
            logging.error('Conn command to L1 timed out')
            return False
        #This might be the echo but check if its not
        output = ''
        buff = ''
        for loopCount in range(90):
            time.sleep(1)
            try:
               buff = self.channel.recv(8000)
            except Exception as e:
               logging.error('recv exception: {} timeout?, loopCount: {}'.format(e,loopCount))
               rr = self.channel.recv_ready()
               logging.debug('recv_ready: {}'.format(rr))
               continue

            output += buff
            if 'DISC' in buff:
                logging.debug('Timedout waiting for echo, loopCount: {}'.format(loopCount))
                loopCount = 0
                break

        if loopCount !=0:
            logging.error('did not find find echo: {}'.format(output))
            return False

        #Now make sure that the connection was torn down
        logging.debug('Echo: {}'.format(output))
        #So get the data from the L1 and check to see that we are indeed loggin
        goodDiscRegex = 'disconnected\.|Association not found!'
        logging.debug('output: {}'.format(output))
        if re.search(goodDiscRegex, output, re.MULTILINE) != None:
            #Disconnect worked
            return True

        #Need to wait for the rest of the output
        for loopCount in range(90):
            time.sleep(1)
            if self.channel.recv_ready():
                logging.debug('recv_ready True, loopCount: {}'.format(loopCount))
                loopCount = 0
                break
        if loopCount != 0:
            logging.error('Disc command to L1 timed out')
            return False

        output = ''
        buff = ''
        for loopCount in range(90):
            time.sleep(1)
            try:
               buff = self.channel.recv(8000)
            except Exception as e:
               logging.error('recv exception: {} timeout?, loopCount: {}'.format(e,loopCount))
               rr = self.channel.recv_ready()
               logging.debug('recv_ready: {}'.format(rr))
               continue

            output += buff
            if '=>' in buff:
                logging.debug('found L1 prompt, loopCount: {}'.format(loopCount))
                loopCount = 0
                break

        if loopCount !=0:
            logging.error('did not find find L1 prompt: {}'.format(output))
            return False

        #So get the data from the L1 and check to see that we are indeed loggin
        logging.debug('output: {}'.format(output))
        if re.search(goodDiscRegex, output, re.MULTILINE) == None:
            #Failed to disconnect
            logging.error('Failed to make disconnect ')
            return False

        #No errors to deal with so the connection is made
        return True

    def logout(self):
        '''
        Exits the L1 switch and ends the pexpect stream
        :return:
        '''

        logging.debug('L1C=>logout')

        #See if its okay to send a command
        if not self.channel.send_ready():
            logging.warning('channel not ready for sending')

        #Now send the data
        try:
            sent = self.channel.send('exit\r\n')
        except Eception as e:
            logging.warning('channel send exception: {}'.format(str(e)))

        return True


#Define a signal handler and catch HUP (Hang up)
def handler(signum, frame):
    logging.info('L1C=>Signal handler called with signal', signum)

signal.signal(signal.SIGHUP, handler)
signal.signal(signal.SIGINT, handler)

def processQueue(inputQueue, L1Conn):
    '''
    This will wait for messages (CON and DIS) to show up on a queue and then process them, which would mean
    sending them to the L1, and waiting for the response back from the L1. It will then reply back to the named
    pipe that is part of the command with the results that go back to the tcl process
    :param inputQueue:
    :return:
    '''

    logging.debug('processQueue')
    logging.debug('inputQueue: {}'.format(inputQueue))

    L1ConnDir = '/home/taasuser/L1Connector/'
    
    pipeRegex = '[LIN|CON|DIS]:(.*)'
    cmdRegex = '[CON|DIS]:(.*):(.*)' 
    # Now wait for messages in the queue
    while True:
        try:
            line = inputQueue.get(block=True)
        except Exception as e:
            logging.error('Exception reading queue: {}'.format(str(e)))
            exit

        logging.debug('line from queue: {}'.format(line))

        if 'LIN:' in line:
            pipe = re.search(pipeRegex, line, re.MULTILINE).group(1)
            logging.info('L1C=>pipe: {}'.format(pipe))
            result = L1Conn.login()

        if 'CON:' in line:
            pipe = re.search(cmdRegex, line, re.MULTILINE).group(1)
            cmd = re.search(cmdRegex, line, re.MULTILINE).group(2)
            logging.info('L1C=>pipe: {}'.format(pipe))
            logging.debug('L1C=>The connection cmd is {}'.format(cmd))
            result = L1Conn.connect_ports(cmd)

        if 'DIS:' in line:
            pipe = re.search(cmdRegex, line, re.MULTILINE).group(1)
            cmd = re.search(cmdRegex, line, re.MULTILINE).group(2)
            logging.info('L1C=>pipe: {}'.format(pipe))
            logging.debug('L1C=>The connection cmd is {}'.format(cmd)) 
            result = L1Conn.disconnect_ports(cmd)

        #Send the result back
        fd = os.open(L1ConnDir + pipe, os.O_WRONLY)
        logging.info('L1C=>pipe fd: {}'.format(fd))
        if result:
            logging.info('L1C=>Connection made')
            os.write(fd, b'PASS')
        else:
            logging.info('L1C=>Connection failed')
            os.write(fd, b'FAIL')
        os.close(fd)
        logging.info('L1C=>Con result sent and flushed')


LintOutRegex = '[LIN|LOUT]:(.*)'
CmdRegex = '[CON|DIS]:(.*):(.*)'

#now login to the L1 switch
#Instead of just one connection to the L1 make a small number of
#connections that will be used in a round robin way to help make
#L1 connections faster
logging.info('L1C=>************ L1 Connector is starting!!!!************')

#Create the L1 connectors, queues and processes
L1CList=[]
L1QList=[]
jobs = []
for i in range(2):
    L1CList.append(L1Connector())
    L1QList.append(Queue())

#Now fire up the processes
for i in range(2):
    p = Process(target=processQueue, args=(L1QList[i],L1CList[i],))
    jobs.append(p)
    p.start()

#setup named pipe to listen to
logging.info('L1C=>Creating L1ConnInput pipe')
L1ConnDir = '/home/taasuser/L1Connector/'
FIFO = L1ConnDir + 'L1ConnInput'
try:
    os.remove(FIFO)
except Exception as e:
    logging.debug('L1C=>Exception on remove: {}'.format(str(e)))

os.mkfifo(FIFO)
logging.info('L1C=>Fifo created')
try:
    os.chmod(FIFO, 0o777)
except Exception as e:
    logging.debug('L1C=>Exception on chmod: {}'.format(str(e)))

logging.info('L1C=> chmod worked')

#Create a list that we will put file dictionaries into
L1QListIdx = 0

while True:
    with open(FIFO) as fifo:
        logging.info('L1C=>waiting for input from {}'.format(fifo))
        logger = logging.getLogger()
        logger.handlers[0].flush()
        for line in fifo:
            logging.info('L1C=>line: {}'.format(line))
            #The line has two parts to the string cmd:string
            #The cmd portion has 4 possibilities: LIN, CON, DIS, LOUT
            #LIN: this is the LogIN request with the response pipe name
            #CON: the rest of the string is the connection request string
            #DIS: the rest of the string is the disconnection request string
            #LOUT: this is teh Log OUT request
            if 'LIN:' in line:
                pipe = re.search(LintOutRegex, line, re.MULTILINE).group(1)
                logging.info('L1C=>The response pipe is {}'.format(pipe))
                L1QList[L1QListIdx].put(line)

            if 'LOUT:' in line:
                pipe = re.search(LintOutRegex, line, re.MULTILINE).group(1)
                logging.info('L1C=>The response pipe is {}'.format(pipe))
                L1QListIdx = (L1QListIdx + 1) % len(L1QList)

            if 'CON:' in line:
                cmd = re.search(CmdRegex, line, re.MULTILINE).group(2)
                logging.debug('L1C=>sending command {} to queue'.format(cmd))
                L1QList[L1QListIdx].put(line)

            if 'DIS:' in line:
                cmd = re.search(CmdRegex, line, re.MULTILINE).group(2)
                logging.debug('L1C=>sending command {} to queue'.format(cmd))
                L1QList[L1QListIdx].put(line)


