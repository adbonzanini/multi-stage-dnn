import socket
import time
import sys
from casadi import *
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os


# Folders and files
directory = '/Users/adbonzanini/Box Sync/Berkeley/Research/Explicit MPC Paper/DNN-MPC-Plasma/Model_ID'
fname = '/20191023_150532inputTrajectories.mat'

# Define time stamp to distinguish saved files
# TimeStamp = datetime.now().strftime('%Y-%m-%d-%Hh%Mm%Ss')

r=sio.loadmat(directory+fname)

u = r['U']  #rows: [power; flow]
t_el = 0.0

############## SETUP TCP/IP SOCKET CLIENT######################################
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = 'pi3.dyn.berkeley.edu'
print (host)
port = 2223

s.connect((host, port))
print('Connection established...')

k_delay=40;    #what units?

j = 0
wIn = u[0,j]
qIn = u[1,j]

U=vertcat(wIn, qIn,t_el)

msg = ','.join(str(e) for e in U.full().flatten())
print('Sending initial point...')
s.send(msg)


# Tdelay = 89.0
Tdelay = r['T'][0][0]   #<=================== CHECK THIS
t_0=time.time()
t_el=0.0

#Storage vector
Tplot = np.zeros((150, 1))
Yplot = np.zeros((150, 1))
Wplot = np.zeros((150, 1))
Qplot = np.zeros((150, 1))




Nsteps = u.shape[1]
while j<Nsteps:
      
    t_el=time.time()-t_0
    
    if t_el>=Tdelay:
        wIn = u[0,j]
        qIn = u[1,j]

        t_el=0
        t_0 = time.time()
        j=j+1
    
    U=vertcat(wIn, qIn,t_el) 
    msg = ','.join(str(e) for e in U.full().flatten())
    s.send(msg)
    ##get measurement
    a=s.recv(1024).decode() #recieve measurement     
    #print('Received Data')
    print(a)
    print(msg)
    
    y_meas=[ [i] for i in a.split('\n')] #convert to float
    try:
        y_m=[float(k) for k in y_meas[-2:][0][0].split(',')] 
        Y=y_m[-1]
        Yplot[:-1] = Yplot[1:]
        Yplot[-1] = Y 
        Wplot[:-1] = Wplot[1:]
        Wplot[-1] = wIn 
        Qplot[:-1] = Qplot[1:]
        Qplot[-1] = qIn 
        
        plt.subplot(311)  
        plt.cla() 
        plt.plot(Yplot)
        plt.ylim([np.min(Yplot)-3, np.max(Yplot)+3])
        plt.ylabel('T')
        plt.subplot(312)  
        plt.cla() 
        plt.plot(Wplot)
        plt.ylim([np.min(Wplot)-0.5, np.max(Wplot)+0.5])
        plt.ylabel('w')
        plt.subplot(313)  
        plt.cla() 
        plt.plot(Qplot)
        plt.ylim([np.min(Qplot)-0.5, np.max(Qplot)+0.5])
        plt.ylabel('q')
        plt.show()
        plt.pause(0.05)
    except:
        pass
    
    print('step %i of %i' %(j, float(u.shape[1])))


s.close()