from utils.presenter import PrintProgress
import time
import sys
N=100

printTrain=PrintProgress()
for i in range(0,N):
    #print('testjfgmjfjvjjadfknvjarjpvnjkdvjrvkmlkmvaoekvmdlfmbnlasmvdbnmfsdbnakslcfm')
    time.sleep(1)
    #print('a\n')
    #sys.stdout.write('test')
    #sys.stdout.write('go')
    #sys.stdout.write('\r')
    #sys.stdout.write('\n')
    #sys.stdout.flush()
    #sys.stdout.write('\b')
    printTrain(i+1, N, ['test'])