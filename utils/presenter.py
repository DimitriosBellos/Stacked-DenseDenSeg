# -*- coding: utf-8 -*-
from __future__ import division
from PIL import Image
import numpy as np
import platform
import os
import sys
from .timer import Timer
from .colors import Colors

class PrintProgress(object):
    def __init__(self):
        self.barDone=True
        self.timer=[]
        self.times=[]
        self.indices=[]

    @staticmethod
    def printStep(seconds_):
        seconds = seconds_
        days = np.floor(seconds / (3600 * 24))
        seconds = seconds - days * 3600 * 24
        hours = np.floor(seconds / 3600)
        seconds = seconds - hours * 3600
        minutes = np.floor(seconds / 60)
        seconds = seconds - minutes * 60
        secondsf = np.floor(seconds)
        seconds = seconds - secondsf
        millis = np.floor(seconds * 1000)
        f = ''
        if days > 0:
            f = f + ('%d' % days) + 'd'
        if hours > 0:
            f = f + ('%d' % hours) + 'h'
        if minutes > 0:
            f = f + ('%d' % minutes) + 'm'
        if secondsf >= 0:
            f = f + ('%d' % secondsf) + 's'
        if millis > 0:
            if millis > 99:
                millis = np.around(millis, -1) / 10
            f = f + ('%d' % millis) + 'ms'
        return f

    @staticmethod
    def printTime(seconds_):
        seconds = seconds_
        days = np.floor(seconds / (3600 * 24))
        seconds = seconds - days * 3600 * 24
        hours = np.floor(seconds / 3600)
        seconds = seconds - hours * 3600
        minutes = np.floor(seconds / 60)
        seconds = seconds - minutes * 60
        secondsf = np.floor(seconds)
        seconds = seconds - secondsf
        millis = np.floor(seconds * 1000)
        f = ''
        if days > 0:
            f = f + ('%d' % days) + 'd'
        if hours >= 0:
            f = f + ('%d' % hours) + 'h'
        if minutes >= 0:
            f = f + ('%d' % minutes) + 'm'
        if secondsf >= 0:
            f = f + ('%d' % secondsf) + 's'
        if millis > 0:
            if millis > 99:
                millis = np.around(millis, -1) / 10
            f = f + ('%d' % millis) + 'ms'
        return f

    @staticmethod
    def getTermLength():
        if platform.system() == 'Windows':
            return 80
        tputf = os.popen('tput cols', 'r')
        w = tputf.read().split('\n')[0]
        if w.isdigit():
            w = int(w)
        else:
            w = 80
        tputf.close()
        return w

    def __call__(self, iteration, iterations, strs):
        termLength = 80#min(PrintProgress.getTermLength(), 80)
        barLength = termLength - 34
        smoothing = 100
        syss=Colors()

        # Compute percentage
        percent = (iteration * barLength) / iterations

        # start new bar
        if self.barDone:
            self.barDone = False
            self.timer = Timer()
            self.times = [self.timer.get_value()]
            self.indices = [iteration]
        else:
            sys.stdout.write('\r')
    
        sys.stdout.write(syss().Magenta + 'Epoch:' + syss().Cyan + (' %d/%d' % (iteration, iterations)))
    
        if not self.barDone:
            # print bar
            sys.stdout.write(syss().Green + '▕')
            prt=(' %3.2f%% ' % (percent*100/barLength))
            for i in range(1,barLength+1):
                if i <= np.floor(percent):
                    sys.stdout.write(syss().Green + '█')
                elif i == np.floor(percent)+1:
                    if percent-np.floor(percent)<0.0625:
                        sys.stdout.write(' ')
                    elif percent-np.floor(percent)<0.1875:
                        sys.stdout.write(syss().Green + '▏')
                    elif percent-np.floor(percent)<0.3125:
                        sys.stdout.write(syss().Green + '▎')
                    elif percent-np.floor(percent)<0.4375:
                        sys.stdout.write(syss().Green + '▍')
                    elif percent-np.floor(percent)<0.5625:
                        sys.stdout.write(syss().Green + '▌')
                    elif percent-np.floor(percent)<0.6875:
                        sys.stdout.write(syss().Green + '▋')
                    elif percent-np.floor(percent)<0.8125:
                        sys.stdout.write(syss().Green + '▊')
                    elif percent-np.floor(percent)<0.9375:
                        sys.stdout.write(syss().Green + '▉')
                    else:
                        sys.stdout.write(syss().Green + '█')
                else:
                    sys.stdout.write(' ')
                if np.ceil(barLength/2)-np.floor(len(prt)/2) < i < np.ceil(barLength/2)+np.ceil(len(prt)/2)+1:
                    lim=np.ceil(barLength/2)-np.floor(len(prt)/2)+1
                    sys.stdout.write('\b' + syss().Green + prt[int(i-lim)])
            sys.stdout.write(syss().Green + '▏')

            # time stats
            elapsed = self.timer.get_value()
            if (iteration - self.indices[0])==0:
                step=0
            else:
                step = (elapsed-self.times[0]) / (iteration - self.indices[0])
            if iteration<=self.indices[0]:
                self.indices = []
                step = 0
            remaining = max(0, (iterations - iteration) * step)
            self.indices.append(iteration)
            self.times.append(elapsed)
            if len(self.indices) > smoothing:
                del self.indices[0]
                del self.times[0]

            # Print remaining time when running or total time when done.
            sys.stdout.write(syss().Red + ' Acc: ' + syss().Green + strs[0])
            tm = ' ETA: ' + PrintProgress.printTime(remaining)
            tm = tm + ' | Elapsed: ' + PrintProgress.printTime(elapsed)
            tm = tm + ' | Step: ' + PrintProgress.printStep(step) + '              '
            sys.stdout.write(syss().Cyan + tm)

            if percent == barLength:
                self.barDone = True
                sys.stdout.write('\n')
            # flush
            sys.stdout.flush()
            return
        else:
            return