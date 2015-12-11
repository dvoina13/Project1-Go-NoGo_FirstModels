# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:27:07 2013

@author: Shawn Olsen
"""

import os
import numpy as np
import itertools
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pickle


    
def pkl2df(fname=None, save=False, fdir=None,**kwargs):
    if not fname:
        fname = FileTools.get_fnames(fdir=fdir)
    elif not isinstance(fname,list):
        fname = [fname]
    for i,v in enumerate(fname):
        fg = ForageSession(path = v)
        (dtable, columnnames) = fg.maketrialsummary()
        tmp = pd.DataFrame(dtable,columns=columnnames)
        tmp['mouse'] = fg.mouseid
        tmp['date'] = fg.data['startdatetime']
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df,tmp])        
    return df     

def loadpkl(fpath=None,fdir=None):
    if fpath is None:
        fpath = get_fnames(fdir=None) 
    f = open(fpath,'rb')
    print "Loading pkl file:",fpath
    data = pickle.load(f)
    return data

def get_onlyfiles(directory,filetype=None):
    """Returns list of paths to files in directory, omitting subdirectories.
    Optionallly, return only files of particular type. """
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory,f))]
    if filetype is not None:
        onlyfiles = [f for f in onlyfiles if filetype in f]
    return onlyfiles  

def compute_hitrateDF(df,params=['Image','Ori','Size','PosY']):
    Img = np.sort(df.Image.unique())
    Ori = np.sort(df.Ori.unique())
    Size = np.sort(df.Size.unique())
    PosY = np.sort(df.PosY.unique())
    columnnames = params + ['yp','y_lo','y_hi']
    data = []
#    parametertable # TO DO: make parameter table instead of for loops (look into iteritems method of DataFrame)
    for ori in Ori:
        for sz in Size:
            for y in PosY:
                for im in Img:
                    v = df[(df['Ori'] == ori) & (df['Size'] == sz) & (df['PosY'] == y)& (df['Image'] == im)]['choice'].values             
                    ci_lo, ci_hi = binomialCI(sum(v),len(v),0.05)
                    yp = np.mean(v)
                    data.append((im,ori,sz,y,yp,ci_lo,ci_hi))
    hDF = pd.DataFrame(data,columns=columnnames)
    return hDF


class ForageSession(object):
    """ Class for analyzing and visualizing data from foraging session. \n
        Arguments: \n
        "path": full path of the pickled log file.
    """
    
    def __init__(self,path,data=None,**kwargs):
        if data is not None:
            self.data = data
        self.path = path
        self._processLogFile(data=data) 
        
    def _processLogFile(self,data=None):
        """ Loads a pickled log file.
        Puts all information into dictionary "data".
        Generates data structures useful for analyzing data including:
        terraintable, encounterorder, and pausetimes.
        """
        # If data was not supplied to ForageSession get it from pkl file.
        if data is None:
            # Load data from pkl file  
            data = loadpkl(fpath=self.path)
            self.data = data

        self.mouseid = data['mouseid']               
        self.laps = data['laps']
        self.rewards = data['rewards']
        self.terrain = data['terrain']
        self.terrainlog = data['terrainlog']
        self.posx = data['posx']
        self.vsyncintervals = data['vsyncintervals']
        self.wheeldiameter = 6.5*2.54 
        self.runningradius = 0.5*(2.0*self.wheeldiameter/3.0) # assumes mouse running 2/3 way from center of wheel 
    
        self._makeTerrainTable()
        self._getEncounterOrder()
        self._computeUpdateTimes()
        self._getEncounterTimes()
        self._getPauseTimes()
        self._computecmPerPixel()
        self._maketrialsummary_df()
        self._makehitrate_df()
        
        # if "summarydata" in data.keys():
        #     self.summarydata = data["summarydata"]
        # else:
        #     self.summarydata = self.create_summary()
            

    # ----- TIMING/SYNCING for OPHYS/EPHYS ANALYSIS ----- #

    # ----- COMPUTATIONS ----- #
    def getTrajectory(self, encounterindex=0, relativeposx=None, units="pixels"):
        """ Returns the position and time vectors for the nth object encountered, 
        where n is the index of the object in the encounterorder list. \n
        INPUT \n
        encounterindex: Index of object encountered.
        relativeposx: This parameters shifts the time vector such that t = 0  at
        the x value given by relativeposx. \n
        OUPUT x,t"""

        if relativeposx is None:
            try: 
                relativeposx = -self.monitorsize[0]/2
            except:
                relativeposx = -1920/2  # Hardcode default monitor size
        lapstart, lapend = self.getTrajectoryBounds(encounterindex)
        t = np.array(self.t[lapstart:lapend])
        x = np.array(self.posx[lapstart:lapend])
        ind = x >= relativeposx
        t = (t - t[ind.argmax()])
        if units == "cm":
            dx = self.data['dx']
            thetaraw = np.cumsum(np.array(dx[lapstart:lapend])) #Wheel rotations
            x = 2*np.pi*self.runningradius*thetaraw/360.0
        return x,t

    def _maketrialsummary_df(self):
        (dtable, columnnames) = self.maketrialsummary()
        tmp = pd.DataFrame(dtable,columns=columnnames)
        tmp['mouse'] = self.mouseid
        tmp['date'] = self.data['startdatetime']
        self.df = tmp

    def getTrajectoryBounds(self, encounterindex=0):
        """  Returns start and end index for encounter number given by encounterindex. \n
        INPUT \n
        encounterindex: Index of lap \n
        OUTPUT \n
        lapstart: Index in posx for beginning of lap \n
        lapend: Index in posx for end of lap"""
        if encounterindex == 0:
            lapstart = 0
        else:
            lapstart = self.laps[encounterindex-1][1]
        if encounterindex == len(self.encounterorder)-1:
            lapend = len(self.posx)
        else:
            lapend = self.laps[encounterindex][1]
        return int(lapstart), int(lapend)
    
    def timeInWindow(self,window=[-250,250]):
        pass
    
    def getCumulativeDistance(self):
        # initialize traw, xraw,& vraw
        traw = np.cumsum(self.vsyncintervals)/1000
        self.wheelDiameter = 6.5*2.54 #6.5" for now, but this should be a paremeter that is saved in the PKL file, in case we start using other diameters.
        self.runningRadius = 0.5*(2.0*self.wheelDiameter/3.0)
        self.dx = np.array(self.data['dx'],dtype=np.float)
        thetaraw = np.cumsum(self.dx)
        arclength = 2*np.pi*self.runningRadius*thetaraw/360.0
        return arclength
    
    def computeAvgSpeed(self,window=[250,10000]):
        avgSpeedPerLap = []
        for i in range(len(self.encounterorder)):
            x,t = self.getTrajectory(i)
            windowIndex = np.logical_and(x >= window[0], x <= window[1])
            x = x[windowIndex]
            t = t[windowIndex]
            x = self.removePositionOutliers(x)
            # Calculate velocity
            v = np.diff(x)/np.diff(t)
            avgSpeedPerLap.append(stats.nanmean(v))
        return avgSpeedPerLap
        
    def computeYesProb(self, confidence = False):
        yp = []
        for terraincode in range(len(self.terraintable)):
            encounters = self.filterEncounter(self.getTerrainInd(terraincode))
            yes = self.yesChoiceArray(encounters['pausetimes'])
            if np.size(yes) != 0:
                yp.append(float(sum(yes))/float(len(yes)))
            else:
                yp.append(np.nan)
            # Compute confidence interval
            if confidence:
                pass
        return yp
    
    def yesChoiceArray(self, pausetimes = None):
        # TODO: Make compatible with adaptive selectiontime
        if pausetimes == None:
            yes = np.array(self.pausetimes) >= self.get_selectiontime()
            self.yes = yes
        else:
            yes = np.array(pausetimes) >= self.get_selectiontime()
        return yes        
        
    def filterEncounter(self,encounterind):  
        encounters = {}
        encounters['pausetimes'] = np.array(self.pausetimes)[encounterind[0:-1]]
        encounters['encountertimes'] = np.array(self.encountertimes)[encounterind[0:-1]]
        encounters['encounterindex'] = encounterind
        return encounters              

    def getTerrainInd(self,terraincode = 0):
        encounterorder = np.array(self.encounterorder)
        i = np.where(encounterorder == np.array(terraincode))[0]
        return i

    def removePositionOutliers(self,x):
        outliers = [i for i, vals in enumerate(np.diff(x)) if abs(vals)>=50]  
        if len(outliers) > 1:
            for idx in range(0,len(outliers),2): 
#                print outliers[idx+1]
                try:
                    x[outliers[idx+1]] = x[outliers[idx]]
                except: # TODO: What' goin on here when index is out of range?
                    pass
        return x
        
    def removeNaNs(self,x):
        """ Removes NaNs from np.array """
        x = np.array(x)
        x = x[np.logical_not(np.isnan(x))]
        return x


    def get_selectiontime(self):
        """ Returns selectiontime in seconds.\n
        If selectiontime was varied in session only final value is returned. """
        try:
            selectiontime = self.data['terrain']['selectiontime']
        except Exception as e: 
            selectiontime = None
        return selectiontime  


# ----- PLOTTING AND VISUALIZATION ----- #
            
# TODO: Including plotting functions or develop separate module
    def computeYesProb_StoppingReport(self):
        yp = []
        CI = []
        for terraincode in range(len(self.terraintable)):
            encounters = self.filterEncounter(self.getTerrainInd(terraincode))
            yes = self.yesChoiceArray(encounters['pausetimes'])
            if np.size(yes) != 0:
                yp.append(float(sum(yes))/float(len(yes)))
            else:
                yp.append(nan)
            # Compute confidence interval
            CI.append(binomialCI(sum(yes),len(yes),(1-confidence)))           
        self.yes = yes
        return yp,CI
        
    def plotYesProb(self, fig=None,confidence=0.95):
        if fig is None:
            fig = plt.figure()
            
        yp,CI = self.computeYesProb_StoppingReport(confidence=T)
        
        ax = fig.add_subplot(111,label='yesChoicePlot')
        l1 = ax.plot(self.yp, 'o-')
        if self.LickDataExists:
            l2 = ax.plot(self.terraincode,self.yp_lick,'-',color=[0.5,0.5,0.5],markersize=8)
#        plt.legend([l1,l2], ["Actual", "Licking only"])
        for i in range(0,len(self.yp)):
#            ax.plot(np.arange(i,len(yp)),yp[i:],'o',color=self.colors(i),markersize=8)
            ax.plot(i,self.yp[i],'o',color=self.colors(i),markersize=8)
            if confidence:
                yerrdown=self.yp[i]-self.CI[i][0]
                yerrup=self.CI[i][1]-self.yp[i]
                ax.errorbar(i,self.yp[i], yerr=[[yerrdown],[yerrup]],color = self.colors(i), ecolor = self.colors(i))
        pt.removeTopRightAxes(ax)   
        ax.set_ylim([-0.05,1.05])
        ax.set_xlim([-0.1, len(self.yp)-0.9])
        ax.set_xticks(range(len(self.yp)))
        ax.set_xticklabels(range(1,len(self.yp)+1))
        l1[0].set_linewidth(2)
        l1[0].set_color('black')
        # Set label
        ax.set_ylabel('Yes \n probability')
        ax.set_xlabel('Objects')
        return ax
                  
      
# ----- ONE-TIME COMPUTATIONS/DEFINITIONS ----- #  
    def _makehitrate_df(self):
        self.hr_df = compute_hitrateDF(self.df)

    def _makeTerrainTable(self):
        """ Makes a table of all possible terrain object parameters (self.terraintable) 
        and assigns an integer code to each possible object (self.terraincode)."""
        dimnames = []
        dimlist = []
        dimcorrect =  []
        for parameter in self.terrain['params']:
            dimnames.append(parameter['name'])
            dimlist.append(parameter['possible'])
            dimcorrect.append(parameter['correct'])
        self.terraintablecorrect = list(itertools.product(*dimcorrect))
        self.terraintable = list(itertools.product(*dimlist))
        self.terraincode = range(len(self.terraintable))
        terraincode_target = []
        terraincode_distractor = []
        for code,parameters in enumerate(self.terraintable):
            if parameters in self.terraintablecorrect:
                terraincode_target.append(code)
            else:
                terraincode_distractor.append(code)
        self.terraincode_target = terraincode_target
        self.terraincode_distractor = terraincode_distractor
                        
    def _getEncounterOrder(self):
        """ Makes a sequential list of objects encountered. The object is coded according
        to it's index in the terraintable. """
        encounterorder = []
        for parameters in self.terrainlog:
            try:
                encounterorder.append(self.terraintable.index(tuple(parameters)))
            except Exception,e:
                print e
                encounterorder.append(int(0))
        self.encounterorder = encounterorder[:-1] # TODO: Deal with reward limit termination
        
    def _checkterrainlog(self):
        """Check is terrainlog contains parameters defined by strings such as images.
        """
        paramtype = [isinstance(x,str) for x in self.terraintable[0]]
        if any(paramtype):
            df = pd.DataFrame(self.terrainlog)
            for i,v in enumerate(paramtype):
                if v:
                    for ind,strval in enumerate(df[i].unique()):
                        df[i][df[i]==strval] = ind
                        # Update terraintable
                        tmp = self.terraintable[ind]
                        tmp = [x for x in tmp]
                        tmp[i] = ind
                        self.terraintable[ind] = tuple(tmp)
                # Update terrainlog
                tmp = [list(x) for x in df.itertuples()]
                self.terrainlog = [x[1:] for x in tmp]
        
    def _computeUpdateTimes(self):
        """ Makes a list, t, of update times in units of seconds based on vsyncintervals.
        The first t value is set to zero (t[0]=0)."""
        t = [0]
        for i,v in enumerate(self.data['vsyncintervals']):
            t.append((t[i]+v))
        if len(t) != len(self.posx):
            t.append(np.nan) # TODO: Why does t have fewer values than posx???
        self.t = np.array(t)/1000.0  # TODO: changes variable from t to something else.
    
    def _getEncounterTimes(self):
        """ Makes a list of encounter times, where the t[n] is the time the nth lap began.
        Output is in units of seconds. """
        encountertimes = []
        for i in range(len(self.encounterorder)):
            lapstart = self.getTrajectoryBounds(i)[0]
            try:
                encountertimes.append(self.t[lapstart])
            except Exception,e: # TODO: What's happening here?
                print "Problem with lapstart index!!"                
                print e
                encountertimes.append(self.t[lapstart-1])
        self.encountertimes = np.array(encountertimes)
    
    def _getPauseTimes(self):
        """ Makes a list of pause times for each object encountered (units = seconds)"""
        # TODO: Consider storing pausetime during operation of foraging program.
        # Also, consider computing number of frames instead of time because this is what is used in foraging program.
        pausetimes = [] 
        for i,v in enumerate(self.encounterorder):
#            print i, "of%s"%len(self.encounterorder)
            x,t = self.getTrajectory(i)
            x = np.array(x)
            t = np.array(t)
            rw = [-self.terrain['windowwidth'], self.terrain['windowwidth']]   # TODO: Will need to change when reward window is made more flexible
            in_rw = np.logical_and(x >= rw[0], x < rw[1]) 
            crossing = np.diff(in_rw)
            idx = crossing.nonzero()[0] 
            if len(idx) != 0:
                idx += 1
                if in_rw[0]:
                    # If the start of condition is True prepend a 0
                    idx = np.r_[0, idx]
                if in_rw[-1]:
                    # If the end of condition is True, append the length of the array
                    idx = np.r_[idx, in_rw.size] # Edit
                idx.shape = (-1,2)
                # Find index of max contiguous region (if multple, first occurrence returned)
                imax = np.argmax(np.diff(idx))
                pauseInd = idx[imax]
                try:
                    if pauseInd[1] < len(t):
                        pauseTime = t[pauseInd[1]+1] - t[pauseInd[0]]
                    else:
                        pauseTime = np.nan
                except:
                    pauseTime = np.nan
                pausetimes.append(pauseTime)
            else:
                pausetimes.append(np.nan)
        self.pausetimes= pausetimes
        
    def _computecmPerPixel(self):
        dd = []
        for i in range(10,20,1):
            x = self.getTrajectory(i)[0]
            d_x = x[-1]-x[0]
            x_cm = self.getTrajectory(i,units="cm")[0]
            d_xcm = x_cm[-1]-x_cm[0]
            cmPerPixels = d_xcm/d_x
            dd.append(cmPerPixels)
        cmPerPixels = np.median(cmPerPixels)
        self.cmPerPixel = cmPerPixels
               
    def _computeAdjLapDistance(self):
        laps = self.laps
        posx = self.posx
        lapstartframe = [x[1] for x in laps]
        lapstartframe = np.array(lapstartframe,dtype=int)
        lapstartframe = np.insert(lapstartframe,int(0),0)
        lapbounds = [(lapstartframe[i],lapstartframe[i+1]-1) for i,val in enumerate(lapstartframe[:-1])]
        lapdistance = [posx[x[1]]-posx[x[0]] for x in lapbounds]
        lapdistance = [0] + list(lapdistance)
        ld = [posx[i[1]-1]+1100 for i in lapbounds]
        ld = [0] + ld
        return ld   

    def getSelectionTime(self):
        selectiontime = []
        for lap in self.laps:        
            if 'staircase' in self.data.keys():
                tmp = self.data['staircase'].log[lap]
            else:
                tmp = self.data['terrain']['selectiontime']  
            selectiontime.append(tmp)
        return selectiontime
                    
    def getIsTargetArray(self):
        istarget = []
        for lap in range(len(self.laps)):        
            tmp = bool(tuple(self.data['terrainlog'][lap]) in self.terraintablecorrect)
            istarget.append(tmp)
        istarget = np.array(istarget)
        return istarget
    
    def maketrialsummary(self,columns=None,timerange=None):
        """Make list of tuples for various trial by trial values, properties, and metrics.\n
        Returns Data and columnsnames which can be directly used to make a DataFrame."""
        # Make an array with columns representing the following:
        data = []
        # TODO: make timestart and limit functional
        self.cutofftimemin = 0
        self.cutofftimemax = 100000
        timestart = self.cutofftimemin #minutes at which to start data export    
        timelimit = self.cutofftimemax #minutes at which to cap data export
        
        terraintablenames = [x['name'] for x in self.terrain['params']]
        terraintablenames.reverse()
        columnnames = ['trial','time','terraincode','pausetime','choice','selectiontime','istarget','speed','lapdistance','timebeforezone','slowing_f']
        columnnames = terraintablenames + columnnames
        # Compute adjusted lapdistance
        adjLD = self._computeAdjLapDistance()
        
        # Compute avg speed
        avgspeed = self.computeAvgSpeed()
        
        # Compute anticipatory slowing
        baseline = []
        w_baseline = [-1100,-850]
        prezone = []
        w_prezone = [-500,-250]
        for i in range(len(self.encounterorder)):
            try:
                x,t = self.getTrajectory(encounterindex=i)
                # baseline period
                windowIndex = np.logical_and(x >= w_baseline[0], x < w_baseline[1])
                tmp = t[windowIndex]
                if np.size(tmp) != 0:
                    baseline.append((tmp[-1]-tmp[0]))
                else:
                    baseline.append(np.nan) # didn't reach window on this lap? 
                # pre reward zone period
                windowIndex = np.logical_and(x >= w_prezone[0], x < w_prezone[1])
                tmp = t[windowIndex]
                if np.size(tmp) != 0:
                    prezone.append((tmp[-1]-tmp[0]))
                else:
                    prezone.append(np.nan) # didn't reach window on this lap?                            
            except:
                baseline.append(np.nan) # didn't reach window on this lap? 
                prezone.append(np.nan)

        # Compute relative slowing factor
        slowing_f = np.divide(np.array(prezone),np.array(baseline))

        for lap in range(0,len(self.laps[np.logical_and(self.laps[:,0]>=(timestart*60.0),self.laps[:,0]<=(timelimit*60.0))])):
            if 'staircase' in self.data.keys():
                if self.data['staircase'] is not None:
                    selection_time = self.data['staircase'].log[lap]
                else:
                    selection_time = self.data['terrain']['selectiontime']
            else:
                selection_time = self.data['terrain']['selectiontime']
            istarget = bool(tuple(self.data['terrainlog'][lap]) in self.terraintablecorrect)
            # Make tuple
            tmp = (lap,\
                self.laps[lap-1][0]-self.data['starttime'],\
                self.encounterorder[lap],\
                self.pausetimes[lap],\
                np.bool(self.yesChoiceArray(self.pausetimes[lap])),\
                selection_time,\
                istarget,\
                avgspeed[lap],\
                adjLD[lap],\
                baseline[lap],\
                slowing_f[lap])

            for param in terraintablenames:
                tmp = (self.terrainlog[lap][terraintablenames.index(param)],) + tmp            
            # Append tuple to data list
            data.append(tmp)
        # How to deal with all the various possible columns to include            
        return data, columnnames


def binomialCI(successes,attempts,alpha):
    """ Calculates the upper and lower confidence intervals on binomial data using the Clopper Pearson method 
    
        Added by Doug Ollerenshaw on 02/12/2014    
    
        input:
                successes = number of successes
                attempts = number of attempts
                alpha = confidence range (.e.g., 0.05 to return the 95% confidence interval)
                
        output:
                lower bound, upper bound
                
        Refs:   
                [1] Clopper, C. and Pearson, S. The use of confidence or fiducial limits illustrated in the case of the Binomial. Biometrika 26: 404-413, 1934
                [2] http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
                [3] http://www.danielsoper.com/statcalc3/calc.aspx?id=85 [an online calculator used to validate the output of this function]
        
        """
        
    from scipy.stats import beta
    import math
    
    x = successes
    n = attempts    
    
    # NOTE: the ppf (percent point function) is equivalent to the inverse CDF
    lower = beta.ppf(alpha/2,x,n-x+1)
    if math.isnan(lower):
        lower = 0
    upper = beta.ppf(1-alpha/2,x+1,n-x)
    if math.isnan(upper):
        upper = 1
    
    return lower,upper


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    tloadMouse=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2-1):-(window_len/2)]

# -- MAIN --- #

if __name__ == "__main__":
    path = r"/Users/Shawn/Dropbox/Examples/Forage Class/140522144659-140522134648-M130587.pkl"
    fg = Forage.ForageSession(path)



    
