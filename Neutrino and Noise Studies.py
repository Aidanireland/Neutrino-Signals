#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np #Required packages in order to run the code
import matplotlib.pyplot as plt
import pandas as pd
import scipy as scipy
from scipy import optimize
from scipy import interpolate
from scipy.stats import linregress
import math
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy import signal
from importlib import reload


# In[22]:


# signal amplitude in ENC
SIGNAL = 1e4
# ENC noise levels
ENC_Y = 200
ENC_U = 500
ENC_V = 600
# shaping time
SHAPING = 2 # us
# sampling rate
TDC = 0.5 # us
# time-ticks for entire waveform
TICKS = 100
sampling_freq=2e6 #Rate at which wire signals are sampled
sampling_time=1/sampling_freq #Time between samples
totaltimeASIC=4.8e-3#Total time window of ADC
#MAX_BIT_RATE=#This average rate cannot be exceeded or it will overwhelm computing time
MAX_BIT=76#ADC read-out bits
WIRES=482720#Total number of wires in DUNE
MAX_RATE=(100e9/WIRES)#Maximum data rate allowable in Dune
argon_wire_rate=407.326951399 #Total number of argon decays seen each second by each wire
electron_conversion=24690#24690 is conversion from MeV to electrons


# In[23]:


ASICBlue=np.transpose(np.loadtxt("ASICBlue.csv", delimiter=","))
ASICSpline=scipy.interpolate.CubicSpline(ASICBlue[0],ASICBlue[1])
fRange=np.linspace(0,sampling_freq/2000,int(totaltimeASIC*sampling_freq))#Frequencies which can be seen up to Nyquist limit
MEAN_NOISE_AMPLITUDES=ASICSpline(fRange)#Average noise amplitudes in frequency domain which are used to generate noise
time=np.linspace(0,totaltimeASIC,int(totaltimeASIC*sampling_freq))#Converted to time domain

RADAR_SPECTRUM=np.transpose(np.loadtxt("Argon-39 Radar Beta Spectrum", delimiter=","))
RADAR_ENERGY=RADAR_SPECTRUM[0]
RADAR_AMP=RADAR_SPECTRUM[1]
ARGONSpline=scipy.interpolate.CubicSpline(RADAR_ENERGY,RADAR_AMP)#Interpolating probability distribution
ARGON_SPEC=ARGONSpline(np.linspace(RADAR_ENERGY[0],RADAR_ENERGY[-1],10000))
ARGON_max=np.max(ARGON_SPEC)#Maximum probability of spectrum of Argon Decay Energies 
     
BEGIN_DOMAIN=RADAR_ENERGY[0]#Important for forming boundaries in which to reject and accept samples
END_DOMAIN=RADAR_ENERGY[-1]


# In[24]:


def purity(triggers,correct_triggers):#Used to calculate purity of triggered signals
    if triggers != 0:
        return correct_triggers/triggers
    else:
        return np.nan


# In[25]:


def rejection_sampling_val():#Performs rejection sampling in order to properly sample argon decay energy spectrum
    production=0
    while production!=1:
        y_rand=np.random.uniform(0,1)
        y_rand=y_rand*ARGON_max
        x_rand=np.random.uniform(BEGIN_DOMAIN,END_DOMAIN)
        ytest=ARGONSpline(x_rand)
        
        if(y_rand<=ytest):
            production=1
    return x_rand


# In[26]:


def Argon_Waveform(mu=0,sigma=0):#This is an Argon-39 decay waveform which mimics that of a supernova neutrino
    offset = np.random.uniform(-TDC/2.,TDC/2.)#Random offset of wave arrival
    number_of_electrons=electron_conversion*rejection_sampling_val()#Here we take a decay energy and convert it into total detected charge
    Argon = np.zeros(TICKS)
    
    for tick in range(TICKS):#Creating our Argon-39 waveform
        Argon[tick] = GAUSS((tick*TDC)+offset, number_of_electrons, (TICKS/2)*TDC, SHAPING )
    return Argon


# In[27]:


def FreqNoise(amplitude,steps,phase):#Here we are generating frequency noise by randomly sampling a Rayleigh distribution
    sigma_square=0.5*steps*np.abs(amplitude)**2#This is our standard deviation squared
    sigma=np.sqrt(sigma_square)#Standard deviation
    a=np.random.uniform(0,1)#This value will be used to map the range of the cdf to the domain. The inverse does infact exist
    r=sigma*np.sqrt(-2*np.log(1-a))#Calculating our domain value
    z=(r/(sigma_square))*np.exp(-r**2/(2*sigma_square))#Returning our distribution value
    phase=np.exp(-1j*phase)#Adding a complex phase
    return z*phase#Our value


# In[28]:


def noiseProper(rms,amplitudes=MEAN_NOISE_AMPLITUDES,steps=1000,ticks=9600):#This is how we will generate our noise spectrum

    #Amplitudes will be the mean noise amplitudes calculated
    #Steps must be a large number for accuracy 
    #rms is that of sense wires
    #ticks is size of noisy waveform. It cannot exceed 9600.
    noise=np.zeros(len(amplitudes))
    scale=(rms-0.00043261938840366554)/(0.013341763970734713)#This is how we scale our distribution according to the rms of adc's
    count=0
    while count<len(amplitudes):#Generating our frequnecy noise
        noise[count]=scale*FreqNoise(amplitudes[count],steps,np.random.uniform(0,2*np.pi))#Use mean amplitudes to generate spectrum
        count=count+1
    noise=np.fft.ifft(noise*len(noise))#Converting it to the time domain
    return 182*noise[0:ticks]#Returning noise array of size <=9600 and appropriately scaling adc magnitude to charge.


# In[29]:


def sigmaLength(length,variables):
    return l*variables[0]+variables[1]


# In[30]:


def digitize(waveform):
    return np.round(waveform,0)


# In[31]:


def noise(ENC,ticks):#Original noise model
    return np.random.normal(0,ENC,ticks)


# In[32]:


def GAUSS(x,A,mu,sigma):#This is used to produce gaussians for supernova neutrinos and Argon-39 decays
    return A * (1./(sigma*math.sqrt(2*math.pi))) * math.exp(-0.5*(((x-mu)/sigma)**2))


# In[33]:


def Insert_Function_mod(child,parent,clen,plen):#Takes two functions and inserts the child within the parent by elements
    insert_point=int(np.random.uniform(0-clen,plen))#Randomly picking a position to insert the child into
    region=np.array([insert_point,insert_point+clen])                                 #This has to be broken up into 3 cases to avoid errors
    if insert_point<0:#Inserting outside the leftmost boundary of the array, we will truncate the child and add it
        diff=0-insert_point
        child=child[diff:clen]
        for X in range(0,clen-diff):
            parent[X]=parent[X]+child[X]
        #print("A") 
        return parent,region
        
        
    if insert_point+clen>plen:#Going outside the rightmost boundary of array we need to truncate and add it
        diff=insert_point+clen-plen
        child=child[0:clen-diff]
        
        for X in range(insert_point,plen):
            parent[X]=parent[X]+child[X-insert_point]
        #print("B") 
        return parent,region
    
    for X in range(insert_point,insert_point+clen):#No issues if it is within bounds
        parent[X]=parent[X]+child[X-insert_point]

        #print("C")
    return parent,region


# In[34]:


def signal(number_of_electrons,totalticks=TICKS):
    # assume signal arrives at time-tick 250
    # but inject random offset 
    # sampling uniformly for one time-tick
    offset = np.random.uniform(-TDC/2.,TDC/2.)
    ADCs = np.zeros(totalticks)#This is our waveform
    for tick in range(totalticks):
        ADCs[tick] = GAUSS( (tick*TDC)+offset, number_of_electrons, (totalticks/2)*TDC, SHAPING )
    return ADCs


# In[35]:


def Modified_Argon_Inject(start,ticks,events,waveform_size):#This will add Argon background a set number of times
    #Start is at which tick the waveform will begin being made in order to cut down on filling array with numbers which are nearly 0
    #Ticks is number of ticks which will have simulated data added to them starting at Start
    #Events is number of argon decays
    #Waveform size is the size of the signal waveform
    
    radiogenic=np.zeros(waveform_size)#Generating array that we will fill with background
    insert_points=np.full((events,2),np.nan)#Where waveforms will be added. Tracked for later calculations.
    electron_count=np.zeros(events)#Tracking number of electrons produced by each Argon Event.
    for N in range(events):
        argon_decay=np.zeros(ticks)
        electron_amount=electron_conversion*rejection_sampling_val()
        electron_count[N]=electron_amount
        for M in range(start,start+ticks):
            argon_decay[M-start] = GAUSS((M*TDC), electron_amount, ((2*start+ticks)/2)*TDC, SHAPING )
        
        radiogenic,insert_points[N]=Insert_Function_mod(argon_decay,radiogenic,ticks,waveform_size)
    
    idx=np.argsort(insert_points[:, 0])
    
    insert_points=insert_points[idx]
    electron_count=electron_count[idx]
    
    return radiogenic, insert_points,electron_count


# In[36]:


def Poisson_Weight(disintergration_rate,total_ticks):#This will give us a poisson weight in order to get results with background
    time=total_ticks*sampling_time
    Lambda=time*disintergration_rate
    prob_zero=1/np.exp(Lambda)
    weight=1-prob_zero#Chance of any number which is not 0
    return weight


# In[37]:


def butter_lowpass_filter(data, cutoff, order,ticks):#Lowpass filter for the data
#Data wll be the noisy data, then there is cutoff frequency, order is polynomial used for continous transition after
#removal of frequency, and ticks is total time ticks of ADC
    T =  ticks*sampling_time        
    fs = sampling_freq      
    nyq = 0.5 * fs
    n = int(T * fs)
    
    

    data2=np.zeros(len(data))
    
    for n in range(len(data)):
        data2[n]=data[n]
    
    
    
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy.signal.filtfilt(b, a, data2)
    return y


# In[40]:


def segment_search_new(wave,insertions,electron_count,interval,threshold):#Will search for signals within discrete chunks of the wavefrom
    #insertions and electron_count is to be provided after running Modified_Argon Inject
    #wave is the input waveform, interval is size of each division, and threshold is triggering voltage
    #Interval must divide the size of waveform or else it will not work
    ctrig=0
    trig=0
    s1=len(wave)
    s2=len(insertions)
    if (s1%interval>0):
        print("Interval must divide size of waveform")
        return
    
    
    segments=int(s1/interval)
    index=np.arange(s2)
    
    bad_locations=np.full((s2,2),np.nan)
    recorded_values=np.array([])
    
    begin=0
    end=interval
    triglimit=0
    
    wave=np.real(wave)
    wave=wave.astype(int)
    index=0
    bad_index=0
    stop=0
    
    ic=0
    
    while end<s1:
        triglimit=0

        for n in range(begin,end):
            stop=0
            ic=0
            
            if wave[n]>=threshold:
                
                for bounds in bad_locations:
                    if (n>=bounds[0]-1) and n<=(bounds[1]-1):
                        stop=1
                        break
                if stop==0:
                    
                    while ic<len(insertions) and len(insertions)>0:
                        if n>=(insertions[ic][0]-1) and n<=(insertions[ic][1]-1):
                            if ctrig<=s2:
                                ctrig=ctrig+1
                                recorded_values=np.append(recorded_values,electron_count[ic])
                            
                            if trig<=segments+s2:
                                trig=trig+1
                                triglimit=1
                            
                            bad_locations[bad_index]=insertions[ic]
                            bad_index=bad_index+1
                            insertions=np.delete(insertions,0,axis=0)
                            electron_count=np.delete(electron_count,ic)
                            break
                        ic=ic+1
    
                    if triglimit==0 and trig<=segments+s2:
                        trig=trig+1
                        triglimit=1
    
    
        begin=end
        end=end+interval
    return ctrig,trig,recorded_values
        


# In[19]:


def studies_segment_search(threshold,waveform_size,sims,rms,start,ticks,interval):#Takes in a triggering threshod,
#signal waveform size, number of simulations, rms of noise, interval for divsision of data, and a start and tick length for
#the shape of argon waveforms to be created
    begin=0
    end=waveform_size
    total_events=0
    full_electron_simulations=np.array([])
    
    tN=0#Tracking statistics for the different filtering methods
    tS=0# S is savgol, L is lowpass, and LS is combined
    tL=0
    tLS=0
    
    cN=0
    cS=0
    cL=0
    cLS=0
    total_electrons=0
    time=waveform_size*sampling_time#Total time of  a waveform
    Lambda=time*argon_wire_rate#This rate will be used in our poisson distribution
    for N in range(sims):
        
        events=0
        events=np.random.poisson(Lambda)
        total_electrons=events+total_electrons
        
        if begin==0 or end>9600:#Making noise in time domain to be added
            noise_total=noiseProper(rms,MEAN_NOISE_AMPLITUDES,1000,9600)
            begin=0
            end=waveform_size
        
        noise=noise_total[begin:end]
        
        radiogenic,regions,electron_count=Modified_Argon_Inject(start,ticks,events,waveform_size)#Makes the radiogenic background to be added to noise
        
        Norm=noise+radiogenic#Combines them together to make the complete signal
        a,b,electrons_measured=segment_search_new(Norm,regions,electron_count,interval,threshold)
        tN=tN+b
        cN=cN+a
    
        Sav=noise+radiogenic    
        Sav=scipy.signal.savgol_filter(Sav,window_length=17,polyorder=2)
        a,b,electrons_measured=segment_search_new(Sav,regions,electron_count,interval,threshold)
        tS=tS+b
        cS=cS+a
        
        Low=noise+radiogenic
        Low=butter_lowpass_filter(Low,2e5,2,waveform_size)
        a,b,electrons_measured=segment_search_new(Low,regions,electron_count,interval,threshold)
        tL=tL+b
        cL=cL+a
        
        LowSav=noise+radiogenic
        LowSav=butter_lowpass_filter(scipy.signal.savgol_filter(LowSav,window_length=17,polyorder=2),2e5,2,waveform_size)
        a,b,electrons_measured=segment_search_new(LowSav,regions,electron_count,interval,threshold)
        tLS=tLS+b
        cLS=cLS+a
    
    trN=cN/total_electrons
    trS=cS/total_electrons
    trL=cS/total_electrons
    trLS=cLS/total_electrons
    
    pN=purity(tN,cN)
    pL=purity(tL,cL)
    pS=purity(tS,cS)
    pLS=purity(tLS,cLS)
    purities=np.array([pN,pS,pL,pLS])
    true_positives=np.array([trN,trS,trL,trLS])
    
    print(purities)
    print(true_positives)
    return purities,true_positives


# In[ ]:




