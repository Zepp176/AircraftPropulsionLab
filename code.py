import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% ======= Data =======

data = np.genfromtxt('Labo_Result_2021_12_02_10_17.txt', delimiter='\t')[230:]

t = data[:,1] + data[:,2]/60
t -= t[0]
ps2 = data[:,3]
ps3 = data[:,4]
pt3 = data[:,5]
pt4 = data[:,6]
pt5 = data[:,7]
Tt3 = data[:,8]
Tt4 = data[:,9]
Tt5 = data[:,10]
Tt6 = data[:,12]
Thrust = data[:,14]
Thrust -= Thrust[0]
RPM = data[:,15]
m_f = data[:,19]
m_f -= m_f[0]

#%% ======= Plots =======

plt.figure(figsize=(7, 4))

plt.plot(t, ps2)
plt.plot(t, ps3)
plt.plot(t, pt3)
plt.plot(t, pt4)
plt.plot(t, pt5)

plt.grid()
plt.title('Static and total pressures')
plt.legend(['ps2', 'ps3', 'pt3', 'pt4', 'pt5'])
plt.xlabel('Time [minute]')
plt.ylabel('Pressure [bar]')
plt.savefig('fig_pressure.png', dpi=300)

#%%

plt.figure(figsize=(7, 4))

plt.plot(t, Tt3)
plt.plot(t, Tt4)
plt.plot(t, Tt5)
plt.plot(t, Tt6)

plt.grid()
plt.title('Total temperatures')
plt.legend(['Tt3', 'Tt4', 'Tt5', 'Tt6'])
plt.xlabel('Time [minute]')
plt.ylabel('Temperature [°C]')
plt.savefig('fig_temperature.png', dpi=300)

#%%

plt.figure(figsize=(7, 4))

plt.plot(t, Thrust, 'k')

plt.grid()
plt.title('Thrust')
plt.xlabel('Time [minute]')
plt.ylabel('T [N]')
plt.savefig('fig_thrust.png', dpi=300)

#%%

plt.figure(figsize=(7, 4))

plt.plot(t, RPM/1e3, 'k')

plt.grid()
plt.title('Revolutions per minute')
plt.xlabel('Time [minute]')
plt.ylabel('RPM x 1000')
plt.savefig('fig_RPM.png', dpi=300)

#%%

plt.figure(figsize=(7, 4))

plt.plot(t, m_f, 'k')

plt.grid()
plt.title('Fuel consumption')
plt.xlabel('Time [minute]')
plt.ylabel('$\dot m_f$ [g/s]')
plt.savefig('fig_m_f.png', dpi=300)






