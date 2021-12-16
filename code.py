import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% ======= data =======

data_cont = np.genfromtxt('Labo_Result_2021_12_02_10_17.txt', delimiter='\t')[230:]
data_inst = np.genfromtxt('Labo_Result_2021_12_02_10_17SaveButton.txt', delimiter='\t')

t = data_cont[:,1] + data_cont[:,2]/60
t -= t[0]
ps2 = data_cont[:,3]
ps3 = data_cont[:,4]
pt3 = data_cont[:,5]
pt4 = data_cont[:,6]
pt5 = data_cont[:,7]
Tt3 = data_cont[:,8]
Tt4 = data_cont[:,9]
Tt5 = data_cont[:,10]
Tt6 = data_cont[:,12]
Thrust = data_cont[:,14]
Thrust_offset = Thrust[0]
Thrust -= Thrust_offset
RPM = data_cont[:,15]
m_f = data_cont[:,19]
m_f_offset = m_f[0]
m_f -= m_f_offset

borne = [20, 45, 75, 105, 125, 150, 175, 195]

ps2i = np.zeros(7)
ps3i = np.zeros(7)
pt3i = np.zeros(7)
pt4i = np.zeros(7)
pt5i = np.zeros(7)
Tt3i = np.zeros(7)
Tt4i = np.zeros(7)
Tt5i = np.zeros(7)
Tt6i = np.zeros(7)
Thrusti = np.zeros(7)
RPMi = np.zeros(7)
m_fi = np.zeros(7)

for j in range(len(borne)-1):
    for i in range(borne[j], borne[j+1]):
        ps2i[j] += data_inst[:,4][i]
        ps3i[j] += data_inst[:,5][i]
        pt3i[j] += data_inst[:,6][i]
        pt4i[j] += data_inst[:,7][i]
        pt5i[j] += data_inst[:,8][i]
        Tt3i[j] += data_inst[:,9][i]
        Tt4i[j] += data_inst[:,10][i]
        Tt5i[j] += data_inst[:,11][i]
        Tt6i[j] += data_inst[:,13][i]
        Thrusti[j] += data_inst[:,15][i]
        RPMi[j] += data_inst[:,16][i]
        m_fi[j] += data_inst[:,20][i]
    ps2i[j] /= (borne[j+1] - borne[j])
    ps3i[j] /= (borne[j+1] - borne[j])
    pt3i[j] /= (borne[j+1] - borne[j])
    pt4i[j] /= (borne[j+1] - borne[j])
    pt5i[j] /= (borne[j+1] - borne[j])
    Tt3i[j] /= (borne[j+1] - borne[j])
    Tt4i[j] /= (borne[j+1] - borne[j])
    Tt5i[j] /= (borne[j+1] - borne[j])
    Tt6i[j] /= (borne[j+1] - borne[j])
    Thrusti[j] /= (borne[j+1] - borne[j])
    RPMi[j] /= (borne[j+1] - borne[j])
    m_fi[j] /= (borne[j+1] - borne[j])

Thrusti -= Thrust_offset
m_fi -= m_f_offset

Thrusti

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

#%%

plt.figure(figsize=(7, 4))

plt.plot(RPMi/1000, ps2i, 'o-')
plt.plot(RPMi/1000, ps3i, 'o-')
plt.plot(RPMi/1000, pt3i, 'o-')
plt.plot(RPMi/1000, pt4i, 'o-')
plt.plot(RPMi/1000, pt5i, 'o-')

plt.grid()
plt.title('Static and total pressures depending on the shaft speed')
plt.legend(['ps2', 'ps3', 'pt3', 'pt4', 'pt5'])
plt.xlabel('RPM x 1000')
plt.ylabel('Pressure [bar]')
plt.savefig('fig_pressure_RPM.png', dpi=300)

#%%

plt.figure(figsize=(7, 4))

plt.plot(RPMi/1000, Tt3i, 'o-')
plt.plot(RPMi/1000, Tt4i, 'o-')
plt.plot(RPMi/1000, Tt5i, 'o-')
plt.plot(RPMi/1000, Tt6i, 'o-')

plt.grid()
plt.title('Total temperatures depending on the shaft speed')
plt.legend(['Tt3', 'Tt4', 'Tt5', 'Tt6'])
plt.xlabel('RPM x 1000')
plt.ylabel('Temperature [°C]')
plt.savefig('fig_temperature_RPM.png', dpi=300)

#%%

plt.figure(figsize=(7, 4))

plt.plot(RPMi/1000, Thrusti, 'ko-')

plt.grid()
plt.title('Thrust depending on the shaft speed')
plt.xlabel('RPM x 1000')
plt.ylabel('T [N]')
plt.savefig('fig_thrust_RPM.png', dpi=300)

#%%

plt.figure(figsize=(7, 4))

plt.plot(RPMi/1000, m_fi, 'ko-')

plt.grid()
plt.title('Fuel consumption depending on the shaft speed')
plt.xlabel('RPM x 1000')
plt.ylabel('$\dot m_f$ [g/s]')
plt.savefig('fig_m_f_RPM.png', dpi=300)




