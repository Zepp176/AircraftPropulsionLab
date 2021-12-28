import numpy as np
import matplotlib.pyplot as plt

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

LHV = 43.7e6
A1 = 5.168e-3
A2 = 2.818e-3
A3 = 4.112e-3
A4 = 6.323e-3
A5 = 3.519e-3
A6 = 3.318e-3
R = 287.1
gamma = 1.4
cp = gamma*R/(gamma-1)

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

plt.plot(RPMi/1000, ps3i/ps2i, 'o-k')

plt.grid()
plt.title('Compression ratio depending on the shaft speed')
plt.xlabel('RPM x 1000')
plt.ylabel('$\\pi_c$ [-]')
plt.savefig('fig_compression_ratio.png', dpi=300)

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

#%% ======= Cycle Analysis =======

state_variables = np.empty((7, 7))  # 7 states and 7 variables
state_variables[:] = np.nan         # x coordinate is for the state number
                                    # y coordinate is for the variable
                                        # 0: ps
                                        # 1: pt
                                        # 2: T
                                        # 3: Tt
                                        # 4: u
                                        # 5: s
                                        # 6: m_dot

def print_states(var, to_print=range(7)):
    for i in to_print:
        print("\n=== State {} ===\n".format(i))
        print("ps{} = {:.4f} bar".format(i, var[i][0]/1e5))
        print("pt{} = {:.4f} bar".format(i, var[i][1]/1e5))
        print("T{} = {:.1f} K".format(i, var[i][2]))
        print("Tt{} = {:.1f} K".format(i, var[i][3]))
        print("u{} = {:.1f} m/s".format(i, var[i][4]))
        print("s{} = {:.2f} J/kgK".format(i, var[i][5]))
        print("m_dot{} = {:.4f} kg/m^3".format(i, var[i][6]))

def get_state_3(var):
    ps3 = var[3][0]
    pt3 = var[3][1]
    Tt3 = var[3][3]
    
    if (ps3 == np.nan or pt3 == np.nan or Tt3 == np.nan):
        print("Error: missing values")
        return
    
    M3 = np.sqrt( 2/(gamma-1) * ( (pt3/ps3)**((gamma-1)/gamma) - 1 ) )
    T3 = Tt3/(1+(gamma-1)/2*M3**2)
    u3 = M3*np.sqrt(gamma*R*T3)
    mdot3 = A3*u3*ps3/(R*T3)
    
    var[3][2] = T3
    var[3][4] = u3
    var[3][6] = mdot3
    
def get_state_2(var):
    ps2 = var[2][0]
    mdot = var[3][6]
    
    if (ps2 == np.nan or mdot == np.nan):
        print("Error: missing values")
        return
    
    T2 = var[3][2]*(ps2/var[3][0])**((gamma-1)/gamma)
    u2 = mdot*R*T2/(A2*ps2)
    M2 = u2/np.sqrt(gamma*R*T2)
    Tt2 = T2*(1+(gamma-1)/2*M2**2)
    pt2 = ps2*(1+(gamma-1)/2*M2**2)**(gamma/(gamma-1))
    
    var[2][1] = pt2
    var[2][2] = T2
    var[2][3] = Tt2
    var[2][4] = u2
    var[2][6] = mdot
    
def get_state_4(var, fuel_mass_rate):
    pt4 = var[4][1]
    Tt4 = var[4][3]
    mdot = var[3][6] + fuel_mass_rate
    
    if (np.isnan(pt4) or np.isnan(Tt4) or np.isnan(mdot)):
        print("Error: missing values")
        return
    
    # ========= TODO =========
    
regime = 1

state_variables[3][0] = ps3i[regime]*1e5
state_variables[3][1] = pt3i[regime]*1e5
state_variables[3][3] = Tt3i[regime]+273.15
state_variables[2][0] = ps2i[regime]*1e5
state_variables[4][1] = pt4i[regime]*1e5
state_variables[4][3] = Tt4i[regime]+273.15

get_state_3(state_variables)
get_state_2(state_variables)
get_state_4(state_variables, m_fi[regime])

print_states(state_variables, [3,2,4])














