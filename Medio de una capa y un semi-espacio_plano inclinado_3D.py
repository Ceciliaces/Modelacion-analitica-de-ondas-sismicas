# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:14:17 2024

@author: CESAR
"""
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Origen
xb0, yb0, zb0 =2, 3, 0
v1, v2 = 1.5, 3                        # Velocidad en km/s de la capa y el semiespacio
de = 0.3                               # Espacio entre puntos
L_x, L_y, P_z = 10, 10, 10             # Dimensiones del dominio horizontal y vertical
xb=np.arange(0,L_x+de,de)              # Dominio horizontal en x
yb=np.arange(0,L_y+de,de)              # Dominio horizontal en y
zb=np.arange(0,P_z+de,de)              # Dominio vertical
Yb, Xb, Zb = np.meshgrid(yb, xb, zb)   # Malla


bf=15                            # Profundidad de la interfaz medida desde el origen 
# Pendientes de la interfaz
mx=-1.5  
my=-1
Nz=len(zb)

# Profundidad en función de la distancia horizontal P(x)
def Distancia_al_Plano(mx, my, b, x, y):
    P=np.zeros(shape=(len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            P[i][j]=mx*x[i]+my*y[j]+b 
    P[(np.where(P<0))]=0
    P[(np.where(P>P_z))]=P_z
    return P   

Pf=Distancia_al_Plano(mx, my, bf, xb, yb) 

yg,xg = np.meshgrid(yb,xb)

# Grafica del plano de la interfaz
fig = plt.figure(11)
ax = plt.axes(projection="3d")
ax.set_xlabel('y [km]')
ax.set_ylabel('x [km]')
ax.set_zlabel('z [km]')
ax.invert_zaxis()
ax.view_init(elev=35, azim=45) 
ax.set_box_aspect([L_y,L_x,P_z], zoom=0.9)
ax.plot_surface(yg,xg,Pf)
plt.show()


# Transformación de coordenadas
N1=(np.sqrt(mx**2+my**2))**(-1)
N2=(np.sqrt(mx**2+my**2+1))**(-1)

# Transformación 1
x=-N1*my*xb+N1*mx*yb  #-N1*my*xb+N1*mx*zb
y=-N1*N2*mx*xb-N1*N2*my*yb-N1*N2*(my**2+mx**2)*zb  #-N1*N2*mx*xb-N1*N2*my*zb-N1*N2*(my**2+mx**2)*yb
z=-N2*mx*xb-N2*my*yb+N2*zb  #-N2*mx*xb-N2*my*zb+N2*yb

X=-N1*my*Xb+N1*mx*Yb  #-N1*my*Xb+N1*mx*Zb 
Y=-N1*N2*mx*Xb-N1*N2*my*Yb-N1*N2*(my**2+mx**2)*Zb #-N1*N2*mx*Xb-N1*N2*my*Zb-N1*N2*(my**2+mx**2)*Yb  
Z=-N2*mx*Xb-N2*my*Yb+N2*Zb #-N2*mx*Xb-N2*my*Zb+N2*Yb 

x0, y0, z0 = -N1*my*xb0+N1*mx*yb0, -N1*N2*mx*xb0-N1*N2*my*yb0-N1*N2*(my**2+mx**2)*zb0, -N2*mx*xb0-N2*my*yb0+N2*zb0 # -N1*my*xb0+N1*mx*zb0, -N1*N2*mx*xb0-N1*N2*my*zb0-N1*N2*(my**2+mx**2)*yb0, -N2*mx*xb0-N2*my*zb0+N2*yb0

P=bf/np.sqrt(mx**2+my**2+1)
a=N2*(mx*xb0+my*yb0+bf-zb0)

# Dirección de z dependiente de la posición de la fuente
if zb0>mx*xb0+my*yb0+bf:
    if bf<0:
        b=abs(Z-P)
    elif bf>0:
        z,Z,z0=-z,-Z,-z0
        a=-a
        b=abs(Z+P)
if zb0<=mx*xb0+my*yb0+bf:
    if bf<0:
        z,Z,z0=-z,-Z,-z0
        a=-a
        b=abs(Z+P)
    elif bf>0:
        b=abs(Z-P)
if mx==0 and my==0:
    a=bf 
    
cx=abs(X-x0)
cy=abs(Y-y0)
c=(cx**2+cy**2)**(0.5)
#########################

# Gráfica de los contornos con contour
T=b

kw = {
    'vmin': 0,
    'vmax': T.max(),
    'levels': np.linspace(0, int(T.max()+1), 100),
}

fig = plt.figure(12)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('y [km]')
ax.set_ylabel('x [km]')
ax.set_zlabel('z [km]')

ax.set_xticks(list(range(0,L_y,int(L_y/5))))
ax.set_yticks(list(range(0,L_x,int(L_x/5))))
ax.set_zticks(list(range(0,P_z,int(P_z/5))))

ax.axes.set_xlim3d(left=0, right=L_y) 
ax.axes.set_ylim3d(bottom=0, top=L_x) 
ax.axes.set_zlim3d(bottom=0, top=P_z) 

ax.invert_zaxis()

ax.view_init(elev=30, azim=45) 
ax.set_box_aspect([L_y,L_x,P_z], zoom=0.9)

_ = ax.contour(Yb[:, :, 0], Xb[:, :, 0], T[:, :, 0], zdir='z', offset=0, **kw, cmap='jet')
_ = ax.contour(Yb[0, :, :], T[-1, :, :], Zb[0, :, :], zdir='y', offset=Yb.max(), **kw, cmap='jet')
Cc = ax.contour(T[:, -1, :], Xb[:, 0, :], Zb[:, 0, :], zdir='x', offset=Xb.max(), **kw, cmap='jet')

ax.grid(visible=None)
plt.show()

################## Solución por Descartes 
A=v2**2-v1**2
B=-2*c*A
C=c**2*A-a**2*v1**2+b**2*v2**2
D=2*a**2*c*v1**2
E=-c**2*a**2*v1**2

P1=-B/(4*A)
P2=1/2*np.sqrt(B**2/(4*A**2) - 2*C/(3*A) + 2**(1/3)*(C**2 - 3*B*D + 12*A*E)/(3*A*(2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E + np.sqrt(-4*(C**2 - 3*B*D + 12*A*E)**3 + (2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E)**2))**(1/3)) + 1/(3*2**(1/3)*A)*(2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E + np.sqrt(-4*(C**2 - 3*B*D + 12*A*E)**3 + (2*C**3 - 9*B*C*D + 27*A*D**2 + 27*E*B**2 - 72*A*C*E)**2))**(1/3))                                 
P3=1/2*np.sqrt(B**2/(2*A**2) - 4*C/(3*A) - 2**(1/3)*(C**2 - 3*B*D + 12*A*E)/(3*A*(2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E + np.sqrt(-4*(C**2 - 3*B*D + 12*A*E)**3 + (2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E)**2))**(1/3)) - 1/(3*2**(1/3)*A)*(2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E + np.sqrt(-4*(C**2 - 3*B*D + 12*A*E)**3 + (2*C**3 - 9*B*C*D + 27*A*D**2 + 27*E*B**2 - 72*A*C*E)**2))**(1/3)-(-B**3/A**3 + 4*B*C/A**2 - 8*D/A)/(4*np.sqrt(B**2/(4*A**2) - 2*C/(3*A) + 2**(1/3)*(C**2 - 3*B*D + 12*A*E)/(3*A*((2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E + np.sqrt(-4*(C**2 - 3*B*D + 12*A*E)**3 + (2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E)**2)))**(1/3)) + 1/(3*2**(1/3)*A)*(2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E + np.sqrt(-4*(C**2 - 3*B*D + 12*A*E)**3 + (2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E)**2))**(1/3))))
P4=1/2*np.sqrt(B**2/(2*A**2) - 4*C/(3*A) - 2**(1/3)*(C**2 - 3*B*D + 12*A*E)/(3*A*(2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E + np.sqrt(-4*(C**2 - 3*B*D + 12*A*E)**3 + (2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E)**2))**(1/3)) - 1/(3*2**(1/3)*A)*(2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E + np.sqrt(-4*(C**2 - 3*B*D + 12*A*E)**3 + (2*C**3 - 9*B*C*D + 27*A*D**2 + 27*E*B**2 - 72*A*C*E)**2))**(1/3)+(-B**3/A**3 + 4*B*C/A**2 - 8*D/A)/(4*np.sqrt(B**2/(4*A**2) - 2*C/(3*A) + 2**(1/3)*(C**2 - 3*B*D + 12*A*E)/(3*A*((2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E + np.sqrt(-4*(C**2 - 3*B*D + 12*A*E)**3 + (2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E)**2)))**(1/3)) + 1/(3*2**(1/3)*A)*(2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E + np.sqrt(-4*(C**2 - 3*B*D + 12*A*E)**3 + (2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E)**2))**(1/3)))) 
                  
xr1=P1-P2-P3
xr2=P1-P2+P3
xr3=P1+P2-P4
xr4=P1+P2+P4

xr=np.zeros(shape=(len(x),len(y),len(z)),dtype='object')
for i in range(len(z)):
    for j in range(len(y)):
        for k in range(len(x)):
            a1=[xr1[k][j][i],xr2[k][j][i],xr3[k][j][i],xr4[k][j][i]]
            a1=np.array(a1)
            ind=np.where(a1>=0)
            if len(ind[0])!=0:
                xrp=a1[np.where(a1>=0)][0]
                xr[k][j][i]=xrp
            else:
                xr[k][j][i]=0
                
                
# Tiempo directo 3D
t1=np.sqrt((Xb-xb0)**2+(Yb-yb0)**2+(Zb-zb0)**2)/v1

# Tiempo refractado
t3=(xr**2+a**2)**(0.5)/v1+((c-xr)**2+b**2)**(0.5)/v2  # Tiempo refractado

theta_c=np.arcsin(v1/v2)
R=a*np.tan(theta_c) 
c2=b*np.tan(theta_c)
xc=(cx**2+cy**2)**(0.5)-R-c2 
# Tiempo refractado criticamente
t2=np.sqrt(R**2+a**2)/v1+xc/v2+np.sqrt(c2**2+b**2)/v1 



t4=np.empty(shape=(len(x),len(y),len(z)),dtype='object')
for i in range(len(z)):
    for j in range(len(y)):
        for k in range(len(x)):
            if i<int(Pf[k][j]/de): 
                t4[k][j][i]=t1[k][j][i]
                if c[k][j][i]>=R: 
                    t1_e=t1[k][j][i]
                    t2_e=t2[k][j][i] # Arriba del plano es el refractado criticamente
                    t4[k][j][i]=min(t1_e,t2_e)
            else:
                t4[k][j][i]=t3[k][j][i] # Debajo del plano es el refractado
        
T=np.reshape(t4,(len(z),len(y),len(x))) 
T=np.array(T, dtype=float)

################## Graficas
# En 3D
fig = plt.figure(1)
ax = plt.axes(projection="3d")
ax.set_xlabel('y (km)')
ax.set_ylabel('x (km)')
ax.set_zlabel('z (km)')
ax.set_title('Tiempos de arribo en un medio de dos \n capas separadas por un plano 3D',fontsize=14)
ax.invert_zaxis()
ax.view_init(elev=35, azim=45) 
ax.set_box_aspect([L_y,L_x,P_z])
cs=ax.scatter3D(Yb, Xb, Zb, c=T, alpha=0.7, marker='.', cmap='jet')
fig.colorbar(cs, shrink=0.8, aspect=10, label='Tiempo (s)') # ticks=cs.levels
plt.savefig('GrafMed_Plano_3D.png')
plt.show()

# Gráfica de todos los puntos con contourf

kw = {
    'vmin': 0,
    'vmax': T.max(),
    'levels': np.linspace(0, int(T.max()+1), 30),
}

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('y [km]')
ax.set_ylabel('x [km]')
ax.set_zlabel('z [km]')

ax.set_xticks(list(range(0,L_y,int(L_y/5))))
ax.set_yticks(list(range(0,L_x,int(L_x/5))))
ax.set_zticks(list(range(0,P_z,int(P_z/5))))

ax.axes.set_xlim3d(left=0, right=L_y) 
ax.axes.set_ylim3d(bottom=0, top=L_x) 
ax.axes.set_zlim3d(bottom=0, top=P_z) 

ax.invert_zaxis()

ax.view_init(elev=30, azim=45) 
ax.set_box_aspect([L_y,L_x,P_z])
ax.set_title('Tiempos de arribo en un medio de \n una capa y un semiespacio 3D',fontsize=14)

_ = ax.contourf(Yb[:, :, 0], Xb[:, :, 0], T[:, :, 0], zdir='z', offset=0, **kw, cmap='jet')
_ = ax.contourf(Yb[0, :, :], T[-1, :, :], Zb[0, :, :], zdir='y', offset=Xb.max(), **kw, cmap='jet')
C = ax.contourf(T[:, -1, :], Xb[:, 0, :], Zb[:, 0, :], zdir='x', offset=Yb.max(), **kw, cmap='jet')

fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Tiempo [s]') 
plt.savefig('GrafMed_Plano_3D_CF.png')
plt.show()

# Gráfica de los contornos con contour

kw = {
    'vmin': 0,
    'vmax': T.max(),
    'levels': np.linspace(0, int(T.max()+1), 30),
}

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')

ax.set_xticks(list(range(0,L_x,int(L_x/5))))
ax.set_yticks(list(range(0,L_y,int(L_y/5))))
ax.set_zticks(list(range(0,P_z,int(P_z/5))))

ax.axes.set_xlim3d(left=0, right=L_x) 
ax.axes.set_ylim3d(bottom=0, top=L_y) 
ax.axes.set_zlim3d(bottom=0, top=P_z) 

ax.invert_zaxis()

ax.view_init(elev=30, azim=45) 
ax.set_box_aspect([L_x,L_y,P_z], zoom=0.9)
ax.set_title('Tiempos de arribo en un medio de \n una capa y un semiespacio 3D',fontsize=14)

_ = ax.contour(Yb[:, :, 0], Xb[:, :, 0], T[:, :, 0], zdir='z', offset=0, **kw, cmap='jet')
_ = ax.contour(Yb[0, :, :], T[-1, :, :], Zb[0, :, :], zdir='y', offset=Xb.max(), **kw, cmap='jet')
Cc = ax.contour(T[:, -1, :], Xb[:, 0, :], Zb[:, 0, :], zdir='x', offset=Yb.max(), **kw, cmap='jet')

fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Tiempo [s]') 
ax.grid(visible=None)
plt.savefig('GrafMed_Plano_3D_C.png')
plt.show()