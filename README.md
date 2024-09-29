# PINN inversión de la ecuación de onda
Esta PINN tiene dos propositos:
* Resolver la ecuación de onda en 2 dimensiones
* Encontrar el campo de velocidades a través del cual se propaga la onda

Esto se busca incluyendo la ecuación de onda en la función de pérdida, junto con condiciones de frontera absorbentes correspondientes a la frontera en el subsuelo y y condiciones de frontera libre para la superficie (Neumann iguales a cero).

Además también se incluyen los tiempos de arribo de la onda a la superficie en distintos puntos en la función de pérdida.

La ecuación de onda en dos dimensiones es:
    \begin{align}
    \frac{1}{c^2} \frac{\partial^2 u}{\partial t^2}= \frac{\partial ^2 u}{\partial x^2}+\frac{\partial ^2 u}{\partial z^2},  \ \ \ \ 0< t ,  \ \ \ 0 < x < 1, \ \ \ 0 < z < 1
    \end{align}
Con condición inicial:
    \begin{align} u(x,z,0)=e^{-\frac{x^2+z^2}{\sigma^2}}, \ \ \sigma = 0.06, \ \ 0 < x < 1, \ \ \ 0< z < 1 \end{align}
Y condiciones de frontera:
    \begin{align} \frac{\partial u}{\partial z}(x,0,t)=0, \frac{\partial u}{\partial t}(x,1,t)+c\frac{\partial u}{\partial z}(x,1,t)=0, \frac{\partial u}{\partial t}(0,z,t)+c\frac{\partial u}{\partial x}(0,z,t)=0, \frac{\partial u}{\partial t}(1,z,t)+c\frac{\partial u}{\partial x}(1,z,t)=0, \ \ t > 0 \end{align}
    
La condición de la frontera superior (z=0) es libre y el resto son absorbentes.
