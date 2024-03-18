import math
import random
import numpy as np

class InertiaFuc:
    """
    Clase que incluye las estrategias para el calculo del coeficiente del peso inercial. 
    """
    # ESTRATEGIAS NO ADAPTATIVAS Y VARIANTES EN EL TIEMPO PARA EL CONTROL DEL PESO INCERCIAL
    def RandomIW(inertia, *args, **kwargs):
        """
        Es una función que selecciona aleatoriamente el coeficiente de inercia 
        de modo que la inercia estará entre (0.5 , 1.0), exhibirá un comportamiento
        convergente cuando inertia < 0,7854 lo que es un 57.08% del tiempo.
        """
        inertia = 0.5 + random.uniform(0, 1)/2
        return inertia

    def LinealDecresingIW(inertia, n_iterations, i, Weightmin = 0.4, Weightmax = 0.9 ):
        """
        Es una fución propuesta por Shi y Eberhart (1998) se propuso como
        método para disminuir linealmente el peso de la inercia a lo largo
        del tiempo, basándose en el consenso general de que la exploración 
        se favorece al principio de un proceso de búsqueda, mientras que la 
        explotación se favorece más tarde.
        Esta función no hace uso del 'social_weight' y 'cognitive_weight'
            Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. In Proceedings of the 1998 IEEE
        international conference on evolutionary computation, (pp. 69–73).
        """
        inertia =  Weightmax - ((Weightmax - Weightmin)  * (i / n_iterations))
        return inertia

    def NoLinearIW(inertia, n_iterations, i,  Weightmin = 0.4, Weightmax = 0.9, alpha = 1/math.pi**2):
        """
        Es una función no-lineal, varible en el tiempo del peso incercial
        que demuestra un rendimiento superior a la lineal, donde alpha es 
        una constante dada por el usuario, usualmente 1/pi^2, exhibe un comportamiento convergente
            Yang, C., Gao, W., Liu, N., & Song, C. (2015). Low-discrepancy sequence initialized particle swarm opti
        mization algorithm with high-order nonlinear time-varying inertia weight. Applied Soft Computing, 29,
        386–394.
        """
        inertia = Weightmax - (Weightmax - Weightmin) * (i/n_iterations)**alpha
        return inertia

    def NonlinearImprovedIW(inertia, i, Unl =  1.0002, *args, **kwargs):
        """
        Es una función no lineal donde el parametro Unl ∈ [1.0001,1.005], 
        Se utiliza por defecto Unl=1.0002 y Se comienda inercia inicial = 0.3.
        Este algoritmo simpre exhibira un comportamiento convergente pero se 
        espera que sea prematuro
            Jiao, B., Lian, Z., & Gu, X. (2008). A dynamic inertia weight particle swarm optimization algorithm. Chaos,
        ractals, 37(3), 698–705.
        """
        inertia = inertia * Unl**(-i)
        return inertia

    def DecresingInertiaWeightIW(inertia, i, *args, **kwargs):
        """
        Es una función decreciente de Fan y Chiu (2007) que emplea una
        estrategia de peso de inercia dependiente del tiempo de modo que
        el coeficiente disminuye con el tiempo de forma no lineal, exhibirá
        un comportamiento convergente general después de que hayan pasado 
        cinco iteraciones.
        """
        inertia = (2/i)**(0.3)
        return inertia

    def ChaoticDescendingIW(inertia, n_iterations, i, Weightmin = 0.4, Weightmax = 0.9 ):
        """
        Es una función descendente caótica por Feng et al. (2007) que adopta 
        el uso de dinámica caótica para adaptar el peso de incercial a lo 
        largo del tiempo. Exhibirá un comportamiento mayormente convergente
        """
        def functZ(i):
            if i == 0:
                return random.uniform(0, 1)
            else:
                return 4 * functZ(i-1) * (1-functZ(i-1))
        inertia = functZ(i) *  Weightmin + (Weightmax - Weightmin) * (n_iterations - i)/n_iterations
        return inertia

    def NaturalExponentIW(inertia, n_iterations, i, Weightmin = 0.4, Weightmax = 0.9 ):
        """
        Es una función de exponente natural de Chen et al. (2006) que utiliza 
        un coeficiente inercial basado en un exponencial, exhibe un comportamiento
        convergente despues de un 2,6% de la busqueda.
        """
        inertia = Weightmin + (Weightmax - Weightmin)*math.e**(-(10*i)/n_iterations)
        return inertia

    def OscillatingIW(inertia, n_iterations, i, Weightmin = 0.3, Weightmax = 0.9, parK = 7):
        """
        Es una función oscilante de Kentzoglanakis y Poole (2009) que no disminuía
        monótonamente, sino que porporcionaba un peso de inercia oscilante durante
        la optimización.
        """
        if i < (3*n_iterations)/4:
            inertia = (Weightmin+Weightmax)/2 + (Weightmax-Weightmin)/2 * math.cos(2*math.pi * (4*parK+6))
        else:
            inertia = Weightmin
        return inertia

    def SugenoIW(inertia, n_iterations, i, parS = 2):
        """
        El algoritmo propuesta por Lei et al. (2006) emplea el uso de la función
        Sugeno que proporciona un peso de inercia monótonamente decreciente.
        """
        def beta(i):
            return i/n_iterations

        inertia = (1-beta(i))/(1+parS*beta(i))
        return inertia 

    def LogarithmDecreasingIW(inertia, n_iterations, i, Weightmin = 0.4, Weightmax = 0.9, parA = 1 ):
        """
        La función propuesta por Gao et al. (2008) emplea un peso de inercia 
        logarítmitmicamente decreciente, exhibirá un comportamiento convergente 
        cuando i/n_iterations > 0.02576 o solo despues del 2.6% de la busqueda.
        """
        inertia = Weightmax + (Weightmin - Weightmax) * math.log10(parA+(10*i)/n_iterations)
        return inertia
    
    def Personalization(funcion_parametro, *args, **kwargs):
        """
        Esta función permite personalizar cualquiera de las funciones de esta clase
        por ejemplo para NoLinearIW recibe como parametro Weightmin, Weightmax y alpha, 
        que podemos rescribir o para SugenoIW recibe un parametro parS que igual podemos 
        elegir el valor a tomar.
        """        
        # Definimos los nuevos parámetros (si se proporcionan)
        new_args = args
        new_kwargs = kwargs
        # Actualizamos los parámetros de la función_parametro con los nuevos valores, si es necesario
        
        if 'Weightmin' in kwargs:
            new_kwargs['Weightmin'] = kwargs['Weightmin']
        if 'Weightmax' in kwargs:
            new_kwargs['Weightmax'] = kwargs['Weightmax']
        if 'alpha' in kwargs:
            new_kwargs['alpha'] = kwargs['alpha']
        if 'beta' in kwargs:
            new_kwargs['beta'] = kwargs['beta']
        if 'eta' in kwargs:
            new_kwargs['eta'] = kwargs['eta']
        if 'initialweight' in kwargs:
            new_kwargs['initialweight'] = kwargs['initialweight']
        if 'Unl' in kwargs:
            new_kwargs['Unl'] = kwargs['Unl']
        if 'Weightinitial' in kwargs:
            new_kwargs['Weightinitial'] = kwargs['Weightinitial']
        if 'Weightfinal' in kwargs:
            new_kwargs['Weightfinal'] = kwargs['Weightfinal']
        if 'parK' in kwargs:
            new_kwargs['parK'] = kwargs['parK']
        if 'parS' in kwargs:
            new_kwargs['parS'] = kwargs['parS']
        if 'parA' in kwargs:
            new_kwargs['parA'] = kwargs['parA']
        # Devolvemos la función_parametro con los nuevos parámetros, si es necesario
        return lambda *new_args, **new_kwargs: funcion_parametro(*new_args, **new_kwargs)        ## Obtenemos la firma de la función original
        

    #    Ejemplo PARA NUESTRA PROPIA FUNCION INERCIAL
    #    def nombreIW(inertia, n_iterations, i):
    #        inertia = FUCTION
    #        return inertia

    #UTILS
    def euclidean_distance(point1, point2):
            """
            Calcula la distancia euclidiana entre dos puntos en un espacio de múltiples dimensiones.

            Argumentos:
            point1 (list): Lista que representa las coordenadas del primer punto.
            point2 (list): Lista que representa las coordenadas del segundo punto.

            Retorna:
            float: La distancia euclidiana entre los dos puntos.
            """
            if len(point1) != len(point2):
                raise ValueError("Los puntos deben tener la misma cantidad de dimensiones.")
            
            squared_diff = sum((coord1 - coord2) ** 2 for coord1, coord2 in zip(point1, point2))
            return math.sqrt(squared_diff)

    def pareto_dominance(solution_a, solution_b, maximize=True):
        """
        Verifica si solution_a domina a solution_b en un espacio de objetivos múltiples.

        Args:
        solution_a (list): Los valores de los objetivos para la solución A.
        solution_b (list): Los valores de los objetivos para la solución B.
        maximize (bool): Indica si se desea maximizar (True) o minimizar (False) los objetivos.

        Returns:
        bool: True si solution_a domina a solution_b, False en caso contrario.
        """
        if maximize:
            is_equal_or_better = all(a >= b for a, b in zip(solution_a, solution_b))
            is_better = any(a > b for a, b in zip(solution_a, solution_b))
        else:
            is_equal_or_better = all(a <= b for a, b in zip(solution_a, solution_b))
            is_better = any(a < b for a, b in zip(solution_a, solution_b))
        return is_equal_or_better and is_better
    
    def has_multiple_elements(arr):
        if isinstance(arr, np.ndarray):
            return arr.size > 1
        elif isinstance(arr, (list, tuple)):
            return len(arr) > 1
        else:
            return False
        
    # ESTRATEGIAS ADAPTATIVAS PARA EL CONTROL DEL PESO INERCIAL
    def SelfRegulatingIWA(particle, n_iterations, best_particle, i, eta = 1, Weightinitial = 0.9, Weightfinal = 0.4, *args, **kwargs):
        """
        El algoritmo de optimización de enjambre de partículas auto-reguladas (SRPSO) 
        de Tanweer et al. (2015) se basa en regular el peso de inercia de cada partícula
        de manera que se aumente para la mejor partícula mientras se reduce para todas 
        las demás. La justificación de este comportamiento es que la mejor partícula 
        debería tener un alto nivel de confianza en su dirección y, por lo tanto, 
        acelerar más rápido, mientras que el resto de las partículas deberían seguir 
        una estrategia de peso de inercia decreciente lineal similar a la de PSO-LDIW.
        
        Tanweer, M.R., Suresh, S., & Sundararajan, N. (2015).Self regulating particle swarm optimization algorithm. Information Sciences, 294, 182–202.
        
        Parameters
        ----------
        eta : `int`, optional
            Constante para controlar la velocidad de aceleración. (default is ``1``)
        Weightinitial : float``, optional 
            Valor inicial del peso inercial. (default is ``0.9``)
        Weightfinal : float``, optional 
            Valor inicial del peso inercial. (default is ``0.4``)
        """
        if i == 0:
            peso_anterior = Weightinitial
        else:
            peso_anterior = particle.inertiaParticle

        diferenciaPeso =  (Weightinitial - Weightfinal)/n_iterations
        if best_particle.value == particle.value:
            return peso_anterior + eta * diferenciaPeso
        else:
            return peso_anterior - diferenciaPeso
        
    def FineGrainedIWA(particle, n_iterations, best_particle, i, initialweight = 0.9, *args, **kwargs):
        """
        El peso de inercia de grano fino (FG-PSO) (Deep et al., 2011; Chauhan et al., 2013) 
        proporciona pesos de inercia individualizados para cada partícula utilizando la función
        de distancia euclidiana entre una partícula y el mejor global. Una nota importante 
        sobre la estrategia FG-PSO es que el peso de inercia nunca aumenta, por lo que siempre
        tenderá hacia 0.4. Este comportamiento probablemente causará que la estrategia FG-PSO 
        muestre un comportamiento convergente a largo plazo, ya que incluso distancias grandes
        del mejor global causarán una disminución no nula en el peso de inercia dado el 
        carácter asintótico del término exponencial. Sin embargo, se espera que el peso de 
        inercia disminuya bastante rápido, especialmente para la partícula de mejor global 
        (que tendrá una distancia de 0), y por lo tanto, se espera que la estrategia FG-PSO 
        sufra de convergencia prematura.

        Chauhan, P., Deep,K., & Pant, M. (2013).Novel inertia weight strategies for particle swarm optimization. Memetic Computing,5(3),229–251.

        Parameters
        ----------
        initialweight : `float`, optional
            Peso inercial inicial de la particula. (default is ``0.9``)
        """
        if i == 0:
            peso_anterior = initialweight
        else:
            peso_anterior = particle.inertiaParticle

        def Cfunc(val):
            return val * math.e**(-InertiaFuc.euclidean_distance(best_particle.position, particle.position)*(i/n_iterations))

        inertia = peso_anterior - Cfunc(peso_anterior - 0.4)
        return inertia
    
    def DoubleExponentialSelfAdaptiveIWA(particle, n_iterations, best_particle, i, initialweight = 0.9, *args, **kwargs):
        """
        El algoritmo de Chauhan et al. (2013) incorpora una función exponencial doble, 
        conocida como 'función Gompertz', para seleccionar el peso de inercia. El algoritmo 
        DE-PSO proporciona pesos de inercia más grandes cuando la distancia al mejor global
        es mayor. Por lo tanto, se espera que el peso de inercia proporcionado por el 
        algoritmo DE-PSO sea grande inicialmente debido al comportamiento errático de 
        las partículas no restringidas (Engelbrecht, 2013b), pero se espera que disminuya 
        con el tiempo a medida que las partículas se acerquen al mejor global. Por lo tanto, 
        el algoritmo DE-PSO exhibirá un comportamiento convergente.

        Engelbrecht, A. P. (2013b). Roaming behavior of unconstrained particles. In Proceedings of the 2013 BRICS congress on computational intelligence and 11th Brazilian congress on computational intelligence (pp.104–111).
        Parameters
        ----------
        initialweight : `float`, optional
            Peso inercial inicial de la particula. (default is ``0.9``)
        """
        def Rfunc():
            return InertiaFuc.euclidean_distance(best_particle.position, particle.position) * (n_iterations-i)/n_iterations 
        if i == 0:
            inertia = initialweight
        else:
            inertia = math.e**(-math.e**-Rfunc())
        return inertia
    
    def ImprovedIWA(particle, best_particle, alpha = 0.5, beta = 0.5, *args, **kwargs):
        """
        El algoritmo mejorado de optimización de enjambre de partículas por Li y Tan (2008)
        (IPSOLT) se basa en la idea de que el peso de inercia debe estar en relación directa
        con el factor de convergencia, así como con el factor de difusión que caracteriza el
        estado del algoritmo. Li y Tan (2008) no proporcionaron orientación para la selección 
        de los valores de los parámetros α y β, por lo que se utiliza el punto medio del rango 
        permitido α = β = 0.5. A medida que avanza la búsqueda, se espera que tanto el factor 
        de convergencia como el factor de difusión tiendan hacia 0 y, por lo tanto, el 
        algoritmo IPSO-LT debería exhibir un comportamiento globalmente convergente.

        Li, Z., & Tan, G. (2008). A Self-Adaptive Mutation-Particle Swarm Optimization Algorithm. 2008 Fourth International Conference on Natural Computation. doi:10.1109/icnc.2008.633 

        Parameters
        ----------
        alpha : `float`, optional
            Constante del usuario. (default is ``0.5``)
        
        beta  : `float`, optional
            Constante del usuario. (default is ``0.5``)
        """
        def factorDifus():
            return abs(particle.value - best_particle.value)/(particle.value + best_particle.value)
        
        def factorConverg():
            return abs(particle.last_position - particle.value)/(particle.last_position + particle.value)
        
        inertia = 1 - abs((alpha * (1 - factorConverg()))/((1 + factorDifus()) * (1 + beta)))
        return inertia

    #def adaptiveIWA(particle, n_iterations, best_particle, i):
    #    inertia = INERTIA FUCNTION
    #    return inertia