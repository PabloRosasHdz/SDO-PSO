import random
import warnings
import numpy as np
from InertiaFunction import InertiaFuc

class Particle:
    """Esta clase representa una nueva partícula con una posición inicial definida por
    una combinación de valores numéricos aleatorios y velocidad 0. El rango
    de posibles valores para cada variable (posición) puede estar acotado. Al
    crear una nueva partícula, solo se dispone de información sobre su posición 
    inicial y velocidad, el resto de atributos están vacíos.
    
    Parameters
    ----------
    num_variables : `int`
        número de variables que definen la posición de la partícula.
        
    lower_limits : `list` or `numpy.ndarray`, optional
        límite inferior de cada variable. Si solo se quiere predefinir límites
        de alguna variable, emplear ``None``. Los ``None`` serán remplazados
        por el valor (-10**3). (default is ``None``)
        
    upper_limits : `list` or `numpy.ndarray`, optional
        límite superior de cada variable. Si solo se quiere predefinir límites
        de alguna variable, emplear ``None``. Los ``None`` serán remplazados
        por el valor (+10**3). (default is ``None``)

    verbose : `bool`, optional
        mostrar información de la partícula creada. (default is ``False``)

    Attributes
    ----------
    num_variables : `int`
        número de variables que definen la posición de la partícula.

    lower_limits : `list` or `numpy.ndarray`
        límite inferior de cada variable. Si solo se quiere predefinir límites
        de alguna variable, emplear ``None``. Los ``None`` serán remplazados por
        el valor (-10**3).

    upper_limits : `list` or `numpy.ndarray`
        límite superior de cada variable. Si solo se quiere predefinir límites
        de alguna variable, emplear ``None``. Los``None`` serán remplazados por
        el valor (+10**3).

    best_value : `numpy.ndarray`
        mejor valor que ha tenido la partícula hasta el momento.

    best_position : `numpy.ndarray`
        posición en la que la partícula ha tenido el mejor valor hasta el momento.

    value : `float`
        valor actual de la partícula. Resultado de evaluar la función objetivo
        con la posición actual.

    velocity : `numpy.ndarray`
        array con la velocidad actual de la partícula.

    Raises
    ------
    raise Exception
        si `lower_limits` es distinto de None y su longitud no coincide con
        `num_variables`.

    raise Exception
        si `upper_limits` es distinto de None y su longitud no coincide con
        `num_variables`.

    Examples
    --------
    Ejemplo creación partícula.

    >>> part = Particle(
                    num_variables = 3,
                    lower_limits = [4,10,20],
                    upper_limits = [-1,2,0],
                    verbose     = True
                    )

    """
    
    def __init__(self, num_variables, lower_limits=None, upper_limits=None,
                 verbose=False):

        # Número de variables de la partícula
        self.num_variables = num_variables
        # Límite inferior de cada variable
        self.lower_limits = lower_limits
        # Límite superior de cada variable
        self.upper_limits = upper_limits
        # Posición de la partícula
        self.position = np.repeat(None, num_variables)
        # Velocidad de la parícula
        self.velocity = np.repeat(None, num_variables)
        # Valor de la partícula
        self.value = np.repeat(None, 1)
        # Mejor valor que ha tenido la partícula hasta el momento
        self.best_value = None
        # Mejor posición en la que ha estado la partícula hasta el momento
        self.best_position = None
        # Inercia de la particula
        self.inertiaParticle = None
        # Posición anterior de la particula
        self.last_position = np.repeat(None, num_variables)
        
        # CONVERSIONES DE TIPO INICIALES
        # ----------------------------------------------------------------------
        # Si lower_limits o upper_limits no son un array numpy, se convierten en
        # ello.
        if self.lower_limits is not None \
        and not isinstance(self.lower_limits,np.ndarray):
            self.lower_limits = np.array(self.lower_limits)

        if self.upper_limits is not None \
        and not isinstance(self.upper_limits,np.ndarray):
            self.upper_limits = np.array(self.upper_limits)
        
        # COMPROBACIONES INICIALES: EXCEPTIONS Y WARNINGS
        # ----------------------------------------------------------------------
        if self.lower_limits is not None \
        and len(self.lower_limits) != self.num_variables:
            raise Exception(
                "lower_limits debe tener un valor por cada variable. " +
                "Si para alguna variable no se quiere límite, emplear None. " +
                "Ejemplo: lower_limits = [10, None, 5]"
                )
        elif self.upper_limits is not None \
        and len(self.upper_limits) != self.num_variables:
            raise Exception(
                "upper_limits debe tener un valor por cada variable. " +
                "Si para alguna variable no se quiere límite, emplear None. " +
                "Ejemplo: upper_limits = [10, None, 5]"
                )
        elif (self.lower_limits is None) or (self.upper_limits is None):
            warnings.warn(
                "Es altamente recomendable indicar los límites dentro de los " + 
                "cuales debe buscarse la solución de cada variable. " + 
                "Por defecto se emplea [-10^3, 10^3]."
                )
        elif any(np.concatenate((self.lower_limits, self.upper_limits)) == None):
            warnings.warn(
                "Los límites empleados por defecto cuando no se han definido " +
                "son: [-10^3, 10^3]."
            )

        # COMPROBACIONES INICIALES: ACCIONES
        # ----------------------------------------------------------------------

        # Si no se especifica lower_limits, el valor mínimo que pueden tomar las 
        # variables es -10^3.
        if self.lower_limits is None:
            self.lower_limits = np.repeat(-10**3, self.num_variables)

        # Si no se especifica upper_limits, el valor máximo que pueden tomar las 
        # variables es 10^3.
        if self.upper_limits is None:
             self.upper_limits = np.repeat(+10**3, self.num_variables)
            
        # Si los límites no son nulos, se reemplazan aquellas posiciones None por
        # el valor por defecto -10^3 y 10^3.
        if self.lower_limits is not None:
            self.lower_limits[self.lower_limits == None] = -10**3
           
        if self.upper_limits is not None:
            self.upper_limits[self.upper_limits == None] = +10**3
        
        # BUCLE PARA ASIGNAR UN VALOR A CADA UNA DE LAS VARIABLES QUE DEFINEN LA
        # POSICIÓN
        # ----------------------------------------------------------------------
        for i in np.arange(self.num_variables):
        # Para cada posición, se genera un valor aleatorio dentro del rango
        # permitido para esa variable.
            self.position[i] = random.uniform(
                                    self.lower_limits[i],
                                    self.upper_limits[i]
                                )
        self.last_position = self.position
        # LA VELOCIDAD INICIAL DE LA PARTÍCULA ES 0
        # ----------------------------------------------------------------------
        self.velocity = np.repeat(0, self.num_variables)

        # INFORMACIÓN DEL PROCESO (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("Nueva partícula creada")
            print("----------------------")
            print("Posición: " + str(self.position))
            print("Límites inferiores de cada variable: " \
                  + str(self.lower_limits))
            print("Límites superiores de cada variable: " \
                  + str(self.upper_limits))
            print("Velocidad: " + str(self.velocity))
            print("")

    def __repr__(self):
        """
        Información que se muestra cuando se imprime un objeto partícula.

        """

        texto = "Partícula" \
                + "\n" \
                + "---------" \
                + "\n" \
                + "Posición: " + str(self.position) \
                + "\n" \
                + "Velocidad: " + str(self.velocity) \
                + "\n" \
                + "Mejor posición: " + str(self.best_position) \
                + "\n" \
                + "Mejor valor: " + str(self.best_value) \
                + "\n" \
                + "Límites inferiores de cada variable: " \
                + str(self.lower_limits) \
                + "\n" \
                + "Límites superiores de cada variable: " \
                + str(self.upper_limits) \
                + "\n"

        return(texto)

    def evaluate_particle(self, objective_function, optimization, verbose = False):
        """Este método evalúa una partícula calculando el valor que toma la
        función objetivo en la posición en la que se encuentra. Además, compara
        si la nueva posición es mejor que las anteriores. Modifica los atributos
        valor, mejor_valor y mejor_position de la partícula.
        
        Parameters
        ----------
        objective_function : `function`
            función que se quiere optimizar.

        optimization : {'maximizar', 'minimizar'}
            dependiendo de esto, el mejor valor histórico de la partícula será
            el mayor o el menor valor que ha tenido hasta el momento.

        verbose : `bool`, optional
            mostrar información del proceso por pantalla. (default is ``False``)
          
        Raises
        ------
        raise Exception
            si el argumento `optimization` es distinto de 'maximizar' o 'minimizar'

        Examples
        --------
        Ejemplo evaluar partícula con una función objetivo.

        >>> part = Particle(
                num_variables = 3,
                lower_limits = [4,10,20],
                upper_limits = [-1,2,None],
                verbose     = True
                )

        >>> def objective_function(x_0, x_1, x_2):
                f= x_0**2 + x_1**2 + x_2**2
                return(f)

        >>> part.evaluate_particle(
                objective_function = objective_function,
                optimization     = "maximizar",
                verbose          = True
                )

        """

        # COMPROBACIONES INICIALES: EXCEPTIONS Y WARNINGS
        # ----------------------------------------------------------------------
        if not optimization in ["maximizar", "minimizar"]:
            raise Exception(
                "El argumento optimization debe ser: maximizar o minimizar"
                )

        # EVALUACIÓN DE LA FUNCIÓN OBJETIVO EN LA POSICIÓN ACTUAL
        # ----------------------------------------------------------------------
        self.value = objective_function(*self.position)
        # MEJOR VALOR Y POSICIÓN
        # ----------------------------------------------------------------------
        # Se compara el valor actual con el mejor valor histórico. La comparación
        # es distinta dependiendo de si se desea maximizar o minimizar.
        # Si no existe ningún valor histórico, se almacena el actual. Si ya existe
        # algún valor histórico se compara con el actual y, de ser mejor este
        # último, se sobrescribe.
        
        if self.best_value is None:
            self.best_value    = np.copy(self.value)
            self.best_position = np.copy(self.position)
        elif(InertiaFuc.has_multiple_elements(self.value)):
            if optimization == "minimizar":
                if InertiaFuc.pareto_dominance(self.best_value, self.best_position, maximize=False):
                    self.best_value    = np.copy(self.value)
                    self.best_position = np.copy(self.position)
            else:
                if InertiaFuc.pareto_dominance(self.best_value, self.best_position, maximize=True):
                    self.best_value    = np.copy(self.value)
                    self.best_position = np.copy(self.position)
        else:
            if optimization == "minimizar":
                if self.value < self.best_value:
                    self.best_value    = np.copy(self.value)
                    self.best_position = np.copy(self.position)
            else:
                if self.value > self.best_value:
                    self.best_value    = np.copy(self.value)
                    self.best_position = np.copy(self.position)

        # INFORMACIÓN DEL PROCESO (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("La partícula ha sido evaluada")
            print("-----------------------------")
            print("Valor actual: " + str(self.value))
            print("")

    def move_particle(self, best_swarm_position, inertia=0.729844, adaptative_inertia = False, 
                      adptativeParameters = None, cognitive_weight_C1=2, social_weight_C2=2, 
                      diversity_weight_C3 = None, diversity_control = None, AverageCurrentVelocity=None,  
                      verbose=False):
        """
        Este método ejecuta el movimiento de una partícula, lo que implica
        actualizar su velocidad y posición. No se permite que la partícula
        salga de la zona de búsqueda acotada por los límites.
        
        Parameters
        ----------
        best_swarm_position : `np.narray`
            mejor posición de todo el enjambre.

        inertia : `float`, optional
            coeficiente de inercia. (default is 0.729844)

        adaptative_inertia : `bool`, optional
            Si la inercia se calcula por particula o en el coeficiente (default `False`) 
        
        adptativeParameters : `list`
            Una lista de los parametros necesarios del enjambre que son inertia_function, n_iterations, i, self.n_particles, self.best_particle
            (default `None`) 

        cognitive_weight_C1 : `float`, optional
            coeficiente cognitivo. (default is 2)

        social_weight_C2 : `float`, optional
            coeficiente social. (default is 2)

        verbose : `bool`, optional
            mostrar información de la partícula creada. (default is ``False``)
          
        Examples
        --------
        Ejemplo mover partícula.

        >>> part = Particle(
                num_variables = 3,
                lower_limits = [4,10,20],
                upper_limits = [-1,2,None],
                verbose     = True
                )

        >>> def objective_function(x_0, x_1, x_2):
                f= x_0**2 + x_1**2 + x_2**2
                return(f)

        >>> part.evaluate_particle(
                objective_function = objective_function,
                optimization     = "maximizar",
                verbose          = True
                )

        >>> part.move_particle(
                best_swarm_position = np.array([-1000,-1000,+1000]),
                adaptative_inertia = True,
                adptativeParameters = [inertia_function, n_iterations, i, self.n_particles, self.best_particle],
                inertia          = 0.8,
                cognitive_weight_C1   = 2,
                social_weight_C2      = 2,
                verbose          = True
                )
       
        """

        # ACTUALIZACIÓN DE LA VELOCIDAD
        # ----------------------------------------------------------------------
        if adptativeParameters:
            velocity_component = inertia * self.velocity
        elif adaptative_inertia:
            inertia = adptativeParameters[0](self, n_iterations = adptativeParameters[1],
                                             best_particle = adptativeParameters[2],
                                             i = adptativeParameters[3])
            self.inertiaParticle = inertia
        velocity_component = inertia * self.velocity
        r1 = np.random.uniform(low=0.0, high=1.0, size = len(self.velocity))
        r2 = np.random.uniform(low=0.0, high=1.0, size = len(self.velocity))
        cognitive_component = cognitive_weight_C1 * r1 * (self.best_position \
                                                      - self.position)
        social_component = social_weight_C2 * r2 * (best_swarm_position \
                                                - self.position)
        new_velocity = velocity_component + cognitive_component \
                          + social_component
        self.velocity = np.copy(new_velocity)
        
        # ACTUALIZACIÓN DE LA POSICIÓN
        # ----------------------------------------------------------------------
        if diversity_control == None or adptativeParameters[3] == 0:
            self.position = self.position + self.velocity
        elif diversity_control == "RandomNoise":
            self.position = self.position + self.velocity + diversity_weight_C3 * np.random.uniform(low=0.0, high=1.0, size = len(self.velocity))
        elif diversity_control == "AverageOfVelocities":
            self.position = self.position + self.velocity + diversity_weight_C3 * np.random.uniform(low=0.0, high=1.0, size = len(self.velocity)) * AverageCurrentVelocity
        elif diversity_control == "PositionAndAverageOfVelocities":
            self.position = self.position + self.velocity + diversity_weight_C3 * np.random.uniform(low=0.0, high=1.0, size = len(self.velocity)) * (self.position*AverageCurrentVelocity)

        # COMPROBAR LÍMITES
        # ----------------------------------------------------------------------
        # Se comprueba si algún valor de la nueva posición supera los límites
        # impuestos. En tal caso, se sobrescribe con el valor del límite
        # correspondiente y se reinicia a 0 la velocidad de la partícula en esa
        # componente.
        for i in np.arange(len(self.position)):
            if self.position[i] < self.lower_limits[i]:
                self.position[i] = self.lower_limits[i]
                self.velocity[i] = 0

            if self.position[i] > self.upper_limits[i]:
                self.position[i] = self.upper_limits[i]
                self.velocity[i] = 0
        self.last_position = self.position   
        # INFORMACIÓN DEL PROCESO (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("La partícula se ha desplazado")
            print("-----------------------------")
            print("Nueva posición: " + str(self.position))
            print("") 
        return self.position, self.velocity, self.best_position
