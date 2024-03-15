from Particle import Particle
from Swarm import Swarm
from InertiaFunction import InertiaFuc
import numpy as np
import matplotlib.pyplot as plt

def funcion_objetivo(x_0, x_1):
    """
    La función de Ackley es una función de prueba comúnmente utilizada en la optimización global.
    Tiene muchas características interesantes, como múltiples mínimos locales y un único mínimo global.
    Esta función tiene un mínimo global en f(0,0)=0 y varios mínimos locales. 
    """
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x_0**2 + x_1**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x_0) + np.cos(2 * np.pi * x_1)))
    return term1 + term2 + 20 + np.exp(1)


def ejemploParticula():
    # Ejemplo creación partícula.
    part = Particle(
            num_variables = 3,
            lower_limits = [4,10,20],
            upper_limits = [-1,2,5],
            verbose     = True
            )
    # Ejemplo evaluar partícula con una función objetivo
    def funcion_objetivo(x_0, x_1, x_2):
        f= x_0**2 + x_1**2 + x_2**2
        return(f)

    part.evaluate_particle(
        objective_function = funcion_objetivo,
        optimization = "maximizar",
        verbose     = True
        )

    # Hasta que la partícula se mueva, el valor actual y mejor valor es el mismo.
    part

    # Ejemplo mover partícula
    part.move_particle(
        best_swarm_position = np.array([-1000,-1000,+1000]),
        inertia          = 0.8,
        cognitive_weight = 2,
        social_weight    = 2,
        verbose          = True
        )

    part

def ejemploEnjambre():
    # Ejemplo crear enjambre
    enjambre = Swarm(
                n_particles = 4,
                num_variables = 3,
                lower_limits  = [-5,-5,-5],
                upper_limits  = [5,5,5],
                verbose      = False
                )
    # Ejemplo evaluar enjambre
    def funcion_objetivo(x_0, x_1, x_2):
        f= x_0**2 + x_1**2 + x_2**2
        return(f)

    enjambre.evaluate_swarm(
        objective_function = funcion_objetivo,
        optimization     = "minimizar",
        verbose          = False
        )
    # Ejemplo mover enjambre
    enjambre.move_swarm(
        inertia          = 0.8,
        cognitive_weight = 2,
        social_weight    = 2,
        verbose          = True
    )

def ejemploInercia():
    # Ejemplo optimización con una función de inercia propia
        #    Ejemplo PARA NUESTRA PROPIA FUNCION INERCIAL NO ADAPTATIVA
        #    Los argumnentos siguientes sonestrictamente necesarios.
            #    def NOMBRE(inertia, n_iterations, i):
            #        inertia = FUCTION
            #        return inertia
        #   Ejemplo PARA NUESTRA PROPIA FUNCION INERCIAL ADAPTATIVA
            #   def ADAPTIVE(inertiaParticle, n_iterations, best_particle, particle, i, totalparticles):
            #       inertia = INERTIA FUCNTION
            #       return inertia

    enjambre = Swarm(
               n_particles = 50,
               num_variables  = 2,
               lower_limits  = [-5, -5,],
               upper_limits  = [5, 5],
               verbose      = False
            )
    #InertiaFuc.Personalization(InertiaFuc.NoLinearIW, Weightmin = 0.1, Weightmax = 0.7, alpha =2)
    enjambre.optimize(
        objective_function = funcion_objetivo,
        optimization     = "minimizar",
        n_iterations    = 4,
        inertia          = 0.729844,
        reduce_inertia    = True,
        inertia_function =  InertiaFuc.Personalization(InertiaFuc.SelfRegulatingIWA, eta = 2, Weightinitial = 1.2, Weightfinal = 0.4),
        adaptative_inertia = True,
        cognitive_weight_C1   = 1,
        social_weight_C2      = 2,
        diversity_weight_C3   = 2,
        diversity_control   = "PositionAndAverageOfVelocities",
        early_stopping  = True,
        stopping_rounds    = 10,
        stopping_tolerance = 10**-2,
        verbose          = False
    )
    # print(enjambre)

    # Evolución de la optimización
    fig = plt.figure(figsize=(6,4))
    enjambre.resultados_df['mejor_valor_enjambre'].plot()
    plt.show()
    # Contour plot función objetivo
    #x_0 = np.linspace(start = -5, stop = 5, num = 100)
    #x_1 = np.linspace(start = -5, stop = 5, num = 100)
    #x_0, x_1 = np.meshgrid(x_0, x_1)
    #z = funcion_objetivo(x_0, x_1)
    # plt.contour(x_0, x_1, z, 35, cmap='RdGy')
    #enjambre.animatePSO(x_0, x_1, z)

if __name__ == "__main__":
    ejemploInercia()