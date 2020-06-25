#Integração de Monte Carlo.
#Usaremos integração simples, sem artifícios de redução de variância.
import concurrent.futures
import time, random         
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import scipy.stats
import math

#Função para calcular o intervalo de confiança
#Código retirado de https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def interval_conf(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

#Varias funções
def normal_pad(aux_x):
        return( (1/math.sqrt(2*math.pi) * math.e**(-(aux_x**2)/2)) )

def sine(aux_x):
    return(math.sin(aux_x))

def fun_elaborada(aux_x):
    return(1/(1 + math.sinh(2*aux_x) + (math.log(aux_x)**2)))

def fun_elaborada2(aux_x):
    return(1/(aux_x**2 + math.sin(aux_x)))

#Função para o calculo da estimativa da integral de MC, com argumentos da função a ser calculada
#, o número de iterações, o limite inferior da integral(a) e superior(b).
def integral_monte_carlo(fun, aux_it_num, a , b):
    vals = []
    for j in range(0, aux_it_num): 
        x = random.uniform(a, b) #Escolhe o valor qualquer da uniforme(a,b)
        val = fun(x)/(1/(b-a)) #Divide o valor da função pela densidade da uniforme, que é constante
        vals.append(val) 
    return np.average(vals) #A média dos valores é a estimativa da Integral.

limite_inf = float(input("Insira limite inferior(finito): "))
limite_sup = float(input("Insira limite superior(finito): "))
           
#num_cores = multiprocessing.cpu_count()
num_cores = 4
qtd = 125
it_num = 100000

###Execução sequencial, rodar com num_cores = 1 é a mesma coisa.

t_seq_mc = time.time()
Int = [0]*qtd
for i in range(qtd):
    Int[i] = integral_monte_carlo(normal_pad, it_num, limite_inf, limite_sup)
    
m,il,ih = interval_conf(Int)

print("\nIntervalo de Confiança: {:0.6f}; {:0.6f}; {:0.6f}".format(il,m,ih))
print("Tempo Sequencial: {}".format(- t_seq_mc + time.time()))

t_par_mc = time.time()
#O comando Parallel cria um grupo de "trabalhadores" com num_cores de número
#o parâmetro delayed tem a função que será chamada com seus parâmetros em seguida
#E finalmente o laço para é quantas vezes a função será repetida para execução.
exe = Parallel(n_jobs=num_cores)(delayed(integral_monte_carlo)(normal_pad, it_num, limite_inf, limite_sup) for i in range(qtd))

#np.average(exe)

m,il,ih = interval_conf(exe)
print("\nIntervalo de Confiança: {:0.6f}; {:0.6f}; {:0.6f}".format(il,m,ih))

print("Tempo Paralelo: {}".format((-t_par_mc + time.time())))

#print(Int)
