"""# Decompose ARX Modell
Every higher order system can be described as a combination of several first order systems.
* Serial coupling
* Parallel coupling
* Feedback coupling



How to find model:
1. Reducing the model order by polynomial division (only when m>n)
2. Decomposition into first order models

"""
#Before Imports
import seaborn as sns
import io
import math
import pandas as pd
import numpy as np
import pymannkendall as mk
import statsmodels.api as sm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import control as ctrl
#from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from scipy import signal

#After Imports
import control as ctrl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def draw_tf_scheme(tf_type, tf1, tf2, tf3):
    """Draws a scheme of the transfer functions using networkx,
    including TF3 in parallel, with nodes in a linear layout."""

    graph = nx.DiGraph()
    graph.add_node("Input", layer=0)
    graph.add_node("Output", layer=2)  # Output layer is now 2

    # Add TF3 in parallel
    graph.add_node("TF3", layer=1)
    graph.add_edges_from([("Input", "TF3"), ("TF3", "Output")])

    if tf_type == "Series":
        graph.add_node("TF1", layer=0)
        graph.add_node("TF2", layer=1)
        graph.add_edges_from([("Input", "TF1"), ("TF1", "TF2"), ("TF2", "Output")])
        pos = nx.multipartite_layout(graph, subset_key="layer")
        plt.title(f"Series Configuration: ({tf1} -> {tf2}) || {tf3}")

    elif tf_type == "Parallel":
        graph.add_node("TF1", layer=1)
        graph.add_node("TF2", layer=1)
        graph.add_edges_from([("Input", "TF1"), ("Input", "TF2"), ("TF1", "Output"), ("TF2", "Output")])
        pos = nx.multipartite_layout(graph, subset_key="layer")
        plt.title(f"Parallel Configuration: ({tf1} || {tf2}) || {tf3}")

    elif tf_type == "Feedback":
        graph.add_node("TF1", layer=1)
        graph.add_node("TF2", layer=1)
        graph.add_edges_from([("Input", "TF1"), ("TF1", "Output"), ("Output", "TF2"), ("TF2", "TF1")])
        pos = nx.multipartite_layout(graph, subset_key="layer")
        plt.title(f"Feedback Configuration: ({tf1} <-> {tf2}) || {tf3}")

    # Draw the graph
    nx.draw(graph, with_labels=True, node_color="skyblue", node_size=1500, pos=pos)
    plt.show()

def decomposition_model(params_a, params_b): #params_a = den = n  ;   params_b = num = m
  result = {}
  m = len(params_b)
  n = len(params_a)-1
  #print(params_a,params_b)
  #coefficients of the polynomials in decreasing powers
  par_a = np.poly1d(np.flip(params_a))
  par_b = np.poly1d(np.flip(params_b))
  #print(par_a,par_b)

  #Step 1  -  Polynomial Division
  if m >= n:
    quotient, remainder = np.polydiv(par_b, par_a)
    #quotient,remainder = deconvolve(par_b,par_a)
    #print(f'Remainder:{remainder}  Quotient:{quotient}')
    params_c = remainder
  else:
    params_c = par_b
    quotient = []
  #Q and C are in parallel
  #print("C:", params_c, "C_0=", params_c[0])
  #print(f'Order a:{len(params_a)}  b:{len(params_a)} c:{len(params_c)} Q:{len(quotient)}\n\n')

  #Step 2 - Decomposition
  #2.a - All possible configurations:
  decompositions = {}
  A1 = params_a[-2]
  A2 = params_a[-1]
  B0 = params_c[0]
  B1 = params_c[1]
  print(f'B0:{B0} B1:{B1} A1:{A1} A2:{A2}')


  #Configurations
  #Series
  s_b1, s_b2, s_a1, s_a2 = serial(B0,B1,A1,A2)
  #Parallel
  p_b1, p_b2, p_a1, p_a2 = parallel(B0,B1,A1,A2)
  #Feedback
  #fr_b1, fr_b2, fr_a1, fr_a2 = reverse_set3(B0,B1,A1,A2)
  fr_b1, fr_b2, fr_a1, fr_a2 = feedback_m(B0,B1,A1,A2)
  f_b1, f_b2, f_a1, f_a2 = feedback(B0,B1,A1,A2)
  fr2_b1, fr2_b2, fr2_a1, fr2_a2 = reverse_set31(B0,B1,A1,A2)
  decompositions = { 'Series': {'b1': s_b1, 'b2': s_b2, 'a1': s_a1, 'a2': s_a2},
                    'Parallel': {'b1': p_b1, 'b2': p_b2, 'a1': p_a1, 'a2': p_a2},
                    'Feedback': {'b1': fr2_b1, 'b2': fr2_b2, 'a1': fr2_a1, 'a2': fr2_a2}}
  print(decompositions)
  #2.b - Remove Imaginary Values
  #print("Configuration Inicial:", configurations)

  #Step 3 - Remove unstable configurations
  #if |a| > 1 ---> 1-a < 0 ---> TC < 0 (not real meaning)
  '''for types in configurations: #'Series', 'Parallel', 'Feedback'
    unstable_configurations = False
    #print("Checking unstable configuration from: {}",types)

    for param in configurations[types]: #'b1','b2','a1','a2'
      if param.startswith('a'): #or configurations[types][param] == 'a2':
        #check if "a" > 1 and save the configuration if not
        if abs(configurations[types][param]) > 1 or isinstance(configurations[types][param], complex): #second part is for complex numbers
          unstable_configurations = True #mark configuration as "impossible"
      if unstable_configurations:
        configurations[types] = {}
        result.update({f"{types}" : {'TF_1':TF_1,'TF_2':TF_2,'TC_1':TC_1,'TC_2':TC_2, 'G1':K1, 'G2':K2}})
        #print(f'Final result: {configurations}\n')
  #Probable real configurations
  for r_types in configurations:
    if configurations[r_types] == {}:
      result.update({f"{r_types}" : {'Unstable'}})
      continue
    #Initialize var
    pre_TF_1 = []
    pre_TF_2 = []
    TF_r = None
    #define the TFs
    for param in configurations[r_types]:
      if param.endswith('1'):
        pre_TF_1.append(configurations[r_types][param]) #(b1,a1)
      elif param.endswith('2'):
        pre_TF_2.append(configurations[r_types][param]) #(b2, a2)
      else:
        pass
        #print("Unkown situation for:",param,"in",r_types)
    pre_TF_1.append(1)
    pre_TF_2.append(1)
    TF_1 = ctrl.TransferFunction(pre_TF_1[0], pre_TF_1[1:][::-1])
    TF_2 = ctrl.TransferFunction(pre_TF_2[0], pre_TF_2[1:][::-1])
    #print(f"Une: {r_types}\n TF1: {TF_1} TF2: {TF_2}")


    #Step 4
    #1 - Time Constants and Gain
    poles1 = ctrl.poles(TF_1)
    TC_1 = -1 / poles1
    K1 = ctrl.dcgain(TF_1) # Method 2: Using dcgain ??? DCgain = -SSG?
    poles2 = ctrl.poles(TF_2)
    TC_2 = -1 / poles2
    K2 = ctrl.dcgain(TF_2) # Method 2: Using dcgain ??? DCgain = -SSG?

    result.update({f"{r_types}" : {'TF_1':TF_1,'TF_2':TF_2,'TC_1':TC_1,'TC_2':TC_2, 'G1':K1, 'G2':K2}})
    '''


    #Flowchart Representation
    #draw_tf_scheme(r_types, TF_1, TF_2, ctrl.tf(quotient.c,1))

  return decompositions
    #Put the two configurations together
'''    if r_types == 'Serial':
      TF_r = ctrl.series(TF_1,TF_2)
    elif r_types == 'Parallel':
      TF_r = ctrl.parallel(TF_1,TF_2)
    elif r_types == 'Feedback':
      TF_r = ctrl.feedback(TF_1,TF_2)
    print(f"{r_types} conformation: {TF_r}")'''

    #2 - chronology









#if quotient == []:
#      print("No quotient")

  #den1 =  np.insert(-parameters_a, 0, 1)
  #den = [1] + (-parameters_a).tolist() #a [1,a2,a3,...] (endogenous)
  #TF = ctrl.TransferFunction(num, den)    #define transfer function














  #Step 4


p_b = np.array([0.0007, -0.0008])
p_a = np.array([1,-1.51104, 0.57089])
decomposition_model(p_a,p_b) # Q = [0.0004, 0.033798] || R = [0.0007, -0.0008]
'''
p_b = np.array([2,1]) # x +2
p_a = np.array([4,5,9,4]) #4x³+5x²+9x+4
decomposition_model(p_a,p_b) # Q = [4,1,3] || R = [-2]

#A(x) = 2x3 + 4x2 – 6x + 8 and B(x) = x2 + 3

p_b = np.array([2,4,-6,8])
p_a = np.array([1,0,3])
decomposition_model(p_a,p_b) # Q = [2,-2] || R = [0, 8]'''


p_b = np.array([0.6,0.3,0.54])
p_a = np.array([-0.56,-0.33])
result = decomposition_model(p_a,p_b)

#Decompose all models
#conformation_result = pd.DataFrame()
conformation_result = []

for i, var_type in enumerate(known_types):  #3X *[M_334_df, M_343_df, WT_df]
  for value, dataset in enumerate(var_type): #0..17 (18x)
    print("value:",value)
    data = dataset
    na = 2
    nb = 2
    d = 0
    y_loader, ysim, parameters_a, parameters_b, SSG, tau, r_squared, aic, Rmse, yic = model_generator(data, na, nb, d)

    #Decomposition
    result = decomposition_model(-parameters_a, parameters_b)

    for coupling in result:
      a1 = result[coupling]['a1']
      a2 = result[coupling]['a2']
      b1 = result[coupling]['b1']
      b2 = result[coupling]['b2']

      if abs(a1)>1 or abs(a2)>1:
        result[coupling] = {}
        continue
    #print(f'\n\nFinal: {result}\n\n')
    #new_row = pd.DataFrame(f'{code_name[i]}_{value}')
    #Calculate TC and SSG
    for possible in result:
      if result[possible] != {} and result[possible]['a1'] != 0 and result[possible]['a2'] != 0:
        print(f'Possible: {possible}')
        #Calculate TC and SSG
        TF1 = result[possible]['b1'], result[possible]['a1']
        TF2 = result[possible]['b2'], result[possible]['a2']
        p_TF1 = ctrl.TransferFunction(result[possible]['b1'], result[possible]['a1'])
        p_TF2 = ctrl.TransferFunction(result[possible]['b2'], result[possible]['a2'])
        poles1 = ctrl.poles(p_TF1)
        poles2 = ctrl.poles(p_TF2)
        TC_1 = -1 / poles1
        TC_2 = -1 / poles2

        K1 = ctrl.dcgain(p_TF1) # Method 2: Using dcgain ??? DCgain = -SSG?
        K2 = ctrl.dcgain(p_TF2) # Method 2: Using dcgain

        #Save
        conformation_result.append({
        'Code_name': f'{code_name[i]}_{value}',
        'Possible_Configuration': possible,
        'TF1': TF1,
        'TF2': TF2,
        'TC_1': TC_1,
        'TC_2': TC_2,
        'SSG1': K1,
        'SSG2': K2
    })
        #new_row = pd.concat([new_row, pd.DataFrame({f'{code_name[i]}_{value}': {possible: {'TF1': TF1, 'TF2': TF2, 'TC_1':TC_1, 'TC_2':TC_2, 'SSG1': K1, 'SSG2': K2} }})], ignore_index=True)
        #print(new_row)

    #conformation_result.append(new_row)#= pd.concat([conformation_result, ], ignore_index=True)
conformation_result

conformation_result

import math
import cmath

# Reverse function for Set 1
def serial(b_0,b_1, a_1, a_2): # Only considers 1 value for "B"
    # b'_{0-1} and b'_{0-2} can be any pair of numbers that multiply to b_0.
    #b_0_1 = math.sqrt(b_0)  # Assume a symmetric solution for simplicity
    b_0_1 = b_0 / 0.5
    b_0_2 = b_0 / b_0_1

    # Solve the quadratic equation for a'_{1-1} and a'_{1-2}
    discriminant = a_1**2 - 4 * a_2
    if discriminant < 0:
        print("Serial: No real solution exists for a_1 and a_2. Mathematically unfeasible Conformation.")
        a_1_1 = (a_1 + cmath.sqrt(discriminant)) / 2
        a_1_2 = (a_1 - cmath.sqrt(discriminant)) / 2
        return b_0_1, b_0_2, a_1_1, a_1_2
        #raise ValueError("No real solution exists for a_1 and a_2.")

    a_1_1 = (a_1 + math.sqrt(discriminant)) / 2
    a_1_2 = (a_1 - math.sqrt(discriminant)) / 2

    return b_0_1, b_0_2, a_1_1, a_1_2

# Reverse function for Set 2
def parallel(b_0, b_1, a_1, a_2):
    # Solve for a'_{1-1} and a'_{1-2} using the quadratic equation
    discriminant = a_1**2 - 4 * a_2
    if discriminant < 0:
        print("Parallel: No real solution exists for a_1 and a_2. Mathematically unfeasible Conformation.")
        a_1_1 = (a_1 + cmath.sqrt(discriminant)) / 2
        a_1_2 = (a_1 - cmath.sqrt(discriminant)) / 2
        return 0, 0, 0, 0
        #raise ValueError("No real solution exists for a_1 and a_2.")
    else:
      a_1_1 = (a_1 + math.sqrt(discriminant)) / 2
      a_1_2 = (a_1 - math.sqrt(discriminant)) / 2


    # Solve for b'_{0-2}
    if a_1_1 != a_1_2:  # Prevent division by zero
        print(a_1_1, a_1_2)
        b_0_2 = (b_1 - b_0 * a_1_2) / (a_1_1 - a_1_2)
    else:
        raise ValueError("a'_{1-1} and a'_{1-2} are equal, division by zero.")

    # Solve for b'_{0-1}
    b_0_1 = b_0 - b_0_2

    return b_0_1, b_0_2, a_1_1, a_1_2

# Reverse function for Set 3
def feedback(b_0, b_1, a_1, a_2):
    # Solve for the denominator (d)
    if b_1 != 0:
        d = b_0 / b_1
    else:
        d = 1 + b_0

    # Solve for b'_{0-1} and b'_{0-2}
    b_0_1 = b_0 * d
    if b_0_1 != 0:
        b_0_2 = (d - 1) / b_0_1
    else:
        b_0_2 = 0

    # Solve the quadratic equation for a'_{1-1} and a'_{1-2}
    discriminant = (a_1 * d)**2 - 4 * a_2 * d
    if discriminant < 0:
      print("Feedback: No real solution exists for a_1 and a_2. Mathematically unfeasible Conformation.")
      a_1_1 = (a_1 * d + cmath.sqrt(discriminant)) / 2
      a_1_2 = (a_1 * d - cmath.sqrt(discriminant)) / 2
      #raise ValueError("No real solution exists for a_1 and a_2.")
      return b_0_1, b_0_2, a_1_1, a_1_2
    else:
      a_1_1 = (a_1 * d + math.sqrt(discriminant)) / 2
      a_1_2 = (a_1 * d - math.sqrt(discriminant)) / 2

    return b_0_1, b_0_2, a_1_1, a_1_2

def feedback_m(b_0, b_1, a_1, a_2):
  a_1_2 = b_1 / b_0
  a_1_1 = (a_2*a_1_2) / (a_1*a_1_2 - a_2)

  d = (a_1_1 + a_1_2)/a_1

  b_0_1 = (a_1_1*a_1_2*b_0) / a_2
  b_0_2 = (((b_0_1 / b_0)-1)/b_0_1)
  b_0_2 = (a_2/b_1) - (1/b_0_1)
  return b_0_1, b_0_2, a_1_1, a_1_2

def reverse_set3(b_0, b_1, a_1, a_2):
    # Iteratively solve for b'_{0-2}
    def solve_b_0_2():
        # Start with an initial guess for b'_{0-2}
        #b_0_2 = -0.5
        for b_0_2 in [-0.5,2]:
          for _ in range(100):  # Iterate to refine the solution
              b_0_1 = b_0 / (1 - b_0 * b_0_2) if (1 - b_0 * b_0_2) != 0 else 0
              if b_0_1 == 0:
                  break
              denominator = 1 + b_0_1 * b_0_2
              a_1_1 = (a_1 * denominator + cmath.sqrt((a_1 * denominator)**2 - 4 * a_2 * denominator)) / 2
              a_1_2 = (a_1 * denominator - cmath.sqrt((a_1 * denominator)**2 - 4 * a_2 * denominator)) / 2
              predicted_b_1 = (b_0_1 * a_1_2) / denominator

              if abs(predicted_b_1 - b_1) < 1e-1:  # Converged
                  print("found")
                  return b_0_1, b_0_2, a_1_1, a_1_2
              b_0_2 *= 0.95  # Adjust guess slightly
          #return -1
          raise ValueError("Failed to converge on a solution for b'_{0-2}.")

    # Solve for b'_{0-2}, b'_{0-1}, a'_{1-1}, a'_{1-2}
    b_0_1, b_0_2, a_1_1, a_1_2 = solve_b_0_2()
    #solve_b_0_2()
    return b_0_1, b_0_2, a_1_1, a_1_2

def reverse_set31(b_0, b_1, a_1, a_2):
    # Step 1: Calculate z (b'_{0-1} * b'_{0-2})
    z = (1 / b_0) - 1

    # Step 2: Solve for a'_{1-1} and a'_{1-2}
    sum_a = a_1 * (1 + z)  # a'_{1-1} + a'_{1-2}
    prod_a = a_2 * (1 + z)  # a'_{1-1} * a'_{1-2}

    discriminant_a = sum_a**2 - 4 * prod_a
    if discriminant_a < 0:
        print("Feedback2: No real solution exists for a_1 and a_2. Mathematically unfeasible Conformation.")
        a_1_1 = (sum_a + cmath.sqrt(discriminant_a)) / 2
        a_1_2 = (sum_a - cmath.sqrt(discriminant_a)) / 2
        #raise ValueError("No real solution for a'_{1-1} and a'_{1-2}.")
        #return -1
    else:
      a_1_1 = (sum_a + math.sqrt(discriminant_a)) / 2
      a_1_2 = (sum_a - math.sqrt(discriminant_a)) / 2

    # Step 3: Solve for b'_{0-1}
    b_0_1 = (b_1 * (1 + z)) / a_1_2

    # Step 4: Solve for b'_{0-2}
    b_0_2 = z / b_0_1

    return b_0_1, b_0_2, a_1_1, a_1_2         #-B1,B2 and A1 >=> A2
#B0= 0.0007 B1= -0.0008, A1= -1.511, A2= 0.5708   ===>    b_1= -0.5393,  b_2= 0.54,  a_1= -0.7558,  a_2= -0.7552
print(serial(0.002375307035356264,-0.002295340647311899 ,1, -1.511))
print(parallel(0.002375307035356264,-0.002295340647311899 ,1, -1.511))
print(reverse_set3(0.002375307035356264,-0.002295340647311899 ,1, -1.511))

print(reverse_set31(0.002375307035356264,-0.002295340647311899 ,1, -1.511))
print(feedback(0.002375307035356264,-0.002295340647311899 ,1, -1.511))
print(feedback_m(0.002375307035356264,-0.002295340647311899 ,1, -1.511))

# Test input values
b_0_1 = 0.002372
b_0_2 = -0.5
a_1_1 = 1.82553
a_1_2 = -0.82672

# Define the functions for the three sets
def set1(b_0_1, b_0_2, a_1_1, a_1_2): #Serial
    b_0 = b_0_1 * b_0_2
    b_1 = 0
    a_1 = a_1_1 + a_1_2
    a_2 = a_1_1 * a_1_2
    return b_0, b_1, a_1, a_2

def set2(b_0_1, b_0_2, a_1_1, a_1_2):#Parallel
    b_0 = b_0_1 + b_0_2
    b_1 = b_0_1 * a_1_2 + b_0_2 * a_1_1
    a_1 = a_1_1 + a_1_2
    a_2 = a_1_1 * a_1_2
    return b_0, b_1, a_1, a_2

def set3(b_0_1, b_0_2, a_1_1, a_1_2):#Feedback
    denominator = 1 + b_0_1 * b_0_2
    b_0 = b_0_1 / denominator
    b_1 = (b_0_1 * a_1_2) / denominator
    a_1 = (a_1_1 + a_1_2) / denominator
    a_2 = (a_1_1 * a_1_2) / denominator
    return b_0, b_1, a_1, a_2

# Calculate results for the test values
result_set33 = set3(0.0004072344486155723, 0.0331, -0.0501, 0.0185)
result_set1 = set1(b_0_1, b_0_2, a_1_1, a_1_2)
result_set2 = set2(b_0_1, b_0_2, a_1_1, a_1_2)
result_set3 = set3(b_0_1, b_0_2, a_1_1, a_1_2)
print("Set 1:", result_set1)
print("Set 2:", result_set2)
print("Set 3:", result_set3)
print("Personal:", result_set33)

# Test output values
b_0_1 = -0.5393
b_0_2 = 0.54
a_1_1 = -0.7558
a_1_2 = -0.7552

tolerance=1e-3

#Calculate Inputs
result_set1 = set1(b_0_1, b_0_2, a_1_1, a_1_2)
result_set2 = set2(b_0_1, b_0_2, a_1_1, a_1_2)
result_set3 = set3(b_0_1, b_0_2, a_1_1, a_1_2)


#Verify if f(input) = output
b_0, b_1, a_1, a_2 = result_set1
s_b1, s_b2, s_a1, s_a2 = serial(b_0, b_1, a_1, a_2)
if (b_0_1, b_0_2, a_1_1, a_1_2) == (s_b1, s_b2, s_a1, s_a2):
  print("TRUEEEE S")
else:
  print("Series: ",s_b1, s_b2, s_a1, s_a2)

b_0, b_1, a_1, a_2 = result_set2
p_b1, p_b2, p_a1, p_a2 = parallel(b_0, b_1, a_1, a_2)

if all(np.isclose(v1, v2, atol=tolerance) for v1, v2 in zip([-b_0_1, -b_0_2, a_1_1, a_1_2], [p_b1, p_b2, p_a1, p_a2])):
  print("Values are CLOSE")

if (b_0_1, b_0_2, a_1_1, a_1_2) == (p_b1, p_b2, p_a1, p_a2):
  print("TRUEEEE P")
else:
  print("Parallel: ",p_b1, p_b2, p_a1, p_a2, "Result",b_0_1, b_0_2, a_1_1, a_1_2)


b_0, b_1, a_1, a_2 = result_set3
fr_b1, fr_b2, fr_a1, fr_a2 = feedback_m(b_0, b_1, a_1, a_2)
f_b1, f_b2, f_a1, f_a2 = feedback(b_0, b_1, a_1, a_2)
fr2_b1, fr2_b2, fr2_a1, fr2_a2 = reverse_set31(b_0, b_1, a_1, a_2)

if (b_0_1, b_0_2, a_1_1, a_1_2) == (f_b1, f_b2, f_a1, f_a2):
  print("Feedback1")
else:
  print("Feedback1: ",f_b1, f_b2, f_a1, f_a2)

if (b_0_1, b_0_2, a_1_1, a_1_2) == (fr_b1, fr_b2, fr_a1, fr_a2):
  print("Feedback2")
else:
  print("Feedback2: ",(fr_b1, fr_b2, fr_a1, fr_a2))

if (b_0_1, b_0_2, a_1_1, a_1_2) == (fr2_b1, fr2_b2, fr2_a1, fr2_a2):
  print("Feedbackr2")
else:
  print("Feedbackr2: ",fr2_b1, fr2_b2, fr2_a1, fr2_a2)

p_b = np.array([0.0007, -0.0008])
p_a = np.array([1,-1.5110, 0.5708])

# Test Input values
b_0 = 0.0007
b_1 = -0.0008
a_1 = -1.510
a_2 = 0.570

# Calculate Outputs

s_b1, s_b2, s_a1, s_a2 = serial(b_0, b_1, a_1, a_2)
result_set1 = set1(s_b1, s_b2, s_a1, s_a2)
if (b_0, b_1, a_1, a_2) == result_set1:
  print(f"Equal Serie: {result_set1}")
else:
  print("Series: ",result_set1)

p_b1, p_b2, p_a1, p_a2 = parallel(b_0, b_1, a_1, a_2)
result_set2 = set2(p_b1, p_b2, p_a1, p_a2)
if (b_0, b_1, a_1, a_2) == result_set2:
  print(f"\nEqual Parallel: {result_set2}\n{p_b1, p_b2, p_a1, p_a2}\n")
else:
  print("Parallel: ",result_set2,p_b1, p_b2, p_a1, p_a2,"\n\n")


fr_b1, fr_b2, fr_a1, fr_a2 = feedback_m(b_0, b_1, a_1, a_2)
result_set3 = set3(fr_b1, fr_b2, fr_a1, fr_a2)
if (b_0, b_1, a_1, a_2) == result_set3:
  print(f"\nEqual Feedback(mine): {result_set3}\n")
else:
  print("Feedback(mine): ",result_set3)

f_b1, f_b2, f_a1, f_a2 = feedback(b_0, b_1, a_1, a_2)
result_set3 = set3(f_b1, f_b2, f_a1, f_a2)
if (b_0, b_1, a_1, a_2) == result_set3:
  print(f"Equal Feedback(norm): {result_set3}")
else:
  print("Feedback(norm): ",result_set3)

fr2_b1, fr2_b2, fr2_a1, fr2_a2 = reverse_set31(b_0, b_1, a_1, a_2)
result_set3 = set3(fr2_b1, fr2_b2, fr2_a1, fr2_a2)
if (b_0, b_1, a_1, a_2) == result_set3:
  print(f"Equal Feedback(31): {result_set3}")
else:
  print("Feedback(31): ",result_set3)

try:
  fr1_b1, fr1_b2, fr1_a1, fr1_a2 = reverse_set3(b_0, b_1, a_1, a_2)
  result_set3 = set3(fr1_b1, fr1_b2, fr1_a1, fr1_a2)
except:
  print("No solution")
finally:
  if (b_0, b_1, a_1, a_2) == result_set3:
    print(f"Equal Feedback(iter): {result_set3}")
  else:
      print("Feedback(iter): ",result_set3)

import networkx as nx
def draw_coupling_diagram(coupling_type, poles, residues):
    """
    Visualize the system as a block diagram for a given coupling type.

    Parameters:
    - coupling_type: str ("series", "parallel", or "feedback")
    - poles: List of poles
    - residues: List of residues
    """
    G = nx.DiGraph()

    # Nodes for each subsystem
    for i, (p, r) in enumerate(zip(poles, residues), 1):
        G.add_node(f"Subsystem {i}\nPole: {p:.2f}\nResidue: {r:.2f}")

    # Define connections based on coupling type
    if coupling_type == "series":
        edges = [(f"Subsystem {i}", f"Subsystem {i+1}") for i in range(1, len(poles))]
    elif coupling_type == "parallel":
        edges = [("Input", f"Subsystem {i}") for i in range(1, len(poles) + 1)]
        edges += [(f"Subsystem {i}", "Output") for i in range(1, len(poles) + 1)]
        G.add_node("Input")
        G.add_node("Output")
    elif coupling_type == "feedback":
        edges = [("Input", "Subsystem 1")]
        edges += [(f"Subsystem {i}", f"Subsystem {i+1}") for i in range(1, len(poles))]
        edges += [(f"Subsystem {len(poles)}", "Input")]

    # Add edges
    G.add_edges_from(edges)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G) if coupling_type != "parallel" else nx.shell_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )
    plt.title(f"{coupling_type.capitalize()} Coupling Diagram", fontsize=14)
    plt.show()

# Example poles and residues (replace with real values from decomposition)
poles = [-1, -2, -3]
residues = [0.5, 0.3, 0.2]

# Visualize each coupling type
draw_coupling_diagram("series", poles, residues)
draw_coupling_diagram("parallel", poles, residues)
draw_coupling_diagram("feedback", poles, residues)

#Using NumPy’s polydiv Function
A = [2, 4, -6, 8]
B = [1, 0, 3]

quotient, remainder = np.polydiv(A, B)

print(f"\nUsing NumPy’s polydiv Function\nQuotient: {quotient}\nRemainder: {remainder}\n")

#Using SymPy’s div Function
from sympy import symbols, div
x = symbols('x')
A = 2*x**3 + 4*x**2 - 6*x + 8
B = x**2 + 3

quotient, remainder = div(A, B, domain='QQ')

print(f"\nUsing SymPy’s div Function\nQuotient: {quotient}\nRemainder: {remainder}\n")

#Manual Polynomial Long Division
def poly_long_div(dividend, divisor):
    quotient = []
    while len(dividend) >= len(divisor):
        lead_coeff = dividend[0] / divisor[0]
        quotient.append(lead_coeff)
        dividend = [coeff - lead_coeff * div_coeff for coeff, div_coeff in zip(dividend, divisor + [0]*(len(dividend)-len(divisor)))]
        dividend.pop(0)
    return quotient, dividend

A = [2, 4, -6, 8]
B = [1, 0, 3]
quotient, remainder = poly_long_div(A, B)
print(f"\nManual Polynomial Long Division\nQuotient: {quotient}\nRemainder: {remainder}\n")

"""# 3. Classification"""

from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
known_measurements = {
    'Cell Type A': {
        'a_parameter': [0.1, 0.2, 0.3, 0.4, 0.5],
        'b_parameter': [0.6, 0.7, 0.8, 0.9, 1.0],
        'c_parameter': [0.15, 0.25, 0.35, 0.45, 0.55]  # Added c_parameter
    },
    'Cell Type B': {
        'a_parameter': [1.1, 1.2, 1.3, 1.4, 1.5],
        'b_parameter': [1.6, 1.7, 1.8, 1.9, 2.0],
        'c_parameter': [1.15, 1.25, 1.35, 1.45, 1.55]  # Added c_parameter
    }
}

unknown_measurements = {
    'Unknown 1': {'a_parameter': 0.25, 'b_parameter': 0.75, 'c_parameter': 0.30},
    'Unknown 2': {'a_parameter': 1.35, 'b_parameter': 1.85, 'c_parameter': 1.40}
}

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for cell_type, measurements in known_measurements.items():
    ax.scatter(measurements['a_parameter'], measurements['b_parameter'], measurements['c_parameter'], label=cell_type)

for name, measurement in unknown_measurements.items():
    ax.scatter(measurement['a_parameter'], measurement['b_parameter'], measurement['c_parameter'], marker='x', s=100, color='red', label=name if name not in [k for k in known_measurements] else "")

ax.set_xlabel('a-parameter')
ax.set_ylabel('b-parameter')
ax.set_zlabel('c-parameter')
ax.set_title('Classification of Unknown Measurements (3D)')
ax.legend()
plt.show()

# Create scatter plot for visualization 2D
plt.figure(figsize=(8, 6))
for cell_type, measurements in known_measurements.items():
  plt.scatter(measurements['a_parameter'], measurements['b_parameter'], label=cell_type)

for name, measurement in unknown_measurements.items():
    plt.scatter(measurement['a_parameter'], measurement['b_parameter'], marker='x', s=100, color='red', label=name if name not in [k for k in known_measurements] else "")

plt.xlabel('a-parameter')
plt.ylabel('b-parameter')
plt.title('Classification of Unknown Measurements')
plt.legend()
plt.grid(True)
plt.show()


# Simple classification based on distance (replace with a more robust method)
def classify(measurement):
  distances = {}
  for cell_type, measurements in known_measurements.items():
    total_distance = 0
    for param in ['a_parameter', 'b_parameter']:
      total_distance += (measurement[param] - sum(measurements[param]) / len(measurements[param]))**2
    distances[cell_type] = total_distance

  return min(distances, key=distances.get)

for name, measurement in unknown_measurements.items():
  predicted_class = classify(measurement)
  print(f'{name}: Predicted Cell Type = {predicted_class}')




def dataloader(data, na, nb, d):

    # Initialize empty arrays for lagged features
    u = data['u']
    X = pd.DataFrame()
    y = data['y']


    # Create lagged features for input ('b' parameters)
    for i in range(0,nb):
        X['u-{}'.format(i + d)] = u.shift(i + d)
    # Create lagged features for output ('a' parameters)
    for i in range(1,na+1):
        X['y-{}'.format(i)] = y.shift(i)


    X = X.iloc[max(na,nb+d):]
    # Set the target values
    y = y[max(na,nb+d):]

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    return X, y
