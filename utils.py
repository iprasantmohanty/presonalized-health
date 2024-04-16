import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import pickle

# Define the constants for distance calculation
MIN_VAL = [0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0.0, 0, 0.0, 0.078] # min possible values of decision variables
MAX_VAL = [82, 1, 1, 272.0, 98.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 17, 122.0, 99, 846.0, 2.42] # max possible values of decision variables
sigma_share = 0.5
alpha=1
POP_SIZE = 1000 # population size
MUT_RATE = 0.05  # mutation rate
MAX_GEN = 20  # maximum generations
f1_max = 1
f1_min = 0
f2_max = 1
f2_min = 0
#grouping for one-hot encoding
group1=[7,8,9,10]
group2=[12,13,14]


# Define the binary feature value mappings
binary_mappings = {
    1: 'yes',
    0: 'no'
}

gender_mapping = {
    1: 'male',
    0: 'female'
}

residence_type_mapping = {
    1: 'urban',
    0: 'rural'
}


# Tuple representing the feature values
#feature_tuple = best_individual #(11, 1, 0, 91.90527, 20.67798, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 79.0053, 17, 308.86574, 0.35611)

# Define the feature names
feature_names = [
    'age',
    'hypertension',
    'heart_disease',
    'avg_glucose_level',
    'bmi',
    'gender',
    'ever_married',
    'working_type',
    'residence_type',
    'smoking_status',
    'Pregnancies',
    'hypertension_cont',
    'SkinThickness',
    'Insulin',
    'DiabetesPedigreeFunction'
]


# Define the feature information as a list of tuples (feature name, index, dtype)
feature_info = [
    ('age', 0, 'int'),
    ('hypertension', 1, 'bin'),
    ('heart_disease', 2, 'bin'),
    ('avg_glucose_level', 3, 'float'),
    ('bmi', 4, 'float'),
    ('gender_Male', 5, 'bin'),
    ('ever_married_Yes', 6, 'bin'),
    ('work_type_Never_worked', 7, 'bin'),
    ('work_type_Private', 8, 'bin'),
    ('work_type_Self-employed', 9, 'bin'),
    ('work_type_children', 10, 'bin'),
    ('Residence_type_Urban', 11, 'bin'),
    ('smoking_status_formerly smoked', 12, 'bin'),
    ('smoking_status_never smoked', 13, 'bin'),
    ('smoking_status_smokes', 14, 'bin'),
    ('Pregnancies', 15, 'int'),
    ('hypertension_cont', 16, 'float'),
    ('SkinThickness', 17, 'int'),
    ('Insulin', 18, 'float'),
    ('DiabetesPedigreeFunction', 19, 'float')
]

work_type_mapping = {
    0000: 'Govt_job',
    1000: 'Never_worked',
    100: 'Private',
    10: 'Self-employed',
    1: 'children'
}

smoking_status_mapping = {
    000: 'Unknown',
    100: 'formaly smoked',
    10: 'never smoked',
    1: 'smokes'
}

# Load the saved model from the pickle file
with open('stroke_model.pickle', 'rb') as file:
    stroke_model = pickle.load(file)


# Load the saved model from the pickle file
with open('diabetes_model.pickle', 'rb') as file:
    diabetes_model = pickle.load(file)


# Problem Definition for stroke prediction
def F1(x):
  #random_instance = np.array([[60,0,0,89.22,37.80,0,0,0,1,0,0,1,0,1,0]])
  random_instance = np.array([[x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14]]])
  return round(float(stroke_model.predict_proba(random_instance)[:, 1][0]),8)

# Problem Definition for diabeties prediction
def F2(x):
  #random_instance = np.array([[4,171,72,0,0,43.6,0.479,26]])
  random_instance = np.array([[x[15], x[3], x[16], x[17], x[18], x[4], x[19], x[0]]])
  return round(float(diabetes_model.predict_proba(random_instance)[:, 1][0]),8)

def generate_population():
  # Create an empty population list
  population = []

  # Generate the population
  for _ in range(POP_SIZE):
      individual = []
      random_index1 = random.choice(group1)
      #print(random_index1)
      remaining_indices1 = [i for i in group1 if i != random_index1]
      #print(remaining_indices1)

      random_index2 = random.choice(group2)
      #print(random_index2)
      remaining_indices2 = [i for i in group2 if i != random_index2]
      #print(remaining_indices2)
      for i, (min_val, max_val) in enumerate(zip(MIN_VAL, MAX_VAL)):
          if i ==random_index1 or i==random_index2:
            if min_val==max_val:
              val=min_val
            else:
              val=1
          elif i in remaining_indices1 or i in remaining_indices2:
            if min_val==max_val:
              val=min_val
            else:
              val=0
          elif i not in group1 and i not in group2:
            if isinstance(min_val, int) or isinstance(max_val, int):
              val = np.random.randint(min_val, max_val + 1) #need to update it
            else:
              val = np.random.uniform(min_val, max_val) #need to update it
              val = round(val,5)
          individual.append(val)
      population.append(individual)

  # Convert the population list to a NumPy array
  population_array = np.array(population)

  # Print the generated population
  #print(population_array)
  return population

def calculate_fitness(population):
    return [[F1(individual), F2(individual)] for individual in population]

def crowding_sort(df,sol,n):
  # Filter the DataFrame based on sol_no values
  filtered_df = df[df['sol_no'].isin(sol)]

  # Sort the DataFrame in ascending order of 'F1'
  sorted_df = filtered_df.sort_values(by='F1')

  # Reset the index after sorting
  sorted_df.reset_index(drop=True, inplace=True)

  # Add a 'dist' column and initialize it to 0
  sorted_df['dist'] = 0

  # Set the first and last rows to contain 'infinite'
  sorted_df.loc[0, 'dist'] = np.inf
  sorted_df.loc[len(sorted_df) - 1, 'dist'] = np.inf

  F1_max = 1
  F1_min = 0.1
  for i in range(1, len(sorted_df) - 1):
      previous_value = sorted_df.at[i, 'dist']
      F1_previous = sorted_df.at[i - 1, 'F1']
      F1_next = sorted_df.at[i + 1, 'F1']

      updated_value = previous_value + (F1_next-F1_previous) / (F1_max - F1_min)
      sorted_df.at[i, 'dist'] = round(updated_value,2)

  # Sort the DataFrame in ascending order of 'F2'
  sorted_df = sorted_df.sort_values(by='F2')

  # Reset the index after sorting
  sorted_df.reset_index(drop=True, inplace=True)

  F2_max = 60
  F2_min = 0
  for i in range(1, len(sorted_df) - 1):
      previous_value = sorted_df.at[i, 'dist']
      F2_previous = sorted_df.at[i - 1, 'F2']
      F2_next = sorted_df.at[i + 1, 'F2']

      updated_value = previous_value + (F2_next-F2_previous) / (F2_max - F2_min)
      sorted_df.at[i, 'dist'] = round(updated_value,2)
  top_n_sol_no = sorted_df.sort_values(by='dist', ascending=False)['sol_no'].head(n).tolist()

  return top_n_sol_no


def sort_by_NSGA_II(population, fitness):
  p=population
  #print("5.pop size = ",len(p))
  # Determine the number of columns (based on the length of inner lists)
  num_columns = len(p[0])

  # Create a dictionary to store the data for each column
  data_dict = {f"x{i+1}": [row[i] for row in p] for i in range(num_columns)}
  #x1_values=[row[0] for row in p]
  #x2_values=[row[1] for row in p]
  # Initialize empty lists to store values
  x_values = [[] for _ in range(num_columns)]

  # Extract values from each row and append to the respective lists
  for row in p:
      for i in range(num_columns):
          x_values[i].append(row[i])

  # Combine x1 and x2 values as pairs
  #x_values_pairs = list(zip(x1_values, x2_values))

  # Initialize an empty list to store the combined pairs
  x_values_pairs = []

  # Combine x_values lists using zip
  for values in zip(*x_values):
      x_values_pairs.append(values)
  # Create the DataFrame from the dictionary
  df = pd.DataFrame(data_dict)


  f=fitness
  # Calculate the rank for each individual in population
  rank=[0 for _ in range(len(p))]  # Initialize the rank list
  F1_values=[x[0] for x in f]
  F2_values=[x[1] for x in f]
  df['F1']=F1_values
  df['F2']=F2_values
  #print(df)

  # Calculate the rank for each row
  df['rank'] = 0  # Initialize the rank column

  for i in range(len(df)):
      dominated_count = 0

      for j in range(len(df)):
          if j != i and F1_values[j] <= F1_values[i] and F2_values[j] <= F2_values[i]:
            dominated_count += 1


      df.at[i, 'rank'] = dominated_count + 1

  # Print the dataframe
  #print(df)
  # Calculate the total number of occurrences of each rank
  N = len(df)
  mu = [0] * N


  for i in range(N):
      rank = df.at[i, 'rank']
      mu[rank - 1] += 1


  df['sol_no']=0
  for i in range(N):
    df.at[i, 'sol_no']=int(i+1)


  # Print the dataframe
  #print(df)

  # Find indices with the same rank
  grouped = df.groupby('rank').apply(lambda x: x['sol_no'].tolist())

  # Print the result
  #print(grouped.tolist())

  #for rank, indices in grouped.items():
   # print(f"Rank {rank}: {indices}")

  sol_set_F=[]
  for rank, indices in grouped.items():
      sol_set_F.append(indices)
  #print(sol_set_F)

  p_new=[]
  i=0
  N=len(p)//2
  while len(p_new)+len(sol_set_F[i])<=N:
    p_new=p_new+sol_set_F[i]
    i=i+1


  #print(i)
  #print(p_new)
  #print(sol_set_F[i])
  #print("9.p size = ",len(p_new))
  if len(p_new)+len(sol_set_F[i])>N:
    p_new=p_new+crowding_sort(df,sol_set_F[i],N-len(p_new))
  #print(p_new)
  #print("8.p size = ",len(p_new))
  # Given list
  original_list = p_new*2

  # Create a list of available elements
  available_elements = original_list.copy()

  # Create 6 different tuples
  tuples_list = []
  for _ in range(len(p_new)):
      if len(available_elements) < 2:
          break
      selected_elements = random.sample(available_elements, 2)
      available_elements.remove(selected_elements[0])
      available_elements.remove(selected_elements[1])
      tuple_ = tuple(selected_elements)
      tuples_list.append(tuple_)

  #print(tuples_list)

  q_new=[]
  for tp in tuples_list:
    tp0_rank = df.loc[df['sol_no'] == tp[0], 'rank'].values[0]
    tp1_rank = df.loc[df['sol_no'] == tp[1], 'rank'].values[0]

    if tp0_rank==tp1_rank:
      #print('same rank ',crowding_sort(df,list(tp),1)[0])
      q_new.append(crowding_sort(df,list(tp),1)[0])
    else:
      #print("different rank ",tp[0] if tp0_rank<=tp1_rank else tp[1])
      q_new.append(tp[0] if tp0_rank<tp1_rank else tp[1])
  #print(q_new)

  sol_data=p_new+q_new
  #print(sol_data)


  #print(x_values_pairs)
  sorted_p = [x_values_pairs[i-1] for i in sol_data] #sorted(x_values_pairs, key=lambda x: (sol_data[x_values_pairs.index(x)]))
  #print(sorted_p)
  return sorted_p

def crossover(parent1,parent2):
  MIN = []
  MAX = []

  for i in range(len(parent1)):
      min_value = min(parent1[i], parent2[i])
      max_value = max(parent1[i], parent2[i])

      MIN.append(min_value)
      MAX.append(max_value)

  offspring=[]

  random_index1 = random.choice(group1)
  #print(random_index1)
  remaining_indices1 = [i for i in group1 if i != random_index1]
  #print(remaining_indices1)

  random_index2 = random.choice(group2)
  #print(random_index2)
  remaining_indices2 = [i for i in group2 if i != random_index2]
  #print(remaining_indices2)
  for i, (min_val, max_val) in enumerate(zip(MIN, MAX)):
      if i ==random_index1 or i==random_index2:
            if min_val==max_val:
              val=min_val
            else:
              val=1
      elif i in remaining_indices1 or i in remaining_indices2:
            if min_val==max_val:
              val=min_val
            else:
              val=0
      elif i not in group1 and i not in group2:
        if isinstance(min_val, int) or isinstance(max_val, int):
          val = np.random.randint(min_val, max_val + 1)
        else:
          val = np.random.uniform(min_val, max_val)
          val = round(val,5)
      offspring.append(val)
  return offspring


def mutation(offspring,gen_no):
  new_offspring=[]
  mut_rate=1/(gen_no+1)
  #print("mut_rate = ",mut_rate)

  for i, (min_val, max_val) in enumerate(zip(MIN_VAL, MAX_VAL)):
        #print("i = ",i)
        rand=random.random()
        #print("rand  = ",rand)
        if rand <= mut_rate:
          if i not in group1 and i not in group2:
            #print("changing i = ",i)
            if isinstance(min_val, int) or isinstance(max_val, int):
              val = np.random.randint(min_val, max_val + 1)
              #print("int val = ",val)
            elif isinstance(min_val, float) or isinstance(max_val, float):
              val = np.random.uniform(min_val, max_val)
              val = round(val,5)
              #print("float val = ",val)
            else:
              val=offspring[i]
          else:
            val=offspring[i]
        else:
          val=offspring[i]
        new_offspring.append(val)
  return new_offspring


def generate_new_population(sorted_population,gen_no):
    new_population = []
    ub=int(POP_SIZE/2)
    while len(new_population) < POP_SIZE:
        parent1 = random.choice(sorted_population[:ub])#sorted_population[0]#random.choice(sorted_population)
        parent2 = random.choice(sorted_population[:ub])#sorted_population[1]#random.choice(sorted_population)
        if not np.array_equal(parent1, parent2):
            offspring = crossover(parent1, parent2)
            new_population.append(mutation(offspring,gen_no))
    return new_population


def calculate_pareto_front(population, fitness):
    if len(population) != len(fitness):
        print(len(population),len(fitness))
        raise ValueError("Population and fitness lists must have the same length")

    pareto_front = []
    for i in range(len(fitness)):
        dominated = False
        for j in range(len(fitness)):
            if i != j and all(fj <= fi for fj, fi in zip(fitness[j], fitness[i])) and any(fj < fi for fj, fi in zip(fitness[j], fitness[i])):
                dominated = True
                break
        if not dominated:
            pareto_front.append(population[i])
    return pareto_front


def find_best_sol():
  population = generate_population()

  pareto_fronts = []  # keep track of Pareto fronts for each generation
  best_solutions = []  # keep track of best solutions for each generation

  for gen_no in range(MAX_GEN):

      fitness = calculate_fitness(population)
      sorted_population = sort_by_NSGA_II(population, fitness)

      population = generate_new_population(sorted_population, gen_no)

      # Calculate the Pareto front of this generation
      pareto_front = calculate_pareto_front(sorted_population, fitness)
      pareto_fronts.append(pareto_front)

      # Get the best individual from this generation
      best_individual = pareto_front[0]
      best_solutions.append((gen_no, best_individual, [F1(best_individual), F2(best_individual)]))

      print(f"Generation {gen_no}: Best individual is {best_individual} with fitness {[F1(best_individual), F2(best_individual)]}")


  # Calculate fitness values
  fitness_values = [F1(best_individual), F2(best_individual)]

  # Format the fitness values to display with 8 decimal places
  formatted_fitness_values = [f"{value:.8f}" for value in fitness_values]

  # Print the final best individual and its fitness values
  print(f"Final Best individual is {best_individual} with fitness {formatted_fitness_values}")

  '''
  # Plot the Pareto fronts
  for i, pareto_front in enumerate(pareto_fronts):
      color = 'red' if i == len(pareto_fronts) - 1 else 'blue'  # use a different color for the best front
      for individual in pareto_front:
          plt.scatter(F1(individual), F2(individual), color=color)

  # Highlight the best solution
  plt.scatter(F1(best_individual), F2(best_individual), color='green')
  plt.title(f'Pareto Front for population:{POP_SIZE}, max. generation:{MAX_GEN}')
  plt.xlabel('Heart Stroke')
  plt.ylabel('Diabetes')
  plt.grid(True)
  plt.show()

  #######################
  '''
  # Plot the graph between generation number and best solution
  gen_numbers = [gen for gen, _, _ in best_solutions]
  best_fitness_values = [[f1, f2] for _, _, [f1, f2] in best_solutions]
  '''
  plt.plot(gen_numbers, best_fitness_values)
  plt.title(f'Best solution in each Generation for population:{POP_SIZE}, max. gen.:{MAX_GEN}')
  plt.xlabel('Generation')
  plt.ylabel('Best Solution Fitness')
  plt.legend(['Heart Stroke', 'Diabetes'])
  plt.grid(True)
  plt.show()
  '''
  return best_individual,formatted_fitness_values


def decode_results(all_best_solutions,fit_val):
  df = pd.DataFrame(columns=feature_names)
  for best_individual in all_best_solutions:
    print(best_individual)
    feature_tuple = best_individual

    # Create a list of indices for non-"bin" features
    non_bin_indices = [index for feature, index, dtype in feature_info if dtype != 'bin']
    feature_values=[None]*len(feature_names)
    # Print the list of non-"bin" indices
    #print(non_bin_indices)
    non_bin_feature_indices=[0,3,4,10,11,12,13,14]
    for i,j in zip(non_bin_indices,non_bin_feature_indices):
      feature_values[j]=feature_tuple[i]


    bin_indices = [index for feature, index, dtype in feature_info if dtype == 'bin']
    bin_no_groups=list(set(bin_indices) - (set(group1)|set(group2)))
    for i in bin_no_groups:
      if i == 5:
        feature_values[i]=gender_mapping[feature_tuple[i]]
      elif i==11:
        feature_values[8]=residence_type_mapping[feature_tuple[i]]
      else:
        feature_values[i]=binary_mappings[feature_tuple[i]]

    wtm=""
    for i in group1:
      wtm=wtm+str(feature_tuple[i])
    #print(int(wtm))
    feature_values[7]=work_type_mapping[int(wtm)]

    ssm=""
    for i in group2:
      ssm=ssm+str(feature_tuple[i])
    #print(int(ssm))
    feature_values[9]=smoking_status_mapping[int(ssm)]


    for i in range(len(feature_names)):
      print(f"{feature_names[i]}: {feature_values[i]}, ")

    #df=df.append(pd.Series(feature_values, index=feature_names), ignore_index=True)
    new_df = pd.DataFrame([feature_values], columns=feature_names)
    # Concatenate the new DataFrame with the original DataFrame along axis 0 (rows)
    df = pd.concat([df, new_df], ignore_index=True)

    
    df['Heart Stroke'] = 0.0  # Initialize with default values
    df['Diabetes'] = 0.0

    # Iterate over the rows and update the new columns with values from the matrix
    for i in range(min(len(df), len(fit_val))):
        df.at[i, 'Heart Stroke'] = fit_val[i][0]
        df.at[i, 'Diabetes'] = fit_val[i][1]
  return df




def update_features(feature_dict):
    
    for i, entry in enumerate(feature_dict):
           
            if entry.strip().lower() in "gender":
               
                if feature_dict[entry].strip().lower() in "male":

                    # Update MIN_VAL and MAX_VAL based on gender value
                    # in MIN_VAL, MAX_VAL index 5 is for gender
                    MIN_VAL[5],MAX_VAL[5] = 1,1
                else:
                    MIN_VAL[5],MAX_VAL[5] = 0,0 
                if MIN_VAL[5]==1: ## if a person is male then no of pregencies must be 0
                  MIN_VAL[15]=0
                  MAX_VAL[15]=0
            elif entry.strip().lower() in "work type":
                
                if feature_dict[entry].strip().lower() in "never worked":
                
                    # Update MIN_VAL and MAX_VAL based on work type value
                    MIN_VAL[7:11] = [1,0,0,0]
                    MAX_VAL[7:11] = [1,0,0,0]

                elif feature_dict[entry].strip().lower() in "private":
                
                    MIN_VAL[7:11] = [0,1,0,0]
                    MAX_VAL[7:11] = [0,1,0,0]
                elif feature_dict[entry].strip().lower() in "self-employed":
                    MIN_VAL[7:11] = [0,0,1,0]
                    MAX_VAL[7:11] = [0,0,1,0]
                elif feature_dict[entry].strip().lower() in "children":
                
                    MIN_VAL[7:11] = [0,0,0,1]
                    MAX_VAL[7:11] = [0,0,0,1]
                else:
                
                    MIN_VAL[7:11] = [0,0,0,0]
                    MAX_VAL[7:11] = [0,0,0,0]

            elif entry.strip().lower() in ["hypertension", "heart disease", "ever married"]:
                
                if feature_dict[entry].strip().lower() == "yes":
                    val = 1
                else:
                    val = 0
                # Update MIN_VAL and MAX_VAL based on relevant feature
                if entry.strip().lower() in "hypertension":
                    MIN_VAL[1] = val
                    MAX_VAL[1] = val
                elif entry.strip().lower() in "heart disease":
                    MIN_VAL[2] = val
                    MAX_VAL[2] = val
                else:
                    MIN_VAL[6] = val
                    MAX_VAL[6] = val
            elif entry.strip().lower() in "residence type":
                res_value = entry.value
                if feature_dict[entry].strip().lower() in "urban":
                    MIN_VAL[11],MAX_VAL[11] = 1,1
                else:
                    MIN_VAL[11],MAX_VAL[11] = 0,0
            elif entry.strip().lower() in "smoking status":
                
                if feature_dict[entry].strip().lower() in "formaly smoked":
                    # Update MIN_VAL and MAX_VAL based on work type value
                    MIN_VAL[12:15] = [1,0,0]
                    MAX_VAL[12:15] = [1,0,0]
                elif feature_dict[entry].strip().lower() in  "never smoked":
                    MIN_VAL[12:15] = [0,1,0]
                    MAX_VAL[12:15] = [0,1,0]
                elif feature_dict[entry].strip().lower() in  "smokes":
                    MIN_VAL[12:15] = [0,0,1]
                    MAX_VAL[12:15] = [0,0,1]
                else:
                    MIN_VAL[12:15] = [0,0,0]
                    MAX_VAL[12:15] = [0,0,0]
            elif entry.strip().lower() in "age":
                MIN_VAL[0] = int(feature_dict[entry])
                MAX_VAL[0] = int(feature_dict[entry])
            else:
                print("invalid feature")
    return MIN_VAL,MAX_VAL


def cont_update_features(feature_dict):
    continuous_features = ['avg_glucose_level', 'bmi', 'Pregnancies', 'hypertension_cont', 'SkinThickness', 'Insulin',
                        'DiabetesPedigreeFunction']
    indices_to_edit = [3, 4, 15, 16, 17, 18, 19]  # Indices for which you want to edit min and max values
    range_perc=10 ## default range percentage
    for entry in feature_dict:
        for i, j in enumerate(indices_to_edit):
            if entry.strip().lower() in continuous_features[i].lower():
                if j in [15, 17]:
                    range_val=int((MAX_VAL[j]-MIN_VAL[j])*range_perc//100)
                    MIN_VAL[j]=max(MIN_VAL[j],int(feature_dict[entry])-range_val)
                    MAX_VAL[j]=min(MAX_VAL[j],int(feature_dict[entry])+range_val)
                else:
                    range_val=round(float((MAX_VAL[j]-MIN_VAL[j])*range_perc/100),5)
                    MIN_VAL[j]=max(MIN_VAL[j],float(feature_dict[entry])-range_val)
                    MAX_VAL[j]=min(MAX_VAL[j],float(feature_dict[entry])+range_val)

    return MIN_VAL,MAX_VAL