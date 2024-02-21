
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from scipy.optimize import curve_fit

import math


service_rate = 1

arrival_rate = 10

time_epochs = [0.1]

#This part is just to create the variables we use 

for i in range (41):

    if (i+2)*0.1 <4.1:

        time_epochs.append(round((i+2)*0.1,1))



def simulate (max_time, numberOfPeople):

    blockedOrNot=[]

    departure_times = []

    system_state = []

    time = 0

    times= []
    
    



    #We need to deal with the starting state first

    if numberOfPeople>0:

        for _ in range(numberOfPeople):

            service_time = np.random.exponential(scale=1/service_rate)

            departure_times.append(service_time)

            



    #print(f"the system starts with {numberOfPeople} in it")

    while time <= max_time:



        #Start by removing the ones who are finished

        if len(departure_times)>0:

            while time >= min(departure_times):

                numberOfPeople = numberOfPeople - 1

                departure_times.remove(min(departure_times))

                if (len(departure_times)==0):

                    break



        #A new person arrives, you generate his time, verify that it's not out of bounds then you save it and generate his service time  

        interarrival_time = np.random.exponential(1/arrival_rate)

        time += interarrival_time

        if time > max_time:

            break

        times.append(time)

        depature_time = time + np.random.exponential(1/service_rate)
        
        



        if (numberOfPeople>=10):

            blockedOrNot.append(1)

            system_state.append(numberOfPeople)

            #if the system is blocked, note in and note the number of people too. We want to note the number of people only when somebody arrives

        else:

            blockedOrNot.append(0)

            departure_times.append(depature_time)

            system_state.append(numberOfPeople)

            numberOfPeople += 1

            #If the person is accepted into the system, note it and add his departure time to the list of outgoing people for future rounds





    return blockedOrNot, system_state, times



def find_closest_number_position(target, number_list):

    #Just a function to find the position of the closest numbers to our time epochs. 

    closest_number = min(number_list, key=lambda x: abs(x - target))

    position = number_list.index(closest_number)

    return position



def count_appearances(target, number_list):

    #to count the number of time the system was block for a time tj in 50 replications

    count = number_list.count(target)

    return count



def probabilities (max_time, state):

    Yjs = []

    for tj in time_epochs:

        Yj = []

        for _ in range (50):

            blockedOrNot, system_state, time = simulate(max_time,state)

            i = find_closest_number_position(tj, time)

            Yj.append(blockedOrNot[i])

        #we simulate the system 50 times, then we looked the rank of the person arrived at the closest time to tj since his acceptance status 

        # has the same position in the BlockedOrNot, we can find it and build our vector

        

        Yjs.append(Yj)

        #we put our vector in a list of vectors: each row is tj and each column a replication

    real_probabilities = []



    for Yj in Yjs:

        count = count_appearances(1,Yj)

        real_probabilities.append(count/50) 

        #we calculate the real probability of founding a blocked system for each time tj. 


    
    # Calculating mean + variance of real probabilities of being blocked from system
    
    counter = 0
    total = 0
    for x in real_probabilities:
        total = total + x
        counter = counter + 1
    mean_real_probabaility =  total/counter
    
    total = 0
    for x in real_probabilities:
        total = total + ((x-mean_real_probabaility)*(x-mean_real_probabaility))
        
    standard_dev_real_probability = math.sqrt((1/counter)*total)  
    
    # From t table for 90 and 80% confidence with dof = 40:
    t_table90 = 1.684
    t_table80 = 1.303
    val = math.sqrt(40)
    plus_minus90 = (t_table90*standard_dev_real_probability)/val
    plus_minus80 = (t_table80*standard_dev_real_probability)/val
                    
    print("Starting state: ",state)
    #print("Mean: ", mean_real_probabaility)
    #print("Standard Deviation: ",standard_dev_real_probability)
    print("Chance of being blocked from system: ")
    print(f'90% Confidence Interval: {round(mean_real_probabaility,3)} +- {round(plus_minus90, 3)}.')
    print(f'80% Confidence Interval {round(mean_real_probabaility, 3)} +- {round(plus_minus80, 3)}.')
    
    
    
    ########################################

    ##Now this is the logistic regression###

    ########################################

    

    Y_matrix = np.array(Yjs)

    Y_bis = np.transpose(Y_matrix)

    Y = Y_bis.flatten()

    X_matrix = np.tile(time_epochs,50)

    X = X_matrix.reshape(-1,1)

    #All that was just transforming the data to have a 2D array in X of the same size of the 1D target array in Y. 

    model = LogisticRegression()

    model.fit(X,Y)





    X_range = np.linspace(0.1, 4.0, 20).reshape(-1, 1)

    probabilities = model.predict_proba(np.sqrt(X_range + 1))[:, 1]

    probabilities2 = model.predict_proba(X_range)[:, 1]

    probabilities3 = model.predict_proba(np.ones_like(X_range))[:, 1]

    





    ######################################################

    ##Now this is the Least Weighted Squares Regression###

    ######################################################

    #Define the Model Function:
    def func(X, a, b):

        return a / ((X + 1) ** 2) + b

    #Again, I just chose the function I taught fitted better



    # Perform weighted least squares regression

    weight = 1/probabilities2




    #curve_fit from SciPy perform the  weighted least square regression. It takes odel functio
    #,input data, output data and the weight
    params, covariance = curve_fit(func, X_range.flatten(), probabilities2, sigma=weight)



    # Obtain the fitted values by evaluating the function with the parameters (params)

    fitted_values = func(X_range.flatten(), *params)





    

    #Now the plotting

    

    plt.plot(time_epochs, real_probabilities, 'o--', color = 'black' ,label='Real Probabilities')

    plt.xlabel('time')

    plt.ylabel('p(x,t)')

    plt.title(f'Real Probabilities with starting state {state}')

    plt.ylim([0, 0.5])

    plt.legend()

    plt.scatter(X_range, probabilities, color='red',marker="o")

    plt.scatter(X_range, probabilities2, color='red',marker="o")

    plt.scatter(X_range, probabilities3, color='red',marker="o" , label='Logistic Regression')

    plt.plot(X_range, fitted_values, color='blue', linewidth=3, label='Weighted Least Squares Fit')

    plt.xlabel('Time (s)')

    plt.ylabel('Probability of System Blockage')

    plt.legend()

    plt.show()

    #plt.show()

            

def main():

    max_time = 10

    for state in range (11):

        #simulate(max_time,state)

        probabilities(max_time,state)

        #logistic_regression(max_time,state)



if __name__ == "__main__":

    main()
