import numpy as np
import csv
import statsmodels.api as sm

def get_data():
    #imports the data from csv and formats the data for the OLS and adjustment regressions
    raw = []
    with open('titanic.csv', newline='', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            raw += [row]

    #remove header    
    data = np.array(raw[1:])

    #set up all desired columns of data
    names = data[:, 0]
    male = data[:, 1] == "Male"

    age = data[:, 2].astype(int)
    young_adult = data[:, 3].astype(int)
    middle_age = data[:, 4].astype(int)
    old = data[:, 5].astype(int)
    first_class = data[:, 7].astype(int)
    survive = data[:, 8].astype(int)
    
    #returns the data in all the correct columns in the right format ready to be used
    return np.array([male, young_adult, middle_age, old, first_class]), survive, age

def run_regression(data, survive):
    #run the simply OLS regression with all chosen data
    inputs = sm.add_constant(data.T)
    model = sm.OLS(survive, inputs).fit()
    print(model.summary()) #print results
    print("Control group = female child, not first class")
    print("x1 = male")
    print("x2 = young_adult")
    print("x3 = middle_age")
    print("x4 = old")
    print("x5 = first_class")

def get_interaction_data():
    #importsthe data from csv and formats the data for the interaction variable regression
    raw = []
    with open('titanic.csv', newline='', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            raw += [row]

    #remove header           
    data = np.array(raw[1:])

    #set up all desired columns of data
    names = data[:, 0]
    male = data[:, 1] == "Male"
    
    age = data[:, 2].astype(int)
    adult = data[:, 3].astype(int) + data[:, 4].astype(int)
    old = data[:, 5].astype(int)
    adult_male = adult * male
    old_male = old * male
    
    first_class = data[:, 7].astype(int)
    survive = data[:, 8].astype(int)

    #returns the data in all the correct columns in the right format ready to be used
    return np.array([male, adult, old, adult_male, old_male, first_class]), survive, age

def run_interaction(data, survive):
    #run the OLS with the interaction variables for age and gender
    inputs = sm.add_constant(data.T)
    model = sm.OLS(survive, inputs).fit()
    print(model.summary()) #print results
    print("Control group = female child, not first class")
    print("x1 = male")
    print("x2 = adult")
    print("x3 = old")
    print("x4 = adult_male")
    print("x5 = old_male")
    print("x6 = first_class")

def run_adjustment(data, survive):
    #run the OLS with the interaction variables for age and gender
    D1_index = data[4] == 1
    D0_index = data[4] == 0

    D = data[4]

    data = data[:-1]

    #set up and prepare the two subsets of data for the treatment and control groups
    D1_data = data[:, D1_index]
    D0_data = data[:, D0_index]

    D1_survive = survive[D1_index]
    D0_survive = survive[D0_index]

    D1_inputs = sm.add_constant(D1_data.T)
    D0_inputs = sm.add_constant(D0_data.T)

    D1_model = sm.OLS(D1_survive, D1_inputs).fit()
    D0_model = sm.OLS(D0_survive, D0_inputs).fit()

    input_data = sm.add_constant(data.T)

    #ATE consists of difference of two average results 
    D1_average = np.mean(D1_model.predict(input_data))
    D0_average = np.mean(D0_model.predict(input_data))
    
    ATE = D1_average - D0_average
    print("Regression adjustment ATE:", ATE)

    #calls boostrap method to find standard error
    SE = bootstrap(input_data, survive, D)
    print("Regression adjustment ATE standard error:", SE)

def bootstrap(input_data, survive, D):
    #runs boostrap simulations to calculate ATE standard error for regression adjustment
    s = 10000 #number of times we sample
    n = 1000 #size of each sample

    ATEs = []

    print("Running bootstrap simulation; this may take a moment.")
    for i in range(s):
        #draw subsample using random selection
        indices = np.random.choice(len(input_data), n, replace=True)
        D_s = D[indices]
        X = input_data[indices]
        Y = survive[indices]

        #calculate new estimates
        D1_model = sm.OLS(Y[D_s == 1], X[D_s == 1]).fit()
        D0_model = sm.OLS(Y[D_s == 0], X[D_s == 0]).fit()

        D1_average = np.mean(D1_model.predict(X))
        D0_average = np.mean(D0_model.predict(X))

        #calculate subsample ATEand add to total
        ATE = D1_average - D0_average

        ATEs += [ATE]

    #check function to show that average ATE is similar to actual ATE
    #print(np.mean(ATEs))
    return np.std(ATEs)
    
def graph_variables(age):
    #graphs the age distrubution for explanatory purposes
    import matplotlib.pyplot as plt
    plt.hist(age, bins=25)
    plt.axvline(15.5, color="red")
    plt.axvline(30.5, color="red")
    plt.axvline(50.5, color="red")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Age Distribution of Titanic Passengers")
    plt.show()

#get the data
data, survive, age = get_data()
interaction_data, survive, age = get_interaction_data()

#run the three different analysis methods
run_regression(data, survive)
run_interaction(interaction_data, survive)
run_adjustment(data, survive)

#optional function to produce graphs showing various aspects of the data
#graph_variables(age)





