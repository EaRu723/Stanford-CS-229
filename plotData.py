import matplotlib.pyplot as plt
import pandas as pd

def plot_data(x, y):
    # pretty straight forward here

    plt.scatter(x, y, marker = 'x', c = 'red')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profitability of Foodtrucks in $10,000s')
    plt.title('Food Truck Profitabliity as a Function of Population')

    plt.show()


