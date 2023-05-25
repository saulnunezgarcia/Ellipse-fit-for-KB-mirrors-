import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ellipse import LsqEllipse
from sklearn.metrics import mean_squared_error, r2_score

#Functions to process the data

def removecomas(df1):
    #To convert numbers like 6,3124 to 6.3124
    for i in range(df1.shape[0]):
        for j in range(df1.shape[1]):
            df1.values[i][j] = float(df1.values[i][j].replace(',','.'))
    return df1

def distance(df1,x_min,x_max):
    #Introduce a new column with the dimensions of the mirror
    df1['Mirror distances'] = np.linspace(x_min,x_max,np.array(df1['01DIST1 [mm]']).shape[0])
    return df1

def datatonumeric(dataframe,x_min,x_max):
    #Reads the csv from the confocal and returns a dataframe with the distance of the mirror as index and
    # the distances measured by the confocal as a column
    df1 = pd.read_csv(dataframe,sep = ';',skiprows=6)
    df1 = df1.drop(['02DIST1 [mm]','02DIST1_PEAK [mm]','02INTENSITY1 [%]','01DIST1_PEAK [mm]','01INTENSITY1 [%]','Zeitstempel [ms]'],axis = 1)
    df1 = removecomas(df1)
    df1 = distance(df1,x_min,x_max).set_index('Mirror distances')
    x = np.array(df1.index)
    y = np.array([float(i) for i in df1.values])
    return df1.rename(columns = {'01DIST1 [mm]':'M1'}),x,y

#Functions for the fit

def elipse(x,y):
    #Returns the coefficients of the ellipse fitting, the parameters and the object model
    X = np.array(list(zip(x, y)))
    obj = LsqEllipse()
    obj.fit(X)
    #obj.as_parameters() gives the center, width, height, phi of the ellipse
    #obj.coefficients gives the coefficients A,B,C,D,E,F in the form Ax**2 + Bxy + Cy**2 + Dx + Ey + F
    return obj,obj.coefficients, obj.as_parameters()

def ellipseplot(x,y,parameters):
    #Returns the plot of the mirror with the fitting of the elipse
    #Parameters must be a variable that has the center, width, height and phi of an ellipse
    center, width, height, phi = parameters
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot()
    ax.plot(x, y, 'r*', zorder=1)
    ellipse = Ellipse(
        xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Ellipse fit', zorder=110
    )
    ax.add_patch(ellipse)
    plt.xlabel('Length of the mirror')
    plt.ylabel('Depth')
    plt.legend()
    plt.show()
    return

def predicciones(x,y):
    #To get a dataframe with the predictions made by the model
    object_ellipse,_,_ = elipse(x,y)
    predictions_filtered = np.array([None])
    t = 10000
    while predictions_filtered.shape[0] != y.shape[0]:
        t+=1
        predictions = pd.DataFrame(object_ellipse.return_fit(n_points=t),columns = ['x','y'])
        predictions_filtered = predictions.loc[predictions['y']>=min(y)]
    predictions_filtered = predictions_filtered.sort_values(by = 'x').reset_index().drop('index',axis = 1)
    return predictions_filtered, print('The MSE is {:.5f}'.format(mean_squared_error(y,predictions_filtered['y']))), print('The r^2 value is {:.5f}'.format(r2_score(y,predictions_filtered['y'])))


#Here we charge the data from an excel datasheet, then we plot the profile of the mirror and later we plot the
#fitting of the mirror with an ellipse
df,x,y = datatonumeric('b15.csv',0,80)
x = x*10**-1 #to convert to cm
y = y*10**-1 #to convert to cm

A,B,C,D,E,F = elipse(x,y)[1] #To get the equation of the ellipse in the form Ax**2 + Bxz + Cz**2 + Dx + Ez + F = 0

plt.plot(df)
plt.xlabel('Length of the mirror')
plt.ylabel('Depth')
ellipseplot(x,y,elipse(x,y)[2])
plt.show()
print('The model for the ellipse is {:f} x^2 + {:f} xy + {:f} y^2 + {:f} x + {:f} y + {:f} = 0 \n'.format(A,B,C,D,E,F))
print(predicciones(x,y)[1:2])