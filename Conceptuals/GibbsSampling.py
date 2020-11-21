import pandas as pd
import random
def flipCoin( p ):
    r = random.random()
    return r < p

def XgivenY(oldy):
    if(oldy==0):
        if flipCoin(0.2):
            return 1
        else:
            return 0
    else:
        if flipCoin(0.6):
            return 1
        else:
            return 0
        
def YgivenX(oldx):
    if(oldx==0):
        if flipCoin(1/7):
            return 1
        else:
            return 0
    else:
        if flipCoin(0.5):
            return 1
        else:
            return 0

x=[1]
y=[1]
for i in range(1,100):
    y.append(YgivenX(x[-1]))
    x.append(XgivenY(y[-1]))
print(len(x))
print(y)
x0y0=0
x0y1=0
x1y0=0
x1y1=0
for i in range(1,100):
        if (x[i],y[i])==(0,0):
            x0y0+=1
        if (x[i],y[i])==(0,1):
            x0y1+=1
        if (x[i],y[i])==(1,0):
            x1y0+=1
        if (x[i],y[i])==(1,1):
            x1y1+=1
            
print(x0y0/100)
print(x0y1/100)
print(x1y0/100)
print(x1y1/100)
