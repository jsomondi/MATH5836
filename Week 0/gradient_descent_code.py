import numpy as np
data = np.loadtxt('data.csv',delimiter= ',')
learning_rate = 0.0001
max_iterations = 1000
m_new = 0
b_new = 0
N= len(data)
m_gradient = 0
b_gradient = 0
tot_error = 0

for i in range(1,max_iterations+1):
    
    for k in range(len(data)):
        x = data[k,0]
        y = data[k,1]
        m_gradient += -(2/N) * x * (y - ((m_new * x) + b_new))
        b_gradient += -(2/N) * (y - ((m_new * x) + b_new))
        tot_error += (1/N) * ((y - (m_new * x + b_new))**2)
    m_new = m_new - (learning_rate * m_gradient)
    b_new = b_new - (learning_rate * b_gradient)
    print('Iteration :{}, m: {:.3f}, b :{:.3f}, total error: {:.3f}'.format(i,m_new,b_new,tot_error))
    m_gradient = 0
    b_gradient = 0
    tot_error = 0
    
    
    
        
