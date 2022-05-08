x_and_y = [(0, 1), (1, 3), (2, 2), (3, 5), (4, 7), (5, 8), (6, 8), (7,9), (8, 10), (9, 12)] # sample dataset
data = pd.DataFrame(x_and_y, columns=['X', 'y'])

x = data['X']
N = len(x)
ones = np.ones(N)
Xp = np.c_[ones, x]

y = data['y']
y = y.values.reshape(1, -1)

print("la w iniziale random: \n")
w = np.random.rand(1, 2)
print(w)

epochs = 10000
learning_rate = 0.01 # reduce if necessary

for epoch in range(epochs):
  y_predicted = w @ Xp.T 
  error = y - y_predicted
  L2 = 0.5*np.mean(error**2)
  gradient = -(1/N)*error @ Xp

  w = w - learning_rate*gradient
  
  if epoch%(epochs/10) == 0:  
    print(f"At step: {epoch}")
    print("the loss of function is: ", L2)

print("final multivariate parameter: \n")
print(w)

figures, (ax1) = plt.subplots(1)
ax1.scatter(x, y)
print("x: \n", x)
print("the prediction of y_predicted:", y_predicted)
result= y_predicted.T
print("final results: \n", result)
ax1.plot(x, result)

