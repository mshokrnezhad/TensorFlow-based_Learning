import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

noo = 1000 # noo: number of observations
x1 = np.random.uniform(low = -10, high = 10, size = (noo, 1))
x2 = np.random.uniform(low = -10, high = 10, size = (noo, 1))
x = np.column_stack((x1, x2))
noise = np.random.uniform(low = -1, high = 1, size = (noo, 1))
t = 2 * x1 - 3 * x2 + 5 + noise
np.savez("TF_Intro", inputs = x, outputs = t)

training_data = np.load("TF_intro.npz")
input_size = 2
output_size = 1
model = tf.keras.Sequential([tf.keras.layers.Dense(output_size)]) # show the function which represents the relation between inputs and outputs
model.compile(optimizer = "SGD", loss = "mean_squared_error")
model.fit(training_data["inputs"], training_data["outputs"], epochs = 100, verbose = 0)
weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]

#model.predict_on_batch(training_data["inputs"]).round(1)
#training_data["outputs"].round(1)

plt.plot(model.predict_on_batch(training_data["inputs"]).round(1), training_data["outputs"].round(1))
plt.xlabel("predicted data")
plt.ylabel("real data")
plt.show()

