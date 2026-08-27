import matplotlib
import matplotlib.pyplot as plt
import cairo

matplotlib.use("Cairo")
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.plot(x, y)
plt.title("Test Plot")

# Display or save
plt.show()  # Opens window or renders in terminal
# plt.savefig("test.png")  # Uncomment to test export
