import matplotlib.pyplot as plt


# Plot the dataset
data = [(0, 0), (-1, 2), (-3, 6), (1, -2), (3, -6)]    
for point in data:
    plt.plot(point[0], point[1], marker="o", color="blue")

# Save the figure
plt.ylabel("x2")
plt.xlabel("x1")
plt.title("Plot of Points")
plt.savefig("q2a.png")
plt.show()