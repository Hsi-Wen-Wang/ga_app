import csv
import matplotlib.pyplot as plt

filename ='iterate.csv'

with open(filename,  newline='') as csvfile:
    iter_list = list(csv.reader(csvfile))

x = []
y = []

# for iter, value in enumerate(iter_list):
#     x.append(iter+1)
#     y.append(value)
for i in range(len(iter_list[0])):
    x.append(i+1)
    y.append(iter_list[0][i])

print(type(x))
print(type(y))
print(x)
print(y)
# print(iter_list[0])

# plt.figure(figsize=(16,6), dpi=100, linewidth=2)
fig, ax= plt.subplots(1, figsize = (16,6))
plt.plot(x, y, color='blue')

ax.invert_yaxis()
plt.xlabel('iter times', fontsize = '10')
plt.ylabel('total time', fontsize = '10')
plt.savefig("linechart.jpg")
plt.show()