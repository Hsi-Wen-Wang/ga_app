import matplotlib.pyplot as plt
import csv
import random as rd
import matplotlib.patches  as mpatches


result_filename = 'result.csv'
data_filename = 'data.csv'
data_list = []
result_list = []
with open(result_filename,newline='') as csvfile:
    result_list = list(csv.reader(csvfile))

with open(data_filename, newline='') as csvfile:
    data_list = list(csv.reader(csvfile))

job_color = {}
keys = []
values = []

rd.seed(10)
num = 0
for i in range(len(data_list)):
    keys.append(data_list[i][0] + str(num))
    num+=1
    values.append("#"+''.join(rd.choice('0123456789ABCDEF')for j in range(6)))
for key, values in zip(keys, values):
    job_color[key] = values

print(keys)
print(job_color)
print(len(result_list[2]))

fix, ax = plt.subplots(1, figsize = (16,6))

for i in range(1,len(result_list)):
    k = 1
    for j in range(int((len(result_list[i])-1)/3)):
        ax.barh(result_list[i][0], int(result_list[i][k*3]) - int(result_list[i][k*3-1]), left=int(result_list[i][k*3-1]), color = job_color[result_list[i][k*3-2][:2]])
        k+=1

patchesColor = [i for i in job_color.values()]
patchesName = [i for i in job_color.keys()]

patches = [mpatches.Patch(color=patchesColor[i], label="{:s}".format(patchesName[i]))for i in range(len(job_color.keys()))]

plt.legend(handles = patches, loc=4)

#設定XY軸標籤
plt.xlabel("Processing Time/Hr", fontsize=15)
plt.ylabel("Processing Machine", fontsize=15)
plt.savefig("gantte.jpg")
plt.show()
