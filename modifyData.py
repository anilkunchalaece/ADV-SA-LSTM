"""
This script is used to add pose+orig_cam to the data
original data only have pose+trans
but It looks like vibe is not working with trans, so I will try with orig_cam parameters

"""
import json
import numpy as np
import os

srcFileName = "behave_out.json"
desFileName = F"modified_{srcFileName}"

with open(srcFileName) as fd :
    data = json.load(fd)

new_data = []
for d in data :
    d["pose_orig_cam"] = [d["pose"][i]+d["orig_cam"][i] for i,v in enumerate(d["pose"])]
    new_data.append(d)

for k in new_data[4].keys() :
    print(F"{k}  => {np.array(new_data[4][k]).shape}")

# with open(desFileName,'w') as fd :
#     json.dump(new_data,fd)
i = 0
for v in new_data :
    i = i + 1
    fileName = os.path.join('data','behave',f'{i}.json')
    with open(fileName,'w') as fd :
        json.dump(v,fd)