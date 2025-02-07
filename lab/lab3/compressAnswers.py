import zipfile

filenames = ["info.yaml", "FruitModel.py", 'model/attention.npy', 'autograd/BaseNode.py', 'autograd/BaseGraph.py']

f = zipfile.ZipFile("answer.zip", "w", zipfile.ZIP_DEFLATED)
for filename in filenames:
    f.write(filename)
f.close()