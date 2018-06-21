import shutil

fname="trainFiles_Signal.txt"
#fname="trainFiles_Background.txt"

folder= '/work/data/multicrab_Training/'
with open(fname) as f:
    content = f.readlines()
content = [x.strip() for x in content]

new_content=[]
for file in content:
	name = file.split('/')[-1]
	name = folder+name
	new_content.append(name)


#Record new locations if there is a need to reverse the moving easily
file = open('new_trainFiles.txt','w')
for item in new_content:
        file.write("%s\n" % item)
file.close()

for i in range(len(content)):
	shutil.move(content[i], new_content[i])

