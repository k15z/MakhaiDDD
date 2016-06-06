fin = open("model/results.txt", "r")
fout = open("model/results.csv", "w")
fout.write("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n")
for line in fin:
    line = line.split()
    target = line[0][10:]
    index1 = int(line[1])
#    index2 = int(line[2])
#    index3 = int(line[3][:-1])

    arr = [str((1-.4)/9)]*10
    arr[index1] = str(.4)
#    arr[index2] = str(.3)
#    arr[index3] = str(.2)
    
    fout.write(target + "," + ",".join(arr)+"\n")
