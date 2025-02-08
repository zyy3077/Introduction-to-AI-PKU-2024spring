n=int(input())
strings=[]
while n>0:
    str=input()
    strings.append(str)
    n=n-1
strings.sort()
for s in strings:
    print(s)