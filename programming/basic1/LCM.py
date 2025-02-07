x,y=map(int,input().split(','))
r1,r2=x,y
while r2>0:
    r1,r2=r2,r1%r2
print(x*y//r1)