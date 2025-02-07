m=int(input().strip())
p1,p2,p3=0.4883,0.5383,0.7883
t1,t2=240,400
if m<=t1*p1:
    res=m/p1
elif m<=(t1*p1+(t2-t1)*p2):
    res=t1+(m-t1*p1)/p2
else:
    res=t2+(m-t1*p1-(t2-t1)*p2)/p3
print("%.2f"% res)