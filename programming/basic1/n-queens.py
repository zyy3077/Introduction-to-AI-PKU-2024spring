n=int(input().strip())
res=0
c=[1]*n
a=[1]*(2*n-1)#i+j
b=[1]*(2*n-1)#i-j+n-1
def put(x):
    global res
    if x==n-1:
        for y in range(n):
            if c[y] and a[x+y] and b[x-y+n-1]:
                res+=1
        return
    for y in range(n):
        if c[y] and a[x+y] and b[x-y+n-1]:
            c[y],a[x+y],b[x-y+n-1]=0,0,0
            put(x+1)
            c[y],a[x+y],b[x-y+n-1]=1,1,1
    return

put(0)
print(res)
    
