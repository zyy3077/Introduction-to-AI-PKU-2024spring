n=int(input())
i=0
num=list(map(int,input().split()))
num.sort()
if n%2:
    print(num[n//2])
else:
    res=((num[n//2-1]+num[n//2])/2)
    if res==int(res):
        print(int(res))
    else:
        print(res)