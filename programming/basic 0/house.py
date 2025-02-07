n,k=map(int,input().strip().split())
i,price=1,200
while i<=20:
    if i>1:
        price=price*(1+k/100)
    if n*i>=price:
        break
    i=i+1
if i<=20:
    print(i)
else:
    print("Impossible")