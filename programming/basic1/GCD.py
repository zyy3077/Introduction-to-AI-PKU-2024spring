n=int(input().strip())
while n>0:
    x,y=map(int,input().strip().split())
    gcd=x
    while y>0:
        gcd,y=y,gcd%y
    print(gcd)
    n=n-1