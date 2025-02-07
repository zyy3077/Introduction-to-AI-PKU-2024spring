def print_addstring(s1,s2):
    res=[]
    i,j,carry,curr=len(s1)-1,len(s2)-1,0,0
    while i>=0 or j>=0 or carry:
        if i<0:
            n1=0
        else:
            n1=int(s1[i])
        if j<0:
            n2=0
        else:
            n2=int(s2[j])
        num=n1+n2+carry
        res.append(str(num%10))
        carry=num//10
        i,j=i-1,j-1
    res=''.join(res[::-1])
    print(res)


N=int(input().strip())
for _ in range(N):
    s1,s2=input().strip().split()
    print_addstring(s1,s2)

