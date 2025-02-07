def rome2digit(exp):
    res=0
    dic={'M':1000,'D':500,'C':100,'L':50,'X':10,'V':5,'I':1}
    for i in range(0,len(exp)-1):
        if dic[exp[i]]<dic[exp[i+1]]:
            res-=dic[exp[i]]
        else:
            res+=dic[exp[i]]
    res+=dic[exp[-1]]
    return res

def digit2rome(num):
    res=""
    rome=['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
    digit=[1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    for i in range(0,13):
        while num>=digit[i]:
            res+=rome[i]
            num-=digit[i]
    return res

s=input().strip()
if s.isdigit():
    print(digit2rome(int(s)))
else:
    print(rome2digit(s))
