n=int(input().strip())
maskMatrix=[list(map(int,input().strip().split())) for _ in range(n)]
x,y=map(int,input().strip().split())

res=0
day=int(input().strip())
currentMatrix=[[0]*n for _ in range(n)]
currentMatrix[x][y]=1
nextMatrix=[[0]*n for _ in range(n)]
for i in range(0,n):
        for j in range(0,n):
            nextMatrix[i][j]=currentMatrix[i][j]
dx=[1,0,-1,0]
dy=[0,1,0,-1]

def beInfected(i,j):
    if currentMatrix[i][j]:
        return True
    if maskMatrix[i][j]:
        if i==0 or i==n-1 or j==0 or j==n-1:
            return False
        for k in range(0,4):
            if currentMatrix[i+dx[k]][j+dy[k]]==0 or maskMatrix[i+dx[k]][j+dy[k]]==1:
                return False
        return True
    for k in range(0,4):
        x1,y1=i+dx[k],j+dy[k]
        if x1>=0 and y1>=0 and x1<n and y1<n:
            if currentMatrix[x1][y1]==1 and maskMatrix[x1][y1]==0:
                return True
    return False

for _ in range(day):
    for i in range(0,n):
        for j in range(0,n):
            nextMatrix[i][j]=int(beInfected(i,j))
    for i in range(0,n):
        for j in range(0,n):
            currentMatrix[i][j]=nextMatrix[i][j]
            

for i in range(0,n):
        for j in range(0,n):
                res+=currentMatrix[i][j]


print(res)




