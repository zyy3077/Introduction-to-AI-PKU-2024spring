r,c=map(int,input().strip().split())
height=[list(map(int,input().strip().split()))for _ in range(r)]
dx=[1,0,-1,0]
dy=[0,1,0,-1]
res=0
dp=[[0]*c for _ in range(r)]
def dfs(x,y):
    if dp[x][y]:
        return dp[x][y]
    for i in range(4):
        x1,y1=x+dx[i],y+dy[i]
        if x1<0 or x1>=r or y1<0 or y1>=c:
            continue
        if height[x1][y1]<height[x][y]:
            dp[x][y]=max(dfs(x1,y1)+1,dp[x][y])
    dp[x][y]=max(dp[x][y],1)
    return dp[x][y]
for x in range(r):
    for  y in range(c):
        res=max(res,dfs(x,y))

print(res)