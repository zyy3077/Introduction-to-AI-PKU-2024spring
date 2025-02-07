n,m=map(int,input().strip().split())
map=[list(input().strip()) for _ in range(n)]
dx=[1,0,-1,0]
dy=[0,1,0,-1]
res=100000
visited=[[[0]*m for _ in range(n)] for _ in range(4)]
def walk(x,y,time,direction):
    #print("walk:",x,y)
    global res
    if time>=res or visited[direction][x][y]:
        return
    if direction>=0:
        visited[direction][x][y]=1
    if map[x][y]=='E':
        res=min(res,time)
        return
    for k in range(4):
        if map[x+dx[k]][y+dy[k]]=='#' or (direction>=0 and k==direction+2 or k==direction-2):
            continue
        while map[x][y]!='#':
            x,y=x+dx[k],y+dy[k]
        x,y=x-dx[k],y-dy[k]
        walk(x,y,time+1,k)
for i in range(n):
    for j in range(m):
        if map[i][j]=='S':
            walk(i,j,0,-1)
            break
if res==100000:
    res=-1
print(res)
        

        