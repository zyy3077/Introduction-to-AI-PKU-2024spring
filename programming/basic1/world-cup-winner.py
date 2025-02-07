n=int(input().strip())
winners=set()
losers=set()
for _ in range(n):
    winner,loser=map(int,input().strip().split())
    winners.add(winner)
    losers.add(loser)
undefeated=winners-losers

if undefeated:
    print(', '.join(map(str,sorted(undefeated))))
else:
    print("None")
