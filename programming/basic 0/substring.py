s1=input().strip()
s2=input().strip()
if s1 in s2:
    print(f"{s1} is substring of {s2}")
elif s2 in s1:
    print(f"{s2} is substring of {s1}")
else:
    print("No substring")