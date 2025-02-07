nums = list(map(int, input().strip().split()))
odd_nums = [num for num in nums if num % 2 == 1]
even_nums = [num for num in nums if num % 2 == 0]
print(abs(max(odd_nums) - min(even_nums)))