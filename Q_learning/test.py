eps = 1
for i in range(100):
    eps = max(eps*0.9, 0.3)
print(eps)