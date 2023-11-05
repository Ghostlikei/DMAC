from algorithm.pyramid import Pyramid

N, M, K = map(int, input("Type N M K: ").split())
passages = [tuple(map(int, input().split())) for _ in range(M)]

pyramid = Pyramid(N, M, K, passages)
result = pyramid.astar()

for length in result:
    print(length)

