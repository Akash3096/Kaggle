# import tensorflow as tf
# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
#
# with tf.Session() as sess:
#     print(sess.run(c))

# import tensorflow
# import matplotlib
# # matplotlib.use('GTKAgg')

# import matplotlib.pyplot as plt
# import numpy as np

# from tensorflow.contrib.eager.python import tfe
# import tensorboard
# tfe.enable_eager_execution()
#
# print("Tensorflow Imported")
# plt.plot(np.arange(100))
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Make a fake dataset:
# height = [3, 12, 5, 18, 45]
# bars = ('A', 'B', 'C', 'D', 'E')
# y_pos = np.arange(len(bars))
# print(y_pos)
# # [0 1 2 3 4]
# # Create bars
# plt.bar(y_pos, height)
#
# # Create names on the x-axis
# plt.xticks(y_pos, bars)

# Show graphic
# plt.show()

# from scipy import stats
# l = [ 1,2 ,4, 5,6,2,4,2]
# mod = stats.mode(l)
# print(int(mod[0]))

# import xml.etree.ElementTree as etree
# xml = '''<feed xml:lang='en'>
#     <title>HackerRank</title>
#     <subtitle lang='en'>Programming challenges</subtitle>
#     <link rel='alternate' type='text/html' href='http://hackerrank.com/'/>
#     <updated>2013-12-25T12:00:00</updated>
# </feed>'''
# tree = etree.ElementTree(etree.fromstring(xml))
# root = tree.getroot()
# ch = root.getchildren
# print(tree)

# from statistics import median
#
# n = int(input())
# arr = [int(x) for x in input().split()]
# arr.sort()
# t = int(len(arr) / 2)
# if len(arr) % 2 == 0:
#     L = arr[:t]
#     U = arr[t:]
# else:
#     L = arr[:t]
#     U = arr[t + 1:]
#
# print(int(median(L)))
# print(int(median(arr)))
# print(int(median(U)))

# from statistics import median
# n = 9
# num = [3,7,8,5,12,14,21,13,18]
# num = sorted(num)
# t = int(len(num)/2)
# if n%2 == 0:
#     L = num[:t]
#     U = num[t:]
# else:
#     L = num[:t]
#     U = num[t+1:]
# # print(L, U)
# print(int(median(L)))
# print(int(median(num)))
# print(int(median(U)))

# def fact(n):
#     return 1 if n == 0 else n*fact(n-1)
#
# def comb(n, x):
#     return fact(n) / (fact(x) * fact(n-x))
#
# def b(x, n, p):
#     return comb(n, x) * p**x * (1-p)**(n-x)
#
# l, r = list(map(float, input().split(" ")))
# odds = l / r
# print(round(sum([b(i, 6, odds / (1 + odds)) for i in range(3, 7)]), 3))


# # import PuLP
# from pulp import *
#
# # Create the 'prob' variable to contain the problem data
# prob = LpProblem("The Miracle Worker", LpMaximize)
#
# # Create problem variables
# x=LpVariable("Medicine_1_units",0,None,LpInteger)
# y=LpVariable("Medicine_2_units",0, None, LpInteger)
# z=LpVariable("Medicine_2_units",0, None, LpInteger)
#
# # The objective function is added to 'prob' first
# prob += (x / (y + z)) + (y / (x + z)) + (z / (y + x))
#
# # The problem data is written to an .lp file
# prob.writeLP("MiracleWorker.lp")
#
# # The problem is solved using PuLP's choice of Solver
# prob.solve()

from sympy.solvers import solve
from sympy import Symbol, linsolve, solveset, Eq

# x = Symbol('x')
# y = Symbol('y')
# z = Symbol('z')
# print(ans)

# from sympy import *
# x, y, z = symbols(['x', 'y', 'z'])
# system = [
#     Eq((x / (y + z)) + (y / (x + z)) + (z / (y + x)), 4)
# ]
# soln = solve(system, [x, y, z])
# print(soln)

# import re
#
# p = re.compile(r'Dr: [^*]+', re.IGNORECASE)
# s = 'I am here to meet Dr. Akash Mehta'
# m = p.search(s)
# if m:
#     print(m.group())    # this includes Subject:
#     print(m.group(1))
