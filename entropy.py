import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

#Let's say it's the number of wings on some bugs you found on a weird exoplanet.
#Weird, eh?

#so you collect some data and find that on this planet, the wing number is such and so,
#x is the type of bug and y is the number of wings and you make a distribution based on probability.
LEN = 2**6
distribution = np.random.rand(int(LEN),1)
#distribution = np.array(2000*[.001])
distribution = distribution/distribution.sum(axis=0, keepdims=1)
#distribution = [0.1, 0.9]
fig1 = plt.figure()
plt.plot(distribution)
#plt.ylim(0,1)
#plt.show()

# we define information as the log2 of surprise (reciprocal of probability), due to some
# neat stuff about logarithms, in particular that log xy = log x + log y, and log x/y = log x - log y,
# which is really cool and also sorta weird, anyway the point is that the information you get
# out of two events are additive

def shannon_entropy_distribution(data):
    s = 0.
    for x in data:
        s += -x * np.log2(x)
    return s

print(shannon_entropy_distribution(distribution))

# shannon enotropy tells us how much information IS NEEDED on average to encode this distribution.
# this tricks me up often:
# - shannon entropy measures the information in the symbols coming in on the wire
# - shannon entropy measures the information *needed* to *encode* these symbols. in log2 it's 
#   the number of bits we need. 

# hokay hokay.

def kl_divergence(P,Q):
    divergence = 0
    for p,q in zip(P, Q):
        divergence += p * (np.log2(p)/np.log2(q))

    return divergence


distribution_q = np.array(LEN * [1.0])
distribution_q = distribution_q / distribution_q.sum(axis = 0, keepdims=1)

print("KL divergence", kl_divergence(distribution, distribution_q))
#plt.plot(distribution_q)
#plt.show()

#let's use kl divergence to train our new distribution to look like the real one.
#for step in range(10000):
def step(i, line):
    global distribution, distribution_q
    modify_index = np.random.randint(LEN)
    new_distribution_q = copy.copy(distribution_q)
    new_distribution_q[modify_index] += (2 * np.random.random() - 1.0)/LEN
    new_distribution_q = new_distribution_q/new_distribution_q.sum(axis = 0, keepdims=1)

    kl = kl_divergence(distribution, distribution_q)
    kl_new = kl_divergence(distribution, new_distribution_q)
    if kl_new < kl:
        distribution_q = new_distribution_q
        print("We've done better! Score: ", kl_new)
        #print(distribution_q)
    line.set_data(range(len(distribution_q)),distribution_q)
    return line,

l, = plt.plot([], [], 'r-')
anim = animation.FuncAnimation(fig1, step, fargs=[l])
anim.save('animation.mp4', writer='imagemagick')
#plt.show()


