""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym

# settings
resume = False # resume from previous checkpoint?
render = False # Wanna see it get crushed by the AI paaainnfuullly slooowwwllyy?

# hyperparameters
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
H = 200 # number of hidden layer neurons

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        # TODO: add actual backward pass logic here
        return x

class Relu:
    def __init__(self):
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x)
        x[x<0] = 0
        return x

    def backward(self, output_gradients):
        # Reduces the gradient for a given output to zero where
        # the corresponding input was negative.
        output_gradients[np.array(self.inputs) <= 0] = 0
        # Reset inputs for next batch.
        self.inputs = []
        return output_gradients

class RmsProp:
    def __init__(self, input_size, output_size, decay_rate, learning_rate):
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.cache = np.zeros((output_size, input_size))

    def optimize(self, weights, weights_gradient):
        # Update the RMSProp Cache with the new gradients average
        self.cache = self.decay_rate * self.cache + (
            1 - self.decay_rate) * weights_gradient**2
        # Update the parameters
        weights += self.learning_rate * weights_gradient / (
            np.sqrt(self.cache) + 1e-5)
        return weights

class Dense:
    def __init__(self, input_size, output_size, optimizer):
        # Initialise the weights and biases to random values (Xavier)
        self.weights = np.random.randn(
            output_size, input_size) / np.sqrt(input_size)
        self.weights_gradient = np.zeros_like(self.weights)
        self.optimizer = optimizer
        self.input = []

    def forward(self, input):
        self.input.append(input)
        output = np.dot(self.weights, input)
        return output

    def backward(self, output_gradient):
        self.weights_gradient += np.dot(output_gradient.T, np.array(self.input))
        input_gradient = np.dot(output_gradient, self.weights)
        self.input = []
        return input_gradient

    def optimize(self):
       self.weights = self.optimizer.optimize(
           self.weights, self.weights_gradient)
       self.weights_gradient = np.zeros_like(self.weights_gradient)

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, output_gradient):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)
        return output_gradient
    
    def optimize(self):
        for layer in self.layers:
          if isinstance(layer, Dense):
              layer.optimize()

if resume:
    network = Network([
        pickle.load(open('pong_2_layer_1.p', 'rb')),
        Relu(),
        pickle.load(open('pong_2_layer_2.p', 'rb')),
        Sigmoid()
    ])
else:
    network = Network([
        Dense(D, H, RmsProp(D, H, decay_rate, learning_rate)),
        Relu(),
        Dense(H, 1, RmsProp(H, 1, decay_rate, learning_rate)),
        Sigmoid()
    ])

def preprocess(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

if render:
    env = gym.make("Pong-v0", render_mode='human')
else:
    env = gym.make("Pong-v0")
observation = env.reset()

prev_x = None # used in computing the difference frame - initially set to None
output_gradient = [] # Store the grad to encourage the action taken to be taken.
rewards = [] # Store rewards for given observations and actions.

running_reward = None
reward_sum = 0
episode_number = 0

while True:
  # preprocess the observation, set input to network to be difference image
  cur_x = np.random.randn(80, 80).ravel()
  # cur_x = preprocess(observation)
  # calculates the difference between the current and previous image
  # if it's the first time, x is just a bunch of zeros
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob = network.forward(x)

  # 2 and 3 are up and down (down and up?) in OpenAI gym
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
  y = 1 if action == 2 else 0 # a "fake label"
  output_gradient.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  #observation, reward, done, info = np.zeros((210, 160, 3)), 1, True, ""
  reward_sum += reward
  rewards.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
  
  if done: # an episode finished - an episode is a collection of games.
    episode_number += 1

    # Gradients of the error with respect to the output of the network
    epdlogp = np.vstack(output_gradient)
    epr = np.vstack(rewards)
    output_gradient, rewards = [], [] # reset array memory
    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)

    # Update the gradients
    network.backward(epdlogp)

    # Tweak the parameters
    if episode_number % batch_size == 0:
      network.optimize()
      print('Parameters updated.')

    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward) )
    if episode_number % 100 == 0 and resume:
      pickle.dump(network[0], open('pong_2_layer_1.p', 'wb'))
      pickle.dump(network[2], open('pong_2_layer_2.p', 'wb'))
      print('Progress saved.')

    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None
