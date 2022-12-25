"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Reinforcement-learning-fundamentals ( reading )

https://medium.datadriveninvestor.com/reinforcement-learning-fundamentals-469a91e40fce
https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/
http://cs229.stanford.edu/extra-notes/hoeffding.pdf
https://github.com/lilianweng/multi-armed-bandit/blob/master/solvers.py#L45
https://gdmarmerola.github.io/ts-for-bernoulli-bandit/

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
from os.path import exists

import tensorflow as tf

from pygame.constants import K_a, K_s, K_d, K_w, K_h, K_SPACE

from datetime import datetime

import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
actions = { "none_1": K_h, "left": K_a, "down": K_s, "right": K_d, "up": K_w, "action": K_SPACE }

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def random_action( ): 

	date = datetime.now()
	second = int(date.strftime("%S"))
	
	temp = tf.random.normal([1, 6], 0.2, 0.8, tf.float32).numpy()
	
	coeff_01 = second * temp[0][1]
	coeff_02 = second * temp[0][1]
	coeff_03 = second * temp[0][2]
	coeff_04 = second * temp[0][3]
	coeff_05 = second * temp[0][4]
	coeff_06 = second * temp[0][5]
	
	temp = tf.random.normal([1, 6], 1, 1, tf.float32)
	temp = tf.math.multiply(temp, tf.constant([ coeff_01, coeff_02, coeff_03, coeff_04, coeff_05, coeff_06 ], shape=(6, 1), dtype=tf.float32))
	temp = tf.nn.softmax(temp[0])
	action = int(tf.math.argmax(temp))

	return action
	
def random_action_learning( ): 

	date = datetime.now()
	second = int(date.strftime("%S"))
	
	temp = tf.random.normal([1, 6], 0.2, 0.8, tf.float32).numpy()
	
	coeff_01 = ( second + 0 ) 
	coeff_02 = ( second + 10 ) 
	coeff_03 = ( second + 20 ) 
	coeff_04 = ( 40 - second ) 
	coeff_05 = ( 50 - second ) 
	coeff_06 = ( 60 - second ) 
	
	temp = tf.random.normal([1, 6], 1, 1, tf.float32)
	temp = tf.math.multiply(temp, tf.constant([ coeff_01, coeff_02, coeff_03, coeff_04, coeff_05, coeff_06 ], shape=(6, 1), dtype=tf.float32))
	temp = tf.nn.softmax(temp[0])
	action = int(tf.math.argmax(temp))

	return action

dict_action = { "action" : [], "value" : [] }
	
for i in range(100):	
	action = random_action_learning( )
	dict_action['action'].append( action )
	dict_action['value'].append( i )



actions = list(dict_action["action"])
values =  list(dict_action["value"])
plt.bar(actions, values, color ='maroon',
        width = 0.4)
		
plt.xlabel("Actions")
plt.ylabel("Frequency")
plt.title("Action Frequency - 100 ( random.normal )")
plt.show()

for i in range(1000 - 100):	
	action = random_action_learning( )
	dict_action['action'].append( action )
	dict_action['value'].append( i )
	
actions = list(dict_action["action"])
values =  list(dict_action["value"])
plt.bar(actions, values, color ='maroon',
        width = 0.4)
		
plt.xlabel("Actions")
plt.ylabel("Frequency")
plt.title("Action Frequency - 1000 ( random.normal ) ")
plt.show()
