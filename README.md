# random_distribution
sample random distribution you should know

1. It self-learning from previous random results from the same node created.
2. Time varies is an external variable for random functions for some time ago, they discovered that random functions should not include time as a parameter it creates some shared distribution value as result. 🧸💬 If we want random from a single node or remote program we often use time % value that is because the client connects with one source or the clock.
3. 🐑💬 Hardware clock is accurate but the random function has more distribution ranges because it is accurate from the design function on varying current information and power input ( bus clock and response ), some techniques measure the running time of function response is blocked by time trying but can you vary time response with hardware⁉️ ( as reading about some security doors and rubic )

## Sample codes ##

To prove we create some vary inputs with different parameters value from ```two random.normal```, ```datetime``` and ```coefficient values``` result of 100 times run and 1,000 times run indicated how the networks learn by itself.

```
import tensorflow as tf

from datetime import datetime

import matplotlib.pyplot as plt

dict_action = { "action" : [], "value" : [] }

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
	temp = tf.math.multiply(temp, tf.constant([ coeff_01, coeff_02, coeff_03, coeff_04, 
                                              coeff_05, coeff_06 ], shape=(6, 1), dtype=tf.float32))
	temp = tf.nn.softmax(temp[0])
	action = int(tf.math.argmax(temp))

	return action
```

## Running 100 times see how different by short time run value frequencies different ##

Start from 0 - 100 from 6 samples, different frequency value return from the function ```random_action_learning( )``` can be observed and the amount is acceptable.

![name-of-you-image](https://github.com/jkaewprateep/random_distribution/blob/main/Figure_1.png)

## Running 1000 times see how different by short time run value frequencies different ##

Start from 0 - 1,000 from 6 samples, different frequency value return from the function ```random_action_learning( )``` not any significants and the amount is to be accepted. The ```random_action_learning( )``` learn by itself to balance the output in frequiency value.

![name-of-you-image](https://github.com/jkaewprateep/random_distribution/blob/main/Figure_2.png)

## Running over 1000 times ##

The Flappy bird player learn to balance from its environment inputs.

![name-of-you-image](https://github.com/jkaewprateep/random_distribution/blob/main/FlappyBird_small.gif)

## Running over 50 times ##

The Ping-Pong player learn to balance from its environment inputs.

![name-of-you-image](https://github.com/jkaewprateep/random_distribution/blob/main/Pong%20Game.gif)

## Files and Directory ##

File name | Description |
--- | --- |
sample.py | sample codes |
Figure_1.png | running 100 times |
Figure_2.png | running 1,000 times |
FlappyBird_small.gif | running over 1,000 times, Flappy Bird |
Pong Game.gif | running over 1,000 times, Pong games |
README.md | readme file |
